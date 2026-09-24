"""Regression tests for retry-safe analysis quota charging (#153)."""

from __future__ import annotations

import uuid
from collections import defaultdict

import pytest
import redis.exceptions

import reviewgate.app.rate_limit.limiter as limiter
from reviewgate.app.settings import AppSettings
from reviewgate.app.storage.repositories import AnalysisNaturalKey


class FakeAtomicRedis:
    """Model the atomic script's externally observable Redis operations."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.markers: set[str] = set()
        self.closed = 0
        self.calls = 0

    def eval(self, script: str, number_of_keys: int, *args: object) -> int:
        assert number_of_keys == 3
        assert 'redis.call("EXISTS", KEYS[3])' in script
        assert 'redis.call("SET", KEYS[3]' in script
        installation, repository, marker, inst_cap, repo_cap, ttl = args
        assert isinstance(installation, str)
        assert isinstance(repository, str)
        assert isinstance(marker, str)
        assert ttl == 3 * 24 * 60 * 60
        self.calls += 1

        if marker in self.markers:
            return 0
        if self.counters[installation] >= inst_cap:
            return 1
        if self.counters[repository] >= repo_cap:
            return 2

        self.counters[installation] += 1
        self.counters[repository] += 1
        self.markers.add(marker)
        return 0

    def close(self) -> None:
        self.closed += 1


def _key(*, pull: int = 1, metadata: str = "metadata") -> AnalysisNaturalKey:
    return AnalysisNaturalKey(
        repository_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        pull_number=pull,
        head_sha="sha",
        config_hash="config",
        pr_metadata_hash=metadata,
    )


def _charge(settings: AppSettings, key: AnalysisNaturalKey) -> str:
    return limiter.check_analysis_rate_limits(
        settings,
        github_installation_id=17,
        github_repository_id=23,
        analysis_key=key,
    )


def test_retry_charges_once_but_distinct_analysis_charges_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = FakeAtomicRedis()
    monkeypatch.setattr(limiter, "connect_redis", lambda _settings: redis_client)
    settings = AppSettings(redis_url="redis://localhost:6379/0")

    assert _charge(settings, _key()) == "ok"
    assert _charge(settings, _key()) == "ok"
    assert _charge(settings, _key(metadata="edited-description")) == "ok"

    assert redis_client.counters[limiter._installation_counter_key(17)] == 2
    assert redis_client.counters[limiter._repository_counter_key(23)] == 2
    assert len(redis_client.markers) == 2
    assert redis_client.closed == 3


def test_existing_charge_remains_allowed_when_installation_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = FakeAtomicRedis()
    monkeypatch.setattr(limiter, "connect_redis", lambda _settings: redis_client)
    settings = AppSettings(redis_url="redis://localhost:6379/0")

    assert _charge(settings, _key()) == "ok"
    inst_key = limiter._installation_counter_key(17)
    repo_key = limiter._repository_counter_key(23)
    redis_client.counters[inst_key] = 500

    assert _charge(settings, _key()) == "ok"
    assert _charge(settings, _key(pull=2)) == "installation_exceeded"
    assert redis_client.counters[inst_key] == 500
    assert redis_client.counters[repo_key] == 1
    assert len(redis_client.markers) == 1


def test_repository_rejection_does_not_leak_installation_quota_or_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = FakeAtomicRedis()
    monkeypatch.setattr(limiter, "connect_redis", lambda _settings: redis_client)
    settings = AppSettings(redis_url="redis://localhost:6379/0")

    repo_key = limiter._repository_counter_key(23)
    inst_key = limiter._installation_counter_key(17)
    redis_client.counters[repo_key] = 100

    assert _charge(settings, _key()) == "repository_exceeded"
    assert redis_client.counters[inst_key] == 0
    assert redis_client.counters[repo_key] == 100
    assert not redis_client.markers

    redis_client.counters[repo_key] = 99
    assert _charge(settings, _key()) == "ok"
    assert redis_client.counters[inst_key] == 1
    assert redis_client.counters[repo_key] == 100
    assert len(redis_client.markers) == 1


def test_retry_across_utc_midnight_is_not_recharged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = FakeAtomicRedis()
    monkeypatch.setattr(limiter, "connect_redis", lambda _settings: redis_client)
    settings = AppSettings(redis_url="redis://localhost:6379/0")
    day = ["2026-09-19"]
    monkeypatch.setattr(limiter, "_utc_day_bucket", lambda: day[0])

    assert _charge(settings, _key()) == "ok"
    day[0] = "2026-09-20"
    assert _charge(settings, _key()) == "ok"
    assert sum(redis_client.counters.values()) == 2
    assert len(redis_client.markers) == 1


def test_atomic_redis_failure_remains_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenRedis:
        closed = False

        def eval(self, *_args: object) -> int:
            raise redis.exceptions.ConnectionError("temporary Redis outage")

        def close(self) -> None:
            self.closed = True

    client = BrokenRedis()
    monkeypatch.setattr(limiter, "connect_redis", lambda _settings: client)
    settings = AppSettings(redis_url="redis://localhost:6379/0")

    assert _charge(settings, _key()) == "ok"
    assert client.closed is True
