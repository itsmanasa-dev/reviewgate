"""Worker-level regression tests for issue #153 quota accounting."""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reviewgate.app.github.client import GitHubRestError
from reviewgate.app.storage.repositories import parse_analysis_job_natural_key


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import reviewgate.app.analysis.jobs as jobs

    monkeypatch.setenv(
        "REVIEWGATE_REDIS_URL",
        "redis://127.0.0.1:6379/0",
    )

    payload = {
        "github_installation_id": 9001,
        "github_repository_id": 4242,
        "reviewgate_repository_id": str(uuid.uuid4()),
        "reviewgate_pull_number": 17,
        "reviewgate_head_sha": "sha-123",
        "reviewgate_config_hash": "config-123",
        "reviewgate_pr_metadata_hash": "metadata-123",
    }

    session = MagicMock()
    cache = MagicMock(return_value=None)
    begin = MagicMock(return_value=(uuid.uuid4(), "created"))
    resolve = MagicMock(
        return_value=SimpleNamespace(github_installation_id=9001),
    )
    limiter = MagicMock(return_value="ok")
    pipeline = MagicMock(
        side_effect=AssertionError("analysis pipeline should not run"),
    )

    monkeypatch.setattr(
        jobs,
        "create_engine_from_settings",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        jobs,
        "create_session_factory",
        lambda _engine: lambda: nullcontext(session),
    )
    monkeypatch.setattr(
        jobs,
        "installation_repository_may_enqueue_jobs",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        jobs,
        "worker_job_lock_hold",
        lambda *_args, **_kwargs: nullcontext(True),
    )
    monkeypatch.setattr(jobs, "get_cached_final_report", cache)
    monkeypatch.setattr(jobs, "begin_analysis_for_job_start", begin)
    monkeypatch.setattr(jobs, "resolve_host_repo_context", resolve)
    monkeypatch.setattr(jobs, "check_analysis_rate_limits", limiter)
    monkeypatch.setattr(jobs, "run_pr_analysis_for_natural_key", pipeline)

    return SimpleNamespace(
        jobs=jobs,
        payload=payload,
        session=session,
        cache=cache,
        begin=begin,
        resolve=resolve,
        limiter=limiter,
        pipeline=pipeline,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize(
    "gate",
    ["lock_lost", "cache_hit", "already_completed", "already_running"],
)
def test_short_circuits_never_charge_quota(
    worker: SimpleNamespace,
    gate: str,
) -> None:
    if gate == "lock_lost":
        worker.monkeypatch.setattr(
            worker.jobs,
            "worker_job_lock_hold",
            lambda *_args, **_kwargs: nullcontext(False),
        )
    elif gate == "cache_hit":
        worker.cache.return_value = {"reviewability": "PASS"}
    else:
        worker.begin.return_value = (uuid.uuid4(), gate)

    worker.jobs.run_pr_analysis_stub.fn(worker.payload)

    worker.limiter.assert_not_called()
    worker.pipeline.assert_not_called()

    if gate == "lock_lost":
        worker.cache.assert_not_called()
        worker.begin.assert_not_called()
    elif gate == "cache_hit":
        worker.begin.assert_not_called()
    else:
        worker.resolve.assert_not_called()


@pytest.mark.parametrize(
    "outcome",
    ["installation_exceeded", "repository_exceeded"],
)
def test_quota_rejection_happens_after_dedupe_and_context_validation(
    worker: SimpleNamespace,
    outcome: str,
) -> None:
    worker.limiter.return_value = outcome

    worker.jobs.run_pr_analysis_stub.fn(worker.payload)

    worker.begin.assert_called_once()
    worker.resolve.assert_called_once()
    worker.limiter.assert_called_once_with(
        worker.limiter.call_args.args[0],
        github_installation_id=9001,
        github_repository_id=4242,
        analysis_key=parse_analysis_job_natural_key(worker.payload),
    )
    worker.pipeline.assert_not_called()
    worker.session.commit.assert_not_called()


def test_retries_pass_same_natural_key_to_idempotent_limiter(
    worker: SimpleNamespace,
) -> None:
    analysis_id = uuid.uuid4()
    worker.begin.side_effect = [
        (analysis_id, "created"),
        (analysis_id, "resumed_from_failed"),
    ]

    charged_keys: set[object] = set()
    new_charges: list[object] = []

    def charge_once(
        _settings: object,
        *,
        github_installation_id: int,
        github_repository_id: int,
        analysis_key: object,
    ) -> str:
        assert github_installation_id == 9001
        assert github_repository_id == 4242
        if analysis_key not in charged_keys:
            charged_keys.add(analysis_key)
            new_charges.append(analysis_key)
        return "ok"

    worker.limiter.side_effect = charge_once
    worker.pipeline.side_effect = GitHubRestError(
        "temporary GitHub failure",
        status_code=503,
        retriable=True,
        request_id="request-123",
    )

    for _ in range(2):
        with pytest.raises(GitHubRestError):
            worker.jobs.run_pr_analysis_stub.fn(worker.payload)

    assert worker.pipeline.call_count == 2
    assert worker.limiter.call_count == 2
    assert new_charges == [
        parse_analysis_job_natural_key(worker.payload),
    ]


def test_envelope_without_analysis_key_is_not_charged(
    worker: SimpleNamespace,
) -> None:
    payload = {
        "github_installation_id": 9001,
        "github_repository_id": 4242,
    }

    worker.jobs.run_pr_analysis_stub.fn(payload)

    worker.limiter.assert_not_called()
    worker.begin.assert_not_called()
    worker.session.commit.assert_called_once()
