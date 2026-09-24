"""Per-installation and per-repository analysis counters (issue #49).

``docs/DESIGN.md`` §22.2 beta defaults: **500** analyses per calendar day (UTC)
per GitHub installation and **100** per GitHub repository. Counters live in Redis
with day-bucket keys; exceeding a cap returns a dedicated outcome so workers can
skip work without touching Postgres (degraded, safe behavior).

When Redis is unavailable or counters cannot be updated, the limiter fails open
with ``ok`` so production is not hard-blocked by transient cache outages.

If the installation counter was incremented and the repository step then fails,
exceeds its cap, or the installation cap is already exceeded for this call, the
installation increment is rolled back so rejected jobs do not permanently
consume installation quota (PR #114 review).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal

import redis.exceptions

from reviewgate.app.redis_client import connect_redis
from reviewgate.app.settings import AppSettings

if TYPE_CHECKING:
    from reviewgate.app.storage.repositories import AnalysisNaturalKey

logger = logging.getLogger(__name__)

_MAX_ANALYSES_PER_INSTALLATION_PER_DAY: Final[int] = 500
_MAX_ANALYSES_PER_REPOSITORY_PER_DAY: Final[int] = 100
#: Keep keys past UTC midnight so ``EXPIRE`` does not race with the next bucket.
_COUNTER_KEY_TTL_SECONDS: Final[int] = 3 * 24 * 60 * 60

# The marker deliberately has no day suffix: retries across UTC midnight
# must not consume another day's quota for the same analysis.
_CHARGE_ONCE_LUA: Final[str] = """
if redis.call("EXISTS", KEYS[3]) == 1 then
    return 0
end

local installation_count = tonumber(redis.call("GET", KEYS[1]) or "0")
local repository_count = tonumber(redis.call("GET", KEYS[2]) or "0")

if installation_count >= tonumber(ARGV[1]) then
    return 1
end
if repository_count >= tonumber(ARGV[2]) then
    return 2
end

redis.call("INCR", KEYS[1])
redis.call("INCR", KEYS[2])
redis.call("EXPIRE", KEYS[1], tonumber(ARGV[3]))
redis.call("EXPIRE", KEYS[2], tonumber(ARGV[3]))
redis.call("SET", KEYS[3], "1", "EX", tonumber(ARGV[3]))
return 0
"""

AnalysisRateLimitOutcome = Literal["ok", "installation_exceeded", "repository_exceeded"]


def _utc_day_bucket() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def _installation_counter_key(github_installation_id: int) -> str:
    return f"reviewgate:rl:v1:installation:{github_installation_id}:{_utc_day_bucket()}"


def _repository_counter_key(github_repository_id: int) -> str:
    return f"reviewgate:rl:v1:repository:{github_repository_id}:{_utc_day_bucket()}"


def _analysis_charge_marker_key(key: AnalysisNaturalKey) -> str:
    """Stable, opaque Redis identity for one five-part analysis natural key."""

    identity = json.dumps(
        [
            str(key.repository_id),
            key.pull_number,
            key.head_sha,
            key.config_hash,
            key.pr_metadata_hash,
        ],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"reviewgate:rl:v1:charged:{digest}"


def _touch_counter_ttl(client: object, key: str) -> None:
    """Best-effort ``EXPIRE``; TTL loss must not fail the rate-limit check."""

    try:
        client.expire(key, _COUNTER_KEY_TTL_SECONDS)
    except redis.exceptions.RedisError as exc:
        logger.info("analysis rate limit: expire skipped for %s (%s)", key, exc)


def _rollback_installation_counter(client: object, inst_key: str) -> None:
    """Undo one installation ``INCR`` when this request must not consume quota.

    Used when the installation cap is exceeded, the repository leg fails or is
    over cap, or Redis errors occur after the installation counter was bumped.
    """

    try:
        client.decr(inst_key)
    except redis.exceptions.RedisError as exc:
        logger.warning(
            "analysis rate limit: installation rollback failed for %s (%s)",
            inst_key,
            exc,
        )


def check_analysis_rate_limits(
    settings: AppSettings,
    *,
    github_installation_id: int,
    github_repository_id: int,
    analysis_key: AnalysisNaturalKey | None = None,
) -> AnalysisRateLimitOutcome:
    """Charge daily quotas, optionally once per analysis natural key.

    The installation counter is incremented first. If the installation cap is
    exceeded, if the repository counter cannot be updated, or if it is over its
    daily cap, the installation increment for this call is rolled back so
    rejected or skipped analyses do not leak installation quota.
    """

    if github_installation_id < 1 or github_repository_id < 1:
        return "ok"

    client = connect_redis(settings)
    if client is None:
        return "ok"

    inst_key = _installation_counter_key(github_installation_id)
    repo_key = _repository_counter_key(github_repository_id)
    if analysis_key is not None:
        # An atomic Lua script handles both counters and the identity marker.
        # A previously charged analysis is allowed through even if another
        # analysis has since filled today's quota.
        try:
            result = int(
                client.eval(
                    _CHARGE_ONCE_LUA,
                    3,
                    inst_key,
                    repo_key,
                    _analysis_charge_marker_key(analysis_key),
                    _MAX_ANALYSES_PER_INSTALLATION_PER_DAY,
                    _MAX_ANALYSES_PER_REPOSITORY_PER_DAY,
                    _COUNTER_KEY_TTL_SECONDS,
                )
            )
            if result == 1:
                return "installation_exceeded"
            if result == 2:
                return "repository_exceeded"
            if result != 0:
                logger.warning("unexpected atomic rate-limit result: %s", result)
            return "ok"
        except (redis.exceptions.RedisError, TypeError, ValueError) as exc:
            logger.info("atomic analysis rate limit skipped (%s)", exc)
            return "ok"
        finally:
            client.close()

    installation_incremented = False

    try:
        inst_count = int(client.incr(inst_key))
        installation_incremented = True
        if inst_count == 1:
            _touch_counter_ttl(client, inst_key)
        if inst_count > _MAX_ANALYSES_PER_INSTALLATION_PER_DAY:
            logger.warning(
                "analysis rate limit: installation %s exceeded daily cap (%s)",
                github_installation_id,
                _MAX_ANALYSES_PER_INSTALLATION_PER_DAY,
            )
            _rollback_installation_counter(client, inst_key)
            return "installation_exceeded"

        repo_raw = client.incr(repo_key)
        repo_count = int(repo_raw)
        if repo_count == 1:
            _touch_counter_ttl(client, repo_key)
        if repo_count > _MAX_ANALYSES_PER_REPOSITORY_PER_DAY:
            logger.warning(
                "analysis rate limit: repository %s exceeded daily cap (%s)",
                github_repository_id,
                _MAX_ANALYSES_PER_REPOSITORY_PER_DAY,
            )
            _rollback_installation_counter(client, inst_key)
            return "repository_exceeded"

        return "ok"
    except (TypeError, ValueError):
        if installation_incremented:
            _rollback_installation_counter(client, inst_key)
        return "ok"
    except redis.exceptions.RedisError as exc:
        if installation_incremented:
            _rollback_installation_counter(client, inst_key)
        logger.info("analysis rate limit counters skipped (Redis error: %s)", exc)
        return "ok"
    finally:
        client.close()
