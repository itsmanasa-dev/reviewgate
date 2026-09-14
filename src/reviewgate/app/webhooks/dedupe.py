"""GitHub webhook delivery dedupe using ``webhook_deliveries`` (``docs/DESIGN.md`` §13.3, §16.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final, Literal

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

from reviewgate.app.settings import AppSettings
from reviewgate.app.storage.db import create_engine_from_settings, create_session_factory
from reviewgate.app.storage.models import WebhookDelivery

ClaimResult = Literal["claimed", "duplicate", "database_unavailable"]

#: Maximum duration in seconds a delivery claim/lease is held before an in-flight
#: or crashed attempt is considered expired and can be reclaimed by a retry.
_DEFAULT_LEASE_TIMEOUT_SECONDS: Final[int] = 60


def claim_github_webhook_delivery(
    settings: AppSettings,
    *,
    delivery_id: str,
    event_name: str,
    lease_timeout_seconds: int = _DEFAULT_LEASE_TIMEOUT_SECONDS,
) -> ClaimResult:
    """Atomically claim a delivery id using PostgreSQL upsert with lease semantics.

    Uses ``INSERT ... ON CONFLICT (github_delivery_id) DO UPDATE ... WHERE processed IS false AND claimed_at < :cutoff RETURNING id``
    so that:
    1. New deliveries are inserted with ``processed=False`` and claimed.
    2. Concurrent in-progress deliveries (where ``processed=False`` and the lease
       has not expired) fail the update condition and return no row (duplicate).
    3. Previously failed or crashed attempts (where ``processed=False`` and the lease
       expired or was released) are atomically updated with a fresh lease and claimed.
    4. Processed deliveries (``processed=True``) fail the update condition and return no row (duplicate).

    Args:
        settings: Application settings (``REVIEWGATE_DATABASE_URL``).
        delivery_id: ``X-GitHub-Delivery`` header value.
        event_name: ``X-GitHub-Event`` header value.
        lease_timeout_seconds: Lease timeout window in seconds (default 60).

    Returns:
        ``claimed`` when a new or expired/released row was atomically claimed,
        ``duplicate`` when the delivery was already marked processed or is currently
        leased by an in-flight request, or ``database_unavailable`` when Postgres
        is unreachable so the HTTP layer can surface a retryable **503**.

    Raises:
        RuntimeError: If ``settings.database_url`` is unset (callers must gate).
    """

    if settings.database_url is None:
        raise RuntimeError(
            "claim_github_webhook_delivery requires REVIEWGATE_DATABASE_URL",
        )

    engine = create_engine_from_settings(settings)
    if engine is None:
        raise RuntimeError(
            "create_engine_from_settings returned None despite database_url being set",
        )

    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(seconds=lease_timeout_seconds)

    session_factory = create_session_factory(engine)
    with session_factory() as session:
        insert_stmt = pg_insert(WebhookDelivery).values(
            github_delivery_id=delivery_id,
            event_name=event_name,
            processed=False,
            claimed_at=now,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["github_delivery_id"],
            set_={
                "event_name": insert_stmt.excluded.event_name,
                "claimed_at": now,
            },
            where=(
                (WebhookDelivery.processed.is_(False))
                & (WebhookDelivery.claimed_at < stale_cutoff)
            ),
        ).returning(WebhookDelivery.id)
        try:
            inserted_id = session.execute(upsert_stmt).scalar_one_or_none()
            session.commit()
        except OperationalError:
            session.rollback()
            return "database_unavailable"
        return "claimed" if inserted_id is not None else "duplicate"


def release_github_webhook_delivery(
    settings: AppSettings,
    *,
    delivery_id: str,
) -> None:
    """Release an in-flight delivery claim upon failure so it can be retried immediately.

    Sets ``claimed_at`` back to UNIX epoch so any subsequent GitHub retry does not have
    to wait for the lease timeout to elapse.

    Args:
        settings: Application settings (``REVIEWGATE_DATABASE_URL``).
        delivery_id: ``X-GitHub-Delivery`` header value.
    """

    if settings.database_url is None:
        return

    engine = create_engine_from_settings(settings)
    if engine is None:
        return

    session_factory = create_session_factory(engine)
    with session_factory() as session:
        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        stmt = (
            update(WebhookDelivery)
            .where(
                (WebhookDelivery.github_delivery_id == delivery_id)
                & (WebhookDelivery.processed.is_(False))
            )
            .values(claimed_at=epoch)
        )
        try:
            session.execute(stmt)
            session.commit()
        except OperationalError:
            session.rollback()


def mark_github_webhook_delivery_processed(
    settings: AppSettings,
    *,
    delivery_id: str,
) -> None:
    """Mark a delivery as successfully processed in ``webhook_deliveries``.

    Args:
        settings: Application settings (``REVIEWGATE_DATABASE_URL``).
        delivery_id: ``X-GitHub-Delivery`` header value.

    Raises:
        RuntimeError: If ``settings.database_url`` is unset (callers must gate).
        OperationalError: If database connection or commit fails.
    """

    if settings.database_url is None:
        raise RuntimeError(
            "mark_github_webhook_delivery_processed requires REVIEWGATE_DATABASE_URL",
        )

    engine = create_engine_from_settings(settings)
    if engine is None:
        raise RuntimeError(
            "create_engine_from_settings returned None despite database_url being set",
        )

    session_factory = create_session_factory(engine)
    with session_factory() as session:
        stmt = (
            update(WebhookDelivery)
            .where(WebhookDelivery.github_delivery_id == delivery_id)
            .values(processed=True)
        )
        try:
            session.execute(stmt)
            session.commit()
        except OperationalError:
            session.rollback()
            raise
