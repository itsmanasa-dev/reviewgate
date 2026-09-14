"""GitHub webhook delivery dedupe using ``webhook_deliveries`` (``docs/DESIGN.md`` §13.3, §16.1)."""

from __future__ import annotations

from typing import Literal

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

from reviewgate.app.settings import AppSettings
from reviewgate.app.storage.db import create_engine_from_settings, create_session_factory
from reviewgate.app.storage.models import WebhookDelivery

ClaimResult = Literal["claimed", "duplicate", "database_unavailable"]


def claim_github_webhook_delivery(
    settings: AppSettings,
    *,
    delivery_id: str,
    event_name: str,
) -> ClaimResult:
    """Atomically claim a delivery id using PostgreSQL upsert with RETURNING.

    Uses ``INSERT ... ON CONFLICT (github_delivery_id) DO UPDATE ... WHERE processed IS false RETURNING id``
    so that:
    1. New deliveries are inserted with ``processed=False`` and claimed.
    2. Existing unprocessed deliveries (from a prior failed attempt) are atomically updated and claimed.
    3. Existing processed deliveries fail the ``WHERE`` clause, returning no row, which indicates a duplicate.

    Args:
        settings: Application settings (``REVIEWGATE_DATABASE_URL``).
        delivery_id: ``X-GitHub-Delivery`` header value.
        event_name: ``X-GitHub-Event`` header value.

    Returns:
        ``claimed`` when a new or unprocessed row was atomically claimed,
        ``duplicate`` when the delivery was already marked processed, or
        ``database_unavailable`` when Postgres is unreachable so the HTTP layer
        can surface a retryable **503**.

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

    session_factory = create_session_factory(engine)
    with session_factory() as session:
        insert_stmt = pg_insert(WebhookDelivery).values(
            github_delivery_id=delivery_id,
            event_name=event_name,
            processed=False,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["github_delivery_id"],
            set_={"event_name": insert_stmt.excluded.event_name},
            where=(WebhookDelivery.processed.is_(False)),
        ).returning(WebhookDelivery.id)
        try:
            inserted_id = session.execute(upsert_stmt).scalar_one_or_none()
            session.commit()
        except OperationalError:
            session.rollback()
            return "database_unavailable"
        return "claimed" if inserted_id is not None else "duplicate"


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
