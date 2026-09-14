"""Unit tests for ``claim_github_webhook_delivery`` and ``mark_github_webhook_delivery_processed`` (issue #154)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from reviewgate.app.settings import AppSettings
from reviewgate.app.webhooks.dedupe import (
    claim_github_webhook_delivery,
    mark_github_webhook_delivery_processed,
)


@pytest.fixture
def app_settings(monkeypatch: pytest.MonkeyPatch) -> AppSettings:
    monkeypatch.setenv(
        "REVIEWGATE_DATABASE_URL",
        "postgresql+psycopg://x:y@127.0.0.1:2/db",
    )
    return AppSettings()


def _session_context(session: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__enter__.return_value = session
    cm.__exit__.return_value = None
    sm = MagicMock(return_value=cm)
    return sm


def test_claim_github_webhook_delivery_requires_database_url() -> None:
    settings = AppSettings(database_url=None)
    with pytest.raises(RuntimeError, match="requires REVIEWGATE_DATABASE_URL"):
        claim_github_webhook_delivery(
            settings,
            delivery_id="d1",
            event_name="pull_request",
        )


def test_mark_github_webhook_delivery_processed_requires_database_url() -> None:
    settings = AppSettings(database_url=None)
    with pytest.raises(RuntimeError, match="requires REVIEWGATE_DATABASE_URL"):
        mark_github_webhook_delivery_processed(
            settings,
            delivery_id="d1",
        )


def test_claim_github_webhook_delivery_new_row_claimed(
    app_settings: AppSettings,
) -> None:
    """When delivery is new, upsert inserts and returns 'claimed'."""
    fake_engine = object()
    session = MagicMock()
    first_result = MagicMock()
    first_result.scalar_one_or_none.return_value = uuid.uuid4()
    session.execute.return_value = first_result
    sm = _session_context(session)

    with patch(
        "reviewgate.app.webhooks.dedupe.create_engine_from_settings",
        return_value=fake_engine,
    ):
        with patch(
            "reviewgate.app.webhooks.dedupe.create_session_factory",
            return_value=sm,
        ):
            result = claim_github_webhook_delivery(
                app_settings,
                delivery_id="deliv-new-1",
                event_name="pull_request",
            )

    assert result == "claimed"
    session.execute.assert_called_once()
    session.commit.assert_called_once()


def test_claim_github_webhook_delivery_unprocessed_existing_row_claimed(
    app_settings: AppSettings,
) -> None:
    """Issue #154: When delivery exists with processed=False, atomic DO UPDATE succeeds and returns 'claimed'."""
    fake_engine = object()
    session = MagicMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = uuid.uuid4()
    session.execute.return_value = exec_result
    sm = _session_context(session)

    with patch(
        "reviewgate.app.webhooks.dedupe.create_engine_from_settings",
        return_value=fake_engine,
    ):
        with patch(
            "reviewgate.app.webhooks.dedupe.create_session_factory",
            return_value=sm,
        ):
            result = claim_github_webhook_delivery(
                app_settings,
                delivery_id="deliv-retry-1",
                event_name="pull_request",
            )

    assert result == "claimed"
    session.execute.assert_called_once()
    session.commit.assert_called_once()


def test_claim_github_webhook_delivery_processed_existing_row_duplicate(
    app_settings: AppSettings,
) -> None:
    """When delivery exists with processed=True, DO UPDATE WHERE fails, returning no row and yielding 'duplicate'."""
    fake_engine = object()
    session = MagicMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = None
    session.execute.return_value = exec_result
    sm = _session_context(session)

    with patch(
        "reviewgate.app.webhooks.dedupe.create_engine_from_settings",
        return_value=fake_engine,
    ):
        with patch(
            "reviewgate.app.webhooks.dedupe.create_session_factory",
            return_value=sm,
        ):
            result = claim_github_webhook_delivery(
                app_settings,
                delivery_id="deliv-dup-1",
                event_name="pull_request",
            )

    assert result == "duplicate"
    session.execute.assert_called_once()
    session.commit.assert_called_once()


def test_claim_github_webhook_delivery_database_error_returns_unavailable(
    app_settings: AppSettings,
) -> None:
    """OperationalError during claim rolls back and returns 'database_unavailable'."""
    fake_engine = object()
    session = MagicMock()
    session.execute.side_effect = OperationalError("conn failed", {}, Exception())
    sm = _session_context(session)

    with patch(
        "reviewgate.app.webhooks.dedupe.create_engine_from_settings",
        return_value=fake_engine,
    ):
        with patch(
            "reviewgate.app.webhooks.dedupe.create_session_factory",
            return_value=sm,
        ):
            result = claim_github_webhook_delivery(
                app_settings,
                delivery_id="deliv-err-1",
                event_name="pull_request",
            )

    assert result == "database_unavailable"
    session.rollback.assert_called_once()


def test_mark_github_webhook_delivery_processed_success(
    app_settings: AppSettings,
) -> None:
    """mark_github_webhook_delivery_processed executes update and commits."""
    fake_engine = object()
    session = MagicMock()
    sm = _session_context(session)

    with patch(
        "reviewgate.app.webhooks.dedupe.create_engine_from_settings",
        return_value=fake_engine,
    ):
        with patch(
            "reviewgate.app.webhooks.dedupe.create_session_factory",
            return_value=sm,
        ):
            mark_github_webhook_delivery_processed(
                app_settings,
                delivery_id="deliv-done-1",
            )

    session.execute.assert_called_once()
    session.commit.assert_called_once()


def test_mark_github_webhook_delivery_processed_operational_error_raises(
    app_settings: AppSettings,
) -> None:
    """OperationalError rolls back and re-raises."""
    fake_engine = object()
    session = MagicMock()
    session.execute.side_effect = OperationalError("conn failed", {}, Exception())
    sm = _session_context(session)

    with patch(
        "reviewgate.app.webhooks.dedupe.create_engine_from_settings",
        return_value=fake_engine,
    ):
        with patch(
            "reviewgate.app.webhooks.dedupe.create_session_factory",
            return_value=sm,
        ):
            with pytest.raises(OperationalError):
                mark_github_webhook_delivery_processed(
                    app_settings,
                    delivery_id="deliv-err-1",
                )

    session.rollback.assert_called_once()
