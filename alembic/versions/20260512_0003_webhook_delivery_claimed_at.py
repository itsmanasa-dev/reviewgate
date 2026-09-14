"""Add ``claimed_at`` to ``webhook_deliveries`` for delivery state lease (issue #154).

Revision ID: 16_1_0003
Revises: 16_1_0002
Create Date: 2026-05-12

"""

from __future__ import annotations

from typing import Final

import sqlalchemy as sa
from alembic import op

from reviewgate.app.storage.models import TABLE_WEBHOOK_DELIVERIES

revision: Final[str] = "16_1_0003"
down_revision: Final[str] = "16_1_0002"
branch_labels: Final[None] = None
depends_on: Final[None] = None


def upgrade() -> None:
    """Add ``claimed_at`` timestamp column to ``webhook_deliveries``."""

    op.add_column(
        TABLE_WEBHOOK_DELIVERIES,
        sa.Column(
            "claimed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Drop ``claimed_at`` column from ``webhook_deliveries``."""

    op.drop_column(TABLE_WEBHOOK_DELIVERIES, "claimed_at")
