"""initial schema

Revision ID: 0001
Revises:
"""
from alembic import op

import app.models  # noqa: F401
from app.db import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Snapshot of the v1 schema. Later schema changes must be explicit op.* migrations,
    # never edits to this file (create_all here is only valid for the first revision).
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
