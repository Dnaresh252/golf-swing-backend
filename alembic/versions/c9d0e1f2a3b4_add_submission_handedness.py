"""add golfer handedness to submissions

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-09-14

The web app asks every golfer whether they are right- or left-handed before
they record. The instructor tool needs the answer to put the club in the
correct hand and apply lead/trail corrections to the right limbs. Nullable:
submissions from before this change have no value, and the tool treats null
as right-handed.
"""
import sqlalchemy as sa
from alembic import op

revision = "c9d0e1f2a3b4"
down_revision = "b8c9d0e1f2a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("submissions", sa.Column("handedness", sa.String(5), nullable=True))


def downgrade() -> None:
    op.drop_column("submissions", "handedness")
