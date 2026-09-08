"""add instructor picker columns to submissions

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-08

Adds the two columns the instructor picker needs:

  requested_coach_id            - the instructor the user asked for. Kept
                                  separate from coach_id, which stays "the
                                  instructor actually reviewing".
  instructor_request_expires_at - when that request lapses and the
                                  submission returns to the general queue.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "submissions",
        sa.Column("requested_coach_id", sa.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "submissions",
        sa.Column(
            "instructor_request_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_submissions_requested_coach_id",
        "submissions",
        "coaches",
        ["requested_coach_id"],
        ["id"],
        ondelete="SET NULL",
    )
    # The queue query filters on this column on every coach queue load.
    op.create_index(
        "ix_submissions_requested_coach_id",
        "submissions",
        ["requested_coach_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_submissions_requested_coach_id", table_name="submissions")
    op.drop_constraint(
        "fk_submissions_requested_coach_id", "submissions", type_="foreignkey"
    )
    op.drop_column("submissions", "instructor_request_expires_at")
    op.drop_column("submissions", "requested_coach_id")
