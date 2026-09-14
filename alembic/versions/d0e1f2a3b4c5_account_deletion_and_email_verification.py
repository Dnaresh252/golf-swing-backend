"""account deletion grace period and email verification

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-09-14

deletion_requested_at starts the 7-day grace period; deleted_at marks the
purge, after which the users row is an anonymous tombstone kept so payments
and submissions still reconcile.

Email verification becomes enforced for submitting and paying. It was never
implemented before, so every existing account is unverified through no fault
of its own. They are marked verified here so nobody who can use the product
today is locked out; only accounts created from now on have to verify.
"""
import sqlalchemy as sa
from alembic import op

revision = "d0e1f2a3b4c5"
down_revision = "c9d0e1f2a3b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("deletion_requested_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_users_deletion_requested_at", "users", ["deletion_requested_at"])
    op.execute("UPDATE users SET is_verified = true WHERE is_verified = false")


def downgrade() -> None:
    # The verified flags are left as they are: there is no record of which
    # accounts were grandfathered, and un-verifying real customers would lock
    # them out of paying.
    op.drop_index("ix_users_deletion_requested_at", table_name="users")
    op.drop_column("users", "deleted_at")
    op.drop_column("users", "deletion_requested_at")
