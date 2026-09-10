"""reconcile schema applied by hand so a fresh database matches app/models

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-09-10

These objects exist in the live database because they were applied by hand,
but no migration created them, so `alembic upgrade head` on an empty database
produced a schema the code could not run against.

Every statement here is idempotent - IF NOT EXISTS throughout - so this is a
no-op against the live database and does the real work on a fresh one.
Postgres 16 allows ALTER TYPE ... ADD VALUE inside a transaction as long as
the new value is not used in the same transaction, which it is not here.
"""
from alembic import op

revision = "b8c9d0e1f2a3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # users
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS suspended BOOLEAN NOT NULL DEFAULT FALSE")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ")
    # Profile self-service fields, also applied by hand.
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS home_club VARCHAR(150)")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS handicap_index DOUBLE PRECISION")

    # coaches: credential + payout counters
    op.execute("ALTER TABLE coaches ADD COLUMN IF NOT EXISTS credential VARCHAR(20) NOT NULL DEFAULT 'golf_coach'")
    op.execute("ALTER TABLE coaches ADD COLUMN IF NOT EXISTS period_reviews INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE coaches ADD COLUMN IF NOT EXISTS period_approvals INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE coaches ADD COLUMN IF NOT EXISTS lifetime_reviews INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE coaches ADD COLUMN IF NOT EXISTS lifetime_approvals INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE coaches ADD COLUMN IF NOT EXISTS lifetime_paid_cents INTEGER NOT NULL DEFAULT 0")

    # submissions
    op.execute("ALTER TABLE submissions ADD COLUMN IF NOT EXISTS avatar_skin_tone VARCHAR(9)")
    op.execute("ALTER TABLE submissions ADD COLUMN IF NOT EXISTS pga_sendback_reason VARCHAR(1000)")

    # the PGA approval state the two-tier review rule depends on
    op.execute("ALTER TYPE submissionstatus ADD VALUE IF NOT EXISTS 'PGA_APPROVAL'")

    # admin settings (price, free mode, payout rates, instructor picker)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key VARCHAR(100) PRIMARY KEY,
            value VARCHAR(500) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    # Deliberately not reversed. These columns carry live production data
    # (payout totals, suspension state, admin settings) and this migration
    # only ever catches a fresh database up to what production already has.
    # Dropping them on a downgrade would destroy real records.
    pass
