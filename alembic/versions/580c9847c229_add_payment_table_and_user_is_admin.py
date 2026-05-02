"""add payment table and user is_admin

Revision ID: 580c9847c229
Revises: 57026aafba41
Create Date: 2026-05-02 06:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "580c9847c229"
down_revision: Union[str, None] = "57026aafba41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. paymentstatus enum (idempotent) ──────────────────────────────────
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'paymentstatus') THEN
                CREATE TYPE paymentstatus AS ENUM (
                    'PENDING', 'COMPLETED', 'FAILED', 'REFUNDED'
                );
            END IF;
        END$$;
        """
    )

    # ── 2. Add is_admin to users (idempotent) ───────────────────────────────
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'is_admin'
            ) THEN
                ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT false;
            END IF;
        END$$;
        """
    )

    # ── 3. Create payments table (idempotent) ───────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS payments (
            id              UUID        PRIMARY KEY NOT NULL,
            user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            submission_id   UUID        REFERENCES submissions(id) ON DELETE SET NULL,
            stripe_payment_intent_id TEXT,
            stripe_customer_id       TEXT,
            amount_cents    INTEGER     NOT NULL,
            currency        TEXT        NOT NULL DEFAULT 'usd',
            status          paymentstatus NOT NULL DEFAULT 'PENDING',
            metadata        JSON,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )

    # ── 4. Indexes (idempotent) ─────────────────────────────────────────────
    op.execute("CREATE INDEX IF NOT EXISTS ix_payments_user_id ON payments (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_payments_submission_id ON payments (submission_id)")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_payments_stripe_payment_intent_id "
        "ON payments (stripe_payment_intent_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_payments_stripe_customer_id ON payments (stripe_customer_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_payments_status ON payments (status)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_payments_user_status ON payments (user_id, status)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_payments_created_at ON payments (created_at)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_payments_created_at")
    op.execute("DROP INDEX IF EXISTS ix_payments_user_status")
    op.execute("DROP INDEX IF EXISTS ix_payments_status")
    op.execute("DROP INDEX IF EXISTS ix_payments_stripe_customer_id")
    op.execute("DROP INDEX IF EXISTS ix_payments_stripe_payment_intent_id")
    op.execute("DROP INDEX IF EXISTS ix_payments_submission_id")
    op.execute("DROP INDEX IF EXISTS ix_payments_user_id")
    op.execute("DROP TABLE IF EXISTS payments")
    op.execute("DROP TYPE IF EXISTS paymentstatus")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'is_admin'
            ) THEN
                ALTER TABLE users DROP COLUMN is_admin;
            END IF;
        END$$;
        """
    )
