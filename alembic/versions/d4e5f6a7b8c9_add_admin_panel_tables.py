"""add admin panel tables (audit_log, announcements, free_codes, user.suspended, user.last_login_at)

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-07-20
"""

from alembic import op

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade():
    # ── users: suspended flag ─────────────────────────────────────────────────
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'suspended'
            ) THEN
                ALTER TABLE users ADD COLUMN suspended BOOLEAN NOT NULL DEFAULT FALSE;
            END IF;
        END$$;
    """)

    # ── users: last_login_at ──────────────────────────────────────────────────
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'last_login_at'
            ) THEN
                ALTER TABLE users ADD COLUMN last_login_at TIMESTAMPTZ;
            END IF;
        END$$;
    """)

    # ── audit_log ─────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            action VARCHAR(200) NOT NULL,
            detail TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_audit_log_timestamp ON audit_log (timestamp DESC);
    """)

    # ── announcements ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS announcements (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            message TEXT NOT NULL DEFAULT '',
            active BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        INSERT INTO announcements (message, active)
        SELECT '', FALSE
        WHERE NOT EXISTS (SELECT 1 FROM announcements);
    """)

    # ── free_codes ────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS free_codes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            code VARCHAR(50) NOT NULL UNIQUE,
            max_uses INTEGER NOT NULL DEFAULT 1,
            uses INTEGER NOT NULL DEFAULT 0,
            expires_at TIMESTAMPTZ,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_free_codes_code ON free_codes (code);
    """)


def downgrade():
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS suspended;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS last_login_at;")
    op.execute("DROP TABLE IF EXISTS audit_log;")
    op.execute("DROP TABLE IF EXISTS announcements;")
    op.execute("DROP TABLE IF EXISTS free_codes;")
