"""add avatar_choice to submissions

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-05
"""

from alembic import op

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'submissions' AND column_name = 'avatar_choice'
            ) THEN
                ALTER TABLE submissions ADD COLUMN avatar_choice VARCHAR(100);
            END IF;
        END$$;
    """)


def downgrade():
    op.execute("ALTER TABLE submissions DROP COLUMN IF EXISTS avatar_choice;")
