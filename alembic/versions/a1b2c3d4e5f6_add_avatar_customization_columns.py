"""add avatar customization columns

Revision ID: a1b2c3d4e5f6
Revises: 580c9847c229
Create Date: 2026-05-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "580c9847c229"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='avatars' AND column_name='skin_tone_value') THEN
                ALTER TABLE avatars ADD COLUMN skin_tone_value INTEGER;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='avatars' AND column_name='body_thickness_value') THEN
                ALTER TABLE avatars ADD COLUMN body_thickness_value INTEGER;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='avatars' AND column_name='attire_selection') THEN
                ALTER TABLE avatars ADD COLUMN attire_selection VARCHAR(50);
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='avatars' AND column_name='shoe_color_value') THEN
                ALTER TABLE avatars ADD COLUMN shoe_color_value FLOAT;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='avatars' AND column_name='headgear_enabled') THEN
                ALTER TABLE avatars ADD COLUMN headgear_enabled BOOLEAN NOT NULL DEFAULT true;
            END IF;
        END$$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE avatars
            DROP COLUMN IF EXISTS skin_tone_value,
            DROP COLUMN IF EXISTS body_thickness_value,
            DROP COLUMN IF EXISTS attire_selection,
            DROP COLUMN IF EXISTS shoe_color_value,
            DROP COLUMN IF EXISTS headgear_enabled;
        """
    )
