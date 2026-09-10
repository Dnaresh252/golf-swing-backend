"""tie earned codes to the account and record verified post urls

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-09-10

free_codes.user_id  - null for admin-created public codes, set for codes
                      granted by verify-post so one golfer cannot redeem
                      another golfer's earned code.
social_sharings.post_video_key - the canonical video id of the verified post,
                      so the same post cannot be cashed in twice.
"""
import sqlalchemy as sa
from alembic import op

revision = "a7b8c9d0e1f2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("free_codes", sa.Column("user_id", sa.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_free_codes_user_id", "free_codes", "users",
        ["user_id"], ["id"], ondelete="CASCADE",
    )
    op.create_index("ix_free_codes_user_id", "free_codes", ["user_id"])

    op.add_column(
        "social_sharings", sa.Column("post_video_key", sa.String(255), nullable=True)
    )
    op.create_index(
        "ix_social_sharing_post_video_key", "social_sharings", ["post_video_key"]
    )


def downgrade() -> None:
    op.drop_index("ix_social_sharing_post_video_key", table_name="social_sharings")
    op.drop_column("social_sharings", "post_video_key")
    op.drop_index("ix_free_codes_user_id", table_name="free_codes")
    op.drop_constraint("fk_free_codes_user_id", "free_codes", type_="foreignkey")
    op.drop_column("free_codes", "user_id")
