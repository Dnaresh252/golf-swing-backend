"""add instructor certification columns to coaches

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-09-09

Records which body certifies an instructor, their licence number, and the
admin who verified it. Nullable so instructor rows created before this
change keep working; every new row sets all five.
"""
import sqlalchemy as sa
from alembic import op

revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("coaches", sa.Column("certifying_body", sa.String(32), nullable=True))
    op.add_column("coaches", sa.Column("license_number", sa.String(64), nullable=True))
    op.add_column("coaches", sa.Column("credential_verification", sa.String(32), nullable=True))
    op.add_column("coaches", sa.Column("credential_verified_by", sa.UUID(as_uuid=True), nullable=True))
    op.add_column("coaches", sa.Column("credential_verified_at", sa.DateTime(timezone=True), nullable=True))
    op.create_foreign_key(
        "fk_coaches_credential_verified_by", "coaches", "users",
        ["credential_verified_by"], ["id"], ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_coaches_credential_verified_by", "coaches", type_="foreignkey")
    op.drop_column("coaches", "credential_verified_at")
    op.drop_column("coaches", "credential_verified_by")
    op.drop_column("coaches", "credential_verification")
    op.drop_column("coaches", "license_number")
    op.drop_column("coaches", "certifying_body")
