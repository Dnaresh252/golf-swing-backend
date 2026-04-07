import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.submission import Submission
    from app.models.user import User


class DiscountCode(Base):
    __tablename__ = "discount_codes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Format: GOLF2024-XXXXX
    code: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False, index=True
    )
    discount_percent: Mapped[int] = mapped_column(
        Integer, nullable=False, default=15, server_default="15"
    )
    valid_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="discount_codes")
    submission: Mapped["Submission"] = relationship(
        "Submission", back_populates="discount_codes"
    )

    __table_args__ = (
        Index("ix_discount_codes_user_id", "user_id"),
        Index("ix_discount_codes_code_used", "code", "used"),
        Index("ix_discount_codes_valid_until", "valid_until"),
    )

    def __repr__(self) -> str:
        return f"<DiscountCode id={self.id} code={self.code} used={self.used}>"
