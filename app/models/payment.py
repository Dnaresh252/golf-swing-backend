import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.submission import Submission
    from app.models.user import User


class PaymentStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    submission_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    stripe_payment_intent_id: Mapped[Optional[str]] = mapped_column(
        Text, unique=True, nullable=True, index=True
    )
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, index=True
    )
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(
        Text, nullable=False, default="usd", server_default="usd"
    )
    status: Mapped[PaymentStatus] = mapped_column(
        Enum(PaymentStatus, name="paymentstatus"),
        nullable=False,
        default=PaymentStatus.PENDING,
        server_default=PaymentStatus.PENDING.value,
        index=True,
    )
    # Column name "metadata" in DB; Python attr "payment_metadata" avoids
    # collision with SQLAlchemy's reserved Table.metadata attribute.
    payment_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="payments")
    submission: Mapped[Optional["Submission"]] = relationship(
        "Submission", back_populates="payment"
    )

    __table_args__ = (
        Index("ix_payments_user_status", "user_id", "status"),
        Index("ix_payments_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Payment id={self.id} status={self.status} amount={self.amount_cents}>"
