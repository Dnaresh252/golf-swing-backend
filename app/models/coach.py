import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.coach_notes import CoachNotes
    from app.models.submission import Submission
    from app.models.user import User


class Coach(Base):
    __tablename__ = "coaches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    bio: Mapped[Optional[str]] = mapped_column(
        __import__("sqlalchemy").String(500), nullable=True
    )
    rating: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default="0.0"
    )
    submissions_completed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="coach_profile")
    submissions: Mapped[List["Submission"]] = relationship(
        "Submission", back_populates="coach"
    )
    coach_notes: Mapped[List["CoachNotes"]] = relationship(
        "CoachNotes", back_populates="coach", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_coaches_user_id", "user_id"),
        Index("ix_coaches_rating", "rating"),
        Index("ix_coaches_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Coach id={self.id} user_id={self.user_id}>"
