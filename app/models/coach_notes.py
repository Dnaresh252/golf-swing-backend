import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.coach import Coach
    from app.models.submission import Submission


class CoachNotes(Base):
    __tablename__ = "coach_notes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    coach_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coaches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    notes_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommendations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    corrected_skeleton_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    auto_saved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
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
    submission: Mapped["Submission"] = relationship(
        "Submission", back_populates="coach_notes"
    )
    coach: Mapped["Coach"] = relationship("Coach", back_populates="coach_notes")

    __table_args__ = (
        Index("ix_coach_notes_submission_id", "submission_id"),
        Index("ix_coach_notes_coach_id", "coach_id"),
    )

    def __repr__(self) -> str:
        return f"<CoachNotes id={self.id} submission_id={self.submission_id}>"
