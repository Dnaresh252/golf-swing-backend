import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.avatar import Avatar
    from app.models.coach import Coach
    from app.models.coach_notes import CoachNotes
    from app.models.correction import CorrectedVideo
    from app.models.discount import DiscountCode
    from app.models.payment import Payment
    from app.models.results import ResultsVideo
    from app.models.social import SocialSharing
    from app.models.submission_file import SubmissionFile
    from app.models.user import User


class SubmissionStatus(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    ANALYZING = "ANALYZING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    IN_REVIEW = "IN_REVIEW"
    CORRECTIONS_MADE = "CORRECTIONS_MADE"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    coach_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coaches.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(SubmissionStatus, name="submissionstatus"),
        nullable=False,
        default=SubmissionStatus.PENDING,
        server_default=SubmissionStatus.PENDING.value,
        index=True,
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
    user: Mapped["User"] = relationship("User", back_populates="submissions")
    coach: Mapped[Optional["Coach"]] = relationship(
        "Coach", back_populates="submissions"
    )
    files: Mapped[List["SubmissionFile"]] = relationship(
        "SubmissionFile", back_populates="submission", cascade="all, delete-orphan"
    )
    avatar: Mapped[Optional["Avatar"]] = relationship(
        "Avatar",
        back_populates="submission",
        uselist=False,
        cascade="all, delete-orphan",
    )
    corrected_videos: Mapped[List["CorrectedVideo"]] = relationship(
        "CorrectedVideo", back_populates="submission", cascade="all, delete-orphan"
    )
    coach_notes: Mapped[Optional["CoachNotes"]] = relationship(
        "CoachNotes",
        back_populates="submission",
        uselist=False,
        cascade="all, delete-orphan",
    )
    results_videos: Mapped[List["ResultsVideo"]] = relationship(
        "ResultsVideo", back_populates="submission", cascade="all, delete-orphan"
    )
    social_sharing: Mapped[Optional["SocialSharing"]] = relationship(
        "SocialSharing",
        back_populates="submission",
        uselist=False,
        cascade="all, delete-orphan",
    )
    discount_codes: Mapped[List["DiscountCode"]] = relationship(
        "DiscountCode", back_populates="submission", cascade="all, delete-orphan"
    )
    payment: Mapped[Optional["Payment"]] = relationship(
        "Payment",
        back_populates="submission",
        uselist=False,
    )

    __table_args__ = (
        Index("ix_submissions_user_status", "user_id", "status"),
        Index("ix_submissions_coach_status", "coach_id", "status"),
        Index("ix_submissions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Submission id={self.id} status={self.status}>"
