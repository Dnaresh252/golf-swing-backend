import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.submission import Submission


class FacePrivacy(str, enum.Enum):
    SHOW = "SHOW"
    BLUR = "BLUR"
    COVER = "COVER"


class ResultsVideo(Base):
    __tablename__ = "results_videos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    video_url: Mapped[str] = mapped_column(Text, nullable=False)
    b2_file_id: Mapped[str] = mapped_column(Text, nullable=False)
    face_privacy: Mapped[FacePrivacy] = mapped_column(
        Enum(FacePrivacy, name="faceprivacy"),
        nullable=False,
        default=FacePrivacy.SHOW,
        server_default=FacePrivacy.SHOW.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    submission: Mapped["Submission"] = relationship(
        "Submission", back_populates="results_videos"
    )

    __table_args__ = (
        Index("ix_results_videos_submission_id", "submission_id"),
    )

    def __repr__(self) -> str:
        return f"<ResultsVideo id={self.id} submission_id={self.submission_id}>"
