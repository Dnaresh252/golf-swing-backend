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


class CorrectionAngle(str, enum.Enum):
    TOP = "TOP"
    FRONT = "FRONT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BACK = "BACK"


class CorrectedVideo(Base):
    __tablename__ = "corrected_videos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    angle: Mapped[CorrectionAngle] = mapped_column(
        Enum(CorrectionAngle, name="correctionangle"), nullable=False
    )
    video_url: Mapped[str] = mapped_column(Text, nullable=False)
    b2_file_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    submission: Mapped["Submission"] = relationship(
        "Submission", back_populates="corrected_videos"
    )

    __table_args__ = (
        Index("ix_corrected_videos_submission_id", "submission_id"),
        Index("ix_corrected_videos_submission_angle", "submission_id", "angle"),
    )

    def __repr__(self) -> str:
        return f"<CorrectedVideo id={self.id} angle={self.angle}>"
