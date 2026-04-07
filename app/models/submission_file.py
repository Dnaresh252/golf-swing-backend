import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.submission import Submission


class FileType(str, enum.Enum):
    FRONT_IMAGE = "FRONT_IMAGE"
    LEFT_IMAGE = "LEFT_IMAGE"
    RIGHT_IMAGE = "RIGHT_IMAGE"
    BACK_IMAGE = "BACK_IMAGE"
    SWING_VIDEO = "SWING_VIDEO"


class SubmissionFile(Base):
    __tablename__ = "submission_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_type: Mapped[FileType] = mapped_column(
        Enum(FileType, name="filetype"), nullable=False
    )
    file_url: Mapped[str] = mapped_column(Text, nullable=False)
    b2_file_id: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    submission: Mapped["Submission"] = relationship(
        "Submission", back_populates="files"
    )

    __table_args__ = (
        Index("ix_submission_files_submission_id", "submission_id"),
        Index("ix_submission_files_submission_type", "submission_id", "file_type"),
    )

    def __repr__(self) -> str:
        return f"<SubmissionFile id={self.id} file_type={self.file_type}>"
