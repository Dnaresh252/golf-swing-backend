import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.submission import Submission


class AvatarStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Avatar(Base):
    __tablename__ = "avatars"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    skeleton_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    avatar_obj_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    avatar_fbx_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    avatar_glb_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    view_top_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    view_front_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    view_left_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    view_right_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    view_back_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    skin_tone_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    body_thickness_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    attire_selection: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    shoe_color_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    headgear_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    status: Mapped[AvatarStatus] = mapped_column(
        Enum(AvatarStatus, name="avatarstatus"),
        nullable=False,
        default=AvatarStatus.PENDING,
        server_default=AvatarStatus.PENDING.value,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
        "Submission", back_populates="avatar"
    )

    __table_args__ = (
        Index("ix_avatars_submission_id", "submission_id"),
        Index("ix_avatars_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Avatar id={self.id} status={self.status}>"
