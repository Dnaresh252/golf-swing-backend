import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.submission import Submission
    from app.models.user import User


class SocialSharing(Base):
    __tablename__ = "social_sharings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    platforms: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    opt_in: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    youtube_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tiktok_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    posted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    submission: Mapped["Submission"] = relationship(
        "Submission", back_populates="social_sharing"
    )
    user: Mapped["User"] = relationship("User", back_populates="social_sharings")

    __table_args__ = (
        Index("ix_social_sharings_submission_id", "submission_id"),
        Index("ix_social_sharings_user_id", "user_id"),
        Index("ix_social_sharings_opt_in", "opt_in"),
    )

    def __repr__(self) -> str:
        return f"<SocialSharing id={self.id} user_id={self.user_id}>"
