import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Announcement(Base):
    __tablename__ = "announcements"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<Announcement active={self.active}>"
