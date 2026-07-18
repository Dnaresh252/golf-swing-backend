import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SubmissionCreate(BaseModel):
    club_type: Optional[str] = None
    avatar_choice: Optional[str] = None


class SubmissionFileResponse(BaseModel):
    id: uuid.UUID
    file_type: str
    file_url: str
    file_size_bytes: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SubmissionResponse(BaseModel):
    id: uuid.UUID
    status: str
    club_type: Optional[str] = None
    avatar_choice: Optional[str] = None
    coach_id: Optional[uuid.UUID] = None
    files: List[SubmissionFileResponse] = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SubmissionListResponse(BaseModel):
    items: List[SubmissionResponse]
    total: int
    page: int
    pages: int
    limit: int


class SubmissionStatusResponse(BaseModel):
    submission_id: uuid.UUID
    status: str
    message: str
    updated_at: datetime


class SubmissionStatusDetailResponse(BaseModel):
    """Detailed status response consumed by the Unity app and frontend polling."""
    submission_id: uuid.UUID
    status: str          # pending | uploading | queued | processing | ready | failed
    avatar_status: str   # none | generating | ready | failed
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class AvatarChoiceRequest(BaseModel):
    avatar_choice: str


class AvatarChoiceResponse(BaseModel):
    submission_id: uuid.UUID
    avatar_choice: str
    updated_at: datetime
