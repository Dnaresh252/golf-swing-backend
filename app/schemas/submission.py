import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SubmissionCreate(BaseModel):
    """Empty body — submission is created from the authenticated user's token."""
    pass


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
