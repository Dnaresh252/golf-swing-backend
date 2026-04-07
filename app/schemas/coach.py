import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class CoachQueueItem(BaseModel):
    id: uuid.UUID
    user_name: str
    status: str
    created_at: datetime
    files_count: int

    model_config = {"from_attributes": True}


class CoachQueueResponse(BaseModel):
    items: List[CoachQueueItem]
    total: int
    page: int
    pages: int
    limit: int


class CoachNotesUpdate(BaseModel):
    notes_text: Optional[str] = None
    recommendations: Optional[str] = None
    corrected_skeleton_json: Optional[Dict[str, Any]] = None

    @field_validator("notes_text", "recommendations", mode="before")
    @classmethod
    def max_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 2000:
            raise ValueError("Field must not exceed 2000 characters.")
        return v

    @field_validator("corrected_skeleton_json", mode="before")
    @classmethod
    def validate_skeleton(cls, v: Optional[Dict]) -> Optional[Dict]:
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("corrected_skeleton_json must be an object.")
            if "joints" not in v:
                raise ValueError(
                    "corrected_skeleton_json must contain a 'joints' array."
                )
        return v


class CoachNotesResponse(BaseModel):
    id: uuid.UUID
    submission_id: uuid.UUID
    notes_text: Optional[str] = None
    recommendations: Optional[str] = None
    corrected_skeleton_json: Optional[Dict[str, Any]] = None
    auto_saved_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class CoachActionRequest(BaseModel):
    reason: Optional[str] = None


class CorrectedVideoUpload(BaseModel):
    angle: str

    @field_validator("angle", mode="before")
    @classmethod
    def validate_angle(cls, v: str) -> str:
        valid = {"top", "front", "left", "right", "back"}
        if v.lower().strip() not in valid:
            raise ValueError(
                f"Invalid angle '{v}'. Must be one of: top, front, left, right, back."
            )
        return v.lower().strip()
