import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, field_validator


class SocialOptInRequest(BaseModel):
    platforms: List[str]
    face_privacy: str = "show"

    @field_validator("platforms", mode="before")
    @classmethod
    def validate_platforms(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one platform is required.")
        valid = {"youtube", "tiktok"}
        normalised = []
        for p in v:
            p_lower = p.lower().strip()
            if p_lower not in valid:
                raise ValueError(
                    f"Invalid platform '{p}'. Must be 'youtube' or 'tiktok'."
                )
            normalised.append(p_lower)
        return list(set(normalised))  # deduplicate

    @field_validator("face_privacy", mode="before")
    @classmethod
    def validate_face_privacy(cls, v: str) -> str:
        valid = {"show", "blur", "cover"}
        v_lower = v.lower().strip()
        if v_lower not in valid:
            raise ValueError(
                f"Invalid face_privacy '{v}'. Must be 'show', 'blur', or 'cover'."
            )
        return v_lower


class SocialOptInResponse(BaseModel):
    submission_id: uuid.UUID
    platforms: List[str]
    face_privacy: str
    opt_in: bool
    message: str


class SocialPostResponse(BaseModel):
    submission_id: uuid.UUID
    youtube_url: Optional[str] = None
    tiktok_url: Optional[str] = None
    posted_at: Optional[datetime] = None
    status: str
