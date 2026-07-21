import logging
import uuid
from datetime import timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.discount import DiscountCode
from app.models.submission import Submission, SubmissionStatus
from app.models.user import User
from app.utils.helpers import get_current_utc

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request schemas (local — not worth a separate file)
# ---------------------------------------------------------------------------

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    bio: Optional[str] = None
    profile_picture_url: Optional[str] = None
    public_profile_enabled: Optional[bool] = None

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters.")
        if len(v) > 100:
            raise ValueError("Name must not exceed 100 characters.")
        return v

    @field_validator("bio", mode="before")
    @classmethod
    def validate_bio(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 500:
            raise ValueError("Bio must not exceed 500 characters.")
        return v


class SettingsUpdateRequest(BaseModel):
    public_profile_enabled: bool


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def _get_user_stats(db: AsyncSession, user_id: uuid.UUID) -> dict:
    """Return submission counts and active discount count for a user."""
    total_subs = await db.execute(
        select(func.count(Submission.id)).where(Submission.user_id == user_id)
    )
    completed_subs = await db.execute(
        select(func.count(Submission.id)).where(
            Submission.user_id == user_id,
            Submission.status == SubmissionStatus.COMPLETED,
        )
    )
    now = get_current_utc()
    active_discounts = await db.execute(
        select(func.count(DiscountCode.id)).where(
            DiscountCode.user_id == user_id,
            DiscountCode.used == False,  # noqa: E712
            DiscountCode.valid_until > now,
        )
    )
    return {
        "total_submissions": total_subs.scalar_one(),
        "completed_submissions": completed_subs.scalar_one(),
        "active_discount_codes": active_discounts.scalar_one(),
    }


# ---------------------------------------------------------------------------
# GET /user/profile
# ---------------------------------------------------------------------------

@router.get("/profile", summary="Get the current user's full profile with stats")
async def get_my_profile(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stats = await _get_user_stats(db, current_user.id)

    credential = None
    if current_user.is_admin:
        role = "admin"
    else:
        from app.models.coach import Coach
        coach_row = await db.execute(
            select(Coach.credential).where(Coach.user_id == current_user.id)
        )
        credential = coach_row.scalar_one_or_none()
        role = "coach" if credential is not None else "user"

    return {
        "status": "success",
        "message": "Profile retrieved.",
        "role": role,
        "credential": credential,
        "data": {
            "id": str(current_user.id),
            "email": current_user.email,
            "name": current_user.name,
            "role": role,
            "credential": credential,
            "bio": current_user.bio,
            "profile_picture_url": current_user.profile_picture_url,
            "public_profile_enabled": current_user.public_profile_enabled,
            "is_verified": current_user.is_verified,
            "created_at": current_user.created_at.isoformat(),
            "stats": stats,
        },
    }


# ---------------------------------------------------------------------------
# GET /user/profile/{username}  (public — no auth)
# ---------------------------------------------------------------------------

@router.get("/profile/{username}", summary="Get a public user profile by username")
async def get_public_profile(
    username: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User).where(User.name == username)
    )
    user = result.scalar_one_or_none()

    # Treat disabled profile the same as not-found — don't reveal existence
    if user is None or not user.public_profile_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found.",
        )

    completed_count = await db.execute(
        select(func.count(Submission.id)).where(
            Submission.user_id == user.id,
            Submission.status == SubmissionStatus.COMPLETED,
        )
    )

    return {
        "status": "success",
        "message": "Public profile retrieved.",
        "data": {
            "name": user.name,
            "bio": user.bio,
            "profile_picture_url": user.profile_picture_url,
            "completed_submissions": completed_count.scalar_one(),
        },
    }


# ---------------------------------------------------------------------------
# PUT /user/profile
# ---------------------------------------------------------------------------

@router.put("/profile", summary="Update the current user's profile")
async def update_profile(
    payload: ProfileUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.name is not None:
        current_user.name = payload.name
    if payload.bio is not None:
        current_user.bio = payload.bio
    if payload.profile_picture_url is not None:
        current_user.profile_picture_url = payload.profile_picture_url
    if payload.public_profile_enabled is not None:
        current_user.public_profile_enabled = payload.public_profile_enabled

    await db.flush()
    await db.refresh(current_user)

    logger.info("Profile updated for user: %s", current_user.id)
    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "data": {
            "id": str(current_user.id),
            "email": current_user.email,
            "name": current_user.name,
            "bio": current_user.bio,
            "profile_picture_url": current_user.profile_picture_url,
            "public_profile_enabled": current_user.public_profile_enabled,
            "is_verified": current_user.is_verified,
            "created_at": current_user.created_at.isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# PUT /user/settings
# ---------------------------------------------------------------------------

@router.put("/settings", summary="Update user privacy settings")
async def update_settings(
    payload: SettingsUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.public_profile_enabled = payload.public_profile_enabled
    await db.flush()

    logger.info("Settings updated for user: %s", current_user.id)
    return {
        "status": "success",
        "message": "Settings updated successfully.",
        "data": {
            "public_profile_enabled": current_user.public_profile_enabled,
        },
    }
