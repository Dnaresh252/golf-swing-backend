import logging
import uuid
from datetime import timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.integrations.backblaze import b2_service
from app.models.discount import DiscountCode
from app.models.submission import Submission, SubmissionStatus
from app.models.user import User
from app.utils.helpers import get_current_utc
from app.utils.security import hash_password, verify_password

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
    home_club: Optional[str] = None
    handicap_index: Optional[float] = None
    # V6 frontend currently sends "handicap" (string) instead of
    # "handicap_index" — accepted as an alias so existing calls don't break.
    handicap: Optional[str] = None

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

    @field_validator("home_club", mode="before")
    @classmethod
    def validate_home_club(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if len(v) > 150:
                raise ValueError("Home club must not exceed 150 characters.")
        return v

    @field_validator("handicap_index")
    @classmethod
    def validate_handicap(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (-10.0 <= v <= 54.0):
            raise ValueError("Handicap index must be between -10 and 54.")
        return v


class SettingsUpdateRequest(BaseModel):
    public_profile_enabled: bool


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        import re
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("New password must have an uppercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("New password must have a digit.")
        return v


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
            "home_club": current_user.home_club,
            "handicap_index": current_user.handicap_index,
            "handicap": str(current_user.handicap_index) if current_user.handicap_index is not None else "",
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
    if payload.home_club is not None:
        current_user.home_club = payload.home_club
    if payload.handicap_index is not None:
        current_user.handicap_index = payload.handicap_index
    elif payload.handicap is not None and payload.handicap.strip():
        try:
            current_user.handicap_index = float(payload.handicap)
        except ValueError:
            pass

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
            "home_club": current_user.home_club,
            "handicap_index": current_user.handicap_index,
            "handicap": str(current_user.handicap_index) if current_user.handicap_index is not None else "",
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


# ---------------------------------------------------------------------------
# POST /user/profile/photo — profile picture upload
# ---------------------------------------------------------------------------

@router.post("/profile/photo", summary="Upload the current user's profile photo")
async def upload_profile_photo(
    request: Request,
    photo: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)

    if photo.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Unsupported image type. Allowed: {', '.join(settings.ALLOWED_IMAGE_TYPES)}",
                "request_id": rid,
            },
        )

    file_bytes = await photo.read()
    if len(file_bytes) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Image exceeds the {settings.MAX_IMAGE_SIZE_MB}MB limit.",
                "request_id": rid,
            },
        )

    ext = "png" if photo.content_type == "image/png" else "jpg"
    file_name = f"profile_photos/{current_user.id}/{uuid.uuid4()}.{ext}"

    try:
        result = b2_service.upload_file(file_bytes, file_name, photo.content_type)
    except Exception as exc:
        logger.error("Profile photo upload failed for %s: %s", current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "status": "error",
                "message": "Photo upload failed. Please try again.",
                "request_id": rid,
            },
        )

    current_user.profile_picture_url = result["file_url"]
    await db.flush()
    await db.refresh(current_user)

    logger.info("Profile photo updated for user: %s", current_user.id)
    return {
        "status": "success",
        "message": "Profile photo updated.",
        "data": {"profile_picture_url": current_user.profile_picture_url},
    }


# ---------------------------------------------------------------------------
# POST /user/change-password
# ---------------------------------------------------------------------------

@router.post("/change-password", summary="Change the current user's password")
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)

    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Current password is incorrect.",
                "request_id": rid,
            },
        )

    current_user.password_hash = hash_password(payload.new_password)
    await db.flush()

    logger.info("Password changed for user: %s", current_user.id)
    return {"status": "success", "message": "Password changed successfully."}


# ---------------------------------------------------------------------------
# POST /user/deactivate — self-service soft delete
# ---------------------------------------------------------------------------

@router.post("/deactivate", summary="Deactivate the current user's own account")
async def deactivate_account(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Same soft-delete approach as the admin panel fixes (2026-07-23): flip
    # is_active off so the account disappears from lists and can no longer
    # log in, but every linked submission/payment/etc stays intact.
    current_user.is_active = False
    await db.flush()

    logger.info("User %s deactivated their own account", current_user.id)
    return {"status": "success", "message": "Your account has been deactivated."}


@router.delete("/me", summary="Deactivate the current user's own account (alias)")
async def deactivate_account_alias(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # V6.2 frontend calls DELETE /users/me for the same self-deactivate
    # action as POST /users/deactivate above. Kept as two routes rather
    # than changing either side, since both are now live in the wild.
    return await deactivate_account(request, db, current_user)
