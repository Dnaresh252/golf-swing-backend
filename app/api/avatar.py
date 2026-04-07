import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.avatar import AvatarStatus
from app.models.user import User
from app.schemas.avatar import AvatarAngleResponse, AvatarResponse, SkeletonData
from app.services.avatar_service import avatar_service

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_ANGLES = {"top", "front", "left", "right", "back"}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


# ---------------------------------------------------------------------------
# GET /submissions/{id}/avatar
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}/avatar",
    summary="Get the avatar for a submission",
)
async def get_avatar(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    avatar = await avatar_service.get_avatar(db, submission_id, current_user.id)

    if avatar.status == AvatarStatus.PROCESSING:
        return {
            "status": "success",
            "message": "Avatar is being generated, please check back.",
            "data": avatar_service.build_avatar_response(avatar).model_dump(),
        }, status.HTTP_202_ACCEPTED

    if avatar.status == AvatarStatus.FAILED:
        logger.info("Avatar FAILED retrieved for submission: %s", submission_id)
        return {
            "status": "success",
            "message": "Avatar generation failed.",
            "data": avatar_service.build_avatar_response(avatar).model_dump(),
        }

    logger.info("Avatar retrieved for submission: %s", submission_id)
    return {
        "status": "success",
        "message": "Avatar retrieved.",
        "data": avatar_service.build_avatar_response(avatar).model_dump(),
    }


# ---------------------------------------------------------------------------
# GET /submissions/{id}/avatar/skeleton
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}/avatar/skeleton",
    summary="Get raw skeleton JSON (used by Unity coach tool)",
)
async def get_skeleton(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    skeleton_json = await avatar_service.get_skeleton_data(
        db, submission_id, current_user.id
    )

    # Hydrate into SkeletonData schema — fall back gracefully if format differs
    try:
        skeleton = SkeletonData(
            submission_id=submission_id,
            frames=skeleton_json.get("video_frames", []),
        )
        data = skeleton.model_dump()
    except Exception:
        # Return raw JSON to Unity if our schema can't parse Specialist 3's format
        data = skeleton_json

    logger.info("Skeleton data retrieved for submission: %s", submission_id)
    return {
        "status": "success",
        "message": "Skeleton data retrieved.",
        "data": data,
    }


# ---------------------------------------------------------------------------
# GET /submissions/{id}/avatar/angles/{angle}
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}/avatar/angles/{angle}",
    summary="Get a specific angle view image URL (top/front/left/right/back)",
)
async def get_angle_view(
    submission_id: uuid.UUID,
    angle: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)

    if angle.lower() not in _VALID_ANGLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "Invalid angle. Use: top, front, left, right, back",
                "request_id": rid,
            },
        )

    view = await avatar_service.get_angle_view(
        db, submission_id, current_user.id, angle
    )

    logger.info(
        "Angle view '%s' retrieved for submission: %s", angle, submission_id
    )
    return {
        "status": "success",
        "message": f"Angle view '{angle}' retrieved.",
        "data": AvatarAngleResponse(
            angle=view["angle"],
            image_url=view["image_url"],
        ).model_dump(),
    }
