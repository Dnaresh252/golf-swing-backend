import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select as _select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.avatar import AvatarStatus
from app.models.coach import Coach
from app.models.submission import Submission
from app.models.user import User
from app.schemas.avatar import AvatarAngleResponse, AvatarCustomizationRequest, AvatarResponse
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
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "processing",
                "message": "Avatar is being generated, please check back.",
                "data": avatar_service.build_avatar_response(avatar).model_dump(mode="json"),
            },
        )

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
    # 404 immediately if submission doesn't exist or belongs to a different user.
    await avatar_service._get_owned_submission_id(db, submission_id, current_user.id)

    avatar = await avatar_service._get_avatar_for_submission(db, submission_id)

    # No avatar record yet — analysis hasn't started.
    if avatar is None:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "processing",
                "message": "Skeleton data not yet available. Analysis has not started.",
                "data": None,
            },
        )

    # Generation failed — client should resubmit, not retry.
    if avatar.status == AvatarStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Avatar generation failed. Please resubmit for analysis.",
        )

    # Still processing (PENDING or PROCESSING) or complete but skeleton missing.
    if avatar.status != AvatarStatus.COMPLETED or avatar.skeleton_json is None:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "processing",
                "message": "Skeleton data is being generated. Please check back shortly.",
                "data": None,
            },
        )

    logger.info("Skeleton data retrieved for submission: %s", submission_id)
    # Return the raw dict from the DB directly — no Pydantic round-trip needed.
    # Cache-Control: immutable because skeleton data never changes after generation.
    return JSONResponse(
        content={
            "status": "success",
            "message": "Skeleton data retrieved.",
            "data": avatar.skeleton_json,
        },
        headers={"Cache-Control": "max-age=86400, immutable"},
    )


# ---------------------------------------------------------------------------
# GET /submissions/{id}/avatar/skeleton/raw  (Unity direct consumption)
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}/avatar/skeleton/raw",
    summary="Get skeleton JSON without envelope wrapper (Unity direct consumption)",
)
async def get_skeleton_raw(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await avatar_service._get_owned_submission_id(db, submission_id, current_user.id)
    avatar = await avatar_service._get_avatar_for_submission(db, submission_id)

    if avatar is None:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "processing", "message": "Skeleton data not yet available. Analysis has not started."},
        )

    if avatar.status == AvatarStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Avatar generation failed. Please resubmit for analysis.",
        )

    if avatar.status != AvatarStatus.COMPLETED or avatar.skeleton_json is None:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "processing", "message": "Skeleton data is being generated. Please check back shortly."},
        )

    logger.info("Raw skeleton data retrieved for submission: %s", submission_id)
    return JSONResponse(
        content=avatar.skeleton_json,
        headers={"Cache-Control": "max-age=86400, immutable"},
    )


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


# ---------------------------------------------------------------------------
# GET /submissions/{id}/avatar/download/{file_type}
# ---------------------------------------------------------------------------

_CONTENT_TYPES = {
    "glb": "model/gltf-binary",
    "fbx": "model/fbx",
    "obj": "text/plain",
}

_URL_ATTR = {
    "glb": "avatar_glb_url",
    "fbx": "avatar_fbx_url",
    "obj": "avatar_obj_url",
}


@router.get(
    "/{submission_id}/avatar/download/{file_type}",
    summary="Proxy-download GLB/FBX/OBJ from B2 (requires auth)",
)
async def download_avatar_file(
    submission_id: uuid.UUID,
    file_type: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    file_type = file_type.lower()

    if file_type not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "file_type must be glb, fbx, or obj",
                "request_id": rid,
            },
        )

    # Ownership check: submission owner OR any active coach
    sub_result = await db.execute(
        _select(Submission.user_id).where(Submission.id == submission_id)
    )
    sub_row = sub_result.one_or_none()
    if sub_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found.",
        )

    if sub_row.user_id != current_user.id:
        coach_result = await db.execute(
            _select(Coach).where(
                Coach.user_id == current_user.id,
                Coach.is_active == True,
            )
        )
        if coach_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found.",
            )

    # Get avatar record
    avatar = await avatar_service._get_avatar_for_submission(db, submission_id)
    if avatar is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar not ready yet.",
        )

    file_url = getattr(avatar, _URL_ATTR[file_type], None)
    if not file_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {file_type.upper()} file available for this submission.",
        )

    # Fetch from B2 with auth token and proxy back to client
    import asyncio
    from app.integrations.backblaze import b2_service

    try:
        file_bytes = await asyncio.to_thread(b2_service.download_file_bytes, file_url)
    except Exception as exc:
        logger.error("B2 fetch failed for submission %s type=%s: %s", submission_id, file_type, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Storage fetch failed: {exc}",
        )

    logger.info("File download proxied for submission %s type=%s", submission_id, file_type)
    return Response(
        content=file_bytes,
        media_type=_CONTENT_TYPES[file_type],
        headers={
            "Content-Disposition": f'attachment; filename="avatar_{submission_id}.{file_type}"',
        },
    )


# ---------------------------------------------------------------------------
# PATCH /submissions/{id}/avatar/customization
# ---------------------------------------------------------------------------

@router.patch(
    "/{submission_id}/avatar/customization",
    summary="Update avatar customization sliders (skin tone, body, attire, shoe color)",
)
async def update_avatar_customization(
    submission_id: uuid.UUID,
    body: AvatarCustomizationRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)

    update_data = body.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "No customization fields provided.",
                "request_id": rid,
            },
        )

    avatar = await avatar_service.get_avatar(db, submission_id, current_user.id)

    for field, value in update_data.items():
        setattr(avatar, field, value)

    await db.commit()
    await db.refresh(avatar)

    logger.info("Avatar customization updated for submission: %s", submission_id)
    return {
        "status": "success",
        "message": "Avatar customization updated.",
        "data": avatar_service.build_avatar_response(avatar).model_dump(),
    }
