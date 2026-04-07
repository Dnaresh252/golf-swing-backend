import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.submission_file import FileType
from app.models.user import User
from app.schemas.submission import (
    SubmissionFileResponse,
    SubmissionListResponse,
    SubmissionResponse,
    SubmissionStatusResponse,
)
from app.services.submission_service import submission_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


# ---------------------------------------------------------------------------
# POST /submissions/create
# ---------------------------------------------------------------------------

@router.post(
    "/create",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new empty submission",
)
async def create_submission(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    submission = await submission_service.create_submission(db, current_user.id)
    logger.info("Submission created: %s by user: %s", submission.id, current_user.id)
    return {
        "status": "success",
        "message": "Submission created successfully.",
        "data": SubmissionResponse.model_validate(submission).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/upload-images
# ---------------------------------------------------------------------------

@router.post(
    "/{submission_id}/upload-images",
    summary="Upload all 4 required swing images at once (front, left, right, back)",
)
async def upload_images(
    submission_id: uuid.UUID,
    request: Request,
    front_image: UploadFile,
    left_image: UploadFile,
    right_image: UploadFile,
    back_image: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)

    files = {
        FileType.FRONT_IMAGE: front_image,
        FileType.LEFT_IMAGE:  left_image,
        FileType.RIGHT_IMAGE: right_image,
        FileType.BACK_IMAGE:  back_image,
    }

    submission = await submission_service.upload_images(
        db=db,
        submission_id=submission_id,
        user_id=current_user.id,
        files=files,
    )

    logger.info("Images uploaded for submission: %s", submission_id)
    return {
        "status": "success",
        "message": "All 4 images uploaded successfully.",
        "data": SubmissionResponse.model_validate(submission).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/upload-video
# ---------------------------------------------------------------------------

@router.post(
    "/{submission_id}/upload-video",
    summary="Upload the swing video (max 15 s, MP4/MOV). Upload images first.",
)
async def upload_video(
    submission_id: uuid.UUID,
    request: Request,
    swing_video: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    submission = await submission_service.upload_video(
        db=db,
        submission_id=submission_id,
        user_id=current_user.id,
        file=swing_video,
    )

    logger.info("Video uploaded for submission: %s", submission_id)
    return {
        "status": "success",
        "message": "Video uploaded successfully.",
        "data": SubmissionResponse.model_validate(submission).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/submit-for-analysis
# ---------------------------------------------------------------------------

@router.post(
    "/{submission_id}/submit-for-analysis",
    summary="Finalise and submit for AI analysis (requires all 4 images + video)",
)
async def submit_for_analysis(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    submission = await submission_service.submit_for_analysis(
        db=db,
        submission_id=submission_id,
        user_id=current_user.id,
    )

    logger.info("Analysis triggered for submission: %s", submission_id)
    return {
        "status": "success",
        "message": "Your swing is being analyzed.",
        "data": SubmissionStatusResponse(
            submission_id=submission.id,
            status=submission.status.value,
            message=submission_service.get_status_message(submission.status),
            updated_at=submission.updated_at,
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# GET /submissions
# ---------------------------------------------------------------------------

@router.get(
    "",
    summary="List the current user's submissions (newest first, paginated)",
)
async def list_submissions(
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await submission_service.get_user_submissions(
        db=db,
        user_id=current_user.id,
        page=page,
        limit=limit,
    )

    return {
        "status": "success",
        "message": "Submissions retrieved.",
        "data": SubmissionListResponse(
            items=[SubmissionResponse.model_validate(s) for s in result["items"]],
            total=result["total"],
            page=result["page"],
            pages=result["pages"],
            limit=result["limit"],
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# GET /submissions/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}",
    summary="Get a single submission with all files",
)
async def get_submission(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    submission = await submission_service.get_submission(
        db=db,
        submission_id=submission_id,
        user_id=current_user.id,
    )

    return {
        "status": "success",
        "message": "Submission retrieved.",
        "data": SubmissionResponse.model_validate(submission).model_dump(),
    }


# ---------------------------------------------------------------------------
# GET /submissions/{id}/status
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}/status",
    summary="Poll submission status — use this for frontend progress tracking",
)
async def get_submission_status(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    submission = await submission_service.get_submission(
        db=db,
        submission_id=submission_id,
        user_id=current_user.id,
    )
    message = submission_service.get_status_message(submission.status)

    return {
        "status": "success",
        "message": message,
        "data": SubmissionStatusResponse(
            submission_id=submission.id,
            status=submission.status.value,
            message=message,
            updated_at=submission.updated_at,
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# DELETE /submissions/{id}
# ---------------------------------------------------------------------------

@router.delete(
    "/{submission_id}",
    summary="Delete a submission (only allowed in PENDING or REJECTED status)",
)
async def delete_submission(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await submission_service.delete_submission(
        db=db,
        submission_id=submission_id,
        user_id=current_user.id,
    )
    logger.info("Submission deleted: %s by user: %s", submission_id, current_user.id)
    return {
        "status": "success",
        "message": "Submission deleted successfully.",
        "data": None,
    }
