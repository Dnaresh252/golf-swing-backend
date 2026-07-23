import logging
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import get_current_user
from app.models.avatar import AvatarStatus
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_file import FileType
from app.models.user import User
from app.schemas.submission import (
    AvatarChoiceRequest,
    AvatarChoiceResponse,
    SubmissionCreate,
    SubmissionFileResponse,
    SubmissionListResponse,
    SubmissionResponse,
    SubmissionStatusDetailResponse,
    SubmissionStatusResponse,
)
from app.services.submission_service import submission_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def _has_payment_or_free_eligibility(
    db: AsyncSession,
    user_id: uuid.UUID,
    submission_id: "uuid.UUID | None",
) -> bool:
    """
    Server-side payment/free-eligibility gate — never trusts the frontend
    called payments/create-intent first. Used at both submissions/create
    and submit-for-analysis so neither can be reached without payment.
    """
    from datetime import timedelta
    from app.models.payment import Payment, PaymentStatus
    from app.services import app_settings
    from app.utils.helpers import get_current_utc as _now

    if await app_settings.get_bool_setting(db, "ALL_SUBMISSIONS_FREE", False):
        return True

    if submission_id is not None:
        linked_row = await db.execute(
            select(Payment).where(
                Payment.submission_id == submission_id,
                Payment.user_id == user_id,
                Payment.status == PaymentStatus.COMPLETED,
            ).limit(1)
        )
        if linked_row.scalar_one_or_none() is not None:
            return True

    # Claim the most recent unlinked completed payment (create-intent runs
    # before submissions/create exists, so the payment has no submission yet)
    recent_cutoff = _now() - timedelta(hours=48)
    unclaimed_row = await db.execute(
        select(Payment).where(
            Payment.user_id == user_id,
            Payment.status == PaymentStatus.COMPLETED,
            Payment.submission_id.is_(None),
            Payment.created_at >= recent_cutoff,
        ).order_by(Payment.created_at.desc()).limit(1)
    )
    unclaimed = unclaimed_row.scalar_one_or_none()
    if unclaimed is not None:
        if submission_id is not None:
            unclaimed.submission_id = submission_id
            await db.flush()
        return True

    # No first-submission-free rule. Pricing is pay or discount/free code —
    # the only free path besides those is the admin ALL_SUBMISSIONS_FREE
    # toggle handled above. A real coach reviews and is paid for every
    # submission, so an unpaid first swing is a direct cost, not a lost sale.
    return False


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
    body: SubmissionCreate = Body(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    club_type = body.club_type if body else None
    avatar_skin_tone = body.avatar_skin_tone if body else None

    # ── Server-side payment / free-eligibility enforcement (create time) ────
    # The frontend now calls create-intent before submissions/create, but the
    # server never trusts that happened — same gate as submit-for-analysis.
    allowed = await _has_payment_or_free_eligibility(db, current_user.id, submission_id=None)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Payment required before creating a submission.",
        )

    submission = await submission_service.create_submission(
        db, current_user.id, club_type=club_type, avatar_skin_tone=avatar_skin_tone
    )
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
    # ── Server-side payment / free-eligibility enforcement ──────────────
    # Do not trust that the frontend went through the payment page first.
    allowed = await _has_payment_or_free_eligibility(db, current_user.id, submission_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Payment required before this swing can be analyzed.",
        )

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

_SUBMISSION_STATUS_MAP = {
    SubmissionStatus.PENDING:          "pending",
    SubmissionStatus.UPLOADING:        "uploading",
    SubmissionStatus.ANALYZING:        "processing",
    SubmissionStatus.READY_FOR_REVIEW: "queued",
    SubmissionStatus.IN_REVIEW:        "processing",
    SubmissionStatus.CORRECTIONS_MADE: "ready",
    SubmissionStatus.COMPLETED:        "ready",
    SubmissionStatus.REJECTED:         "failed",
}


@router.get(
    "/{submission_id}/status",
    summary="Poll submission + avatar status — use this for frontend and Unity progress tracking",
)
async def get_submission_status(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(Submission)
        .where(Submission.id == submission_id)
        .options(selectinload(Submission.avatar))
    )
    result = await db.execute(stmt)
    submission: Submission = result.scalar_one_or_none()

    if submission is None or submission.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found.")

    mapped_status = _SUBMISSION_STATUS_MAP.get(submission.status, submission.status.value.lower())

    avatar = submission.avatar
    if avatar is None or avatar.status == AvatarStatus.PENDING:
        avatar_status = "none"
    elif avatar.status == AvatarStatus.PROCESSING:
        avatar_status = "generating"
    elif avatar.status == AvatarStatus.COMPLETED:
        avatar_status = "ready"
    else:
        avatar_status = "failed"

    error_message = (avatar.error_message if avatar and avatar.status == AvatarStatus.FAILED else None)

    return {
        "status": "success",
        "message": submission_service.get_status_message(submission.status),
        "data": SubmissionStatusDetailResponse(
            submission_id=submission.id,
            status=mapped_status,
            avatar_status=avatar_status,
            created_at=submission.created_at,
            updated_at=submission.updated_at,
            error_message=error_message,
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/select-avatar
# ---------------------------------------------------------------------------

_VALID_AVATAR_CHOICES = {
    "1st", "2nd", "3rd", "4th", "5th",
    "6th", "7th", "8th", "9th", "10th",
}


@router.post(
    "/{submission_id}/select-avatar",
    summary="Save the golfer's avatar selection (1st–10th)",
)
async def select_avatar(
    submission_id: uuid.UUID,
    body: AvatarChoiceRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.avatar_choice not in _VALID_AVATAR_CHOICES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid avatar_choice '{body.avatar_choice}'. Must be one of: {', '.join(sorted(_VALID_AVATAR_CHOICES))}.",
        )

    result = await db.execute(
        select(Submission).where(Submission.id == submission_id)
    )
    submission: Submission = result.scalar_one_or_none()

    if submission is None or submission.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found.")

    submission.avatar_choice = body.avatar_choice
    await db.commit()
    await db.refresh(submission)

    logger.info(
        "Avatar choice '%s' saved for submission: %s by user: %s",
        body.avatar_choice, submission_id, current_user.id,
    )
    return {
        "status": "success",
        "message": f"Avatar '{body.avatar_choice}' selected.",
        "data": AvatarChoiceResponse(
            submission_id=submission.id,
            avatar_choice=submission.avatar_choice,
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
