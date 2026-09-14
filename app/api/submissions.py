import logging
import re
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
    avatar_choice = body.avatar_choice if body else None

    # Normalised to the canonical "avatar_N" id before storage, and the
    # skin tone checked, so nothing free-text ever reaches the column and
    # the coach tool always receives one predictable form.
    avatar_choice = _normalise_avatar_choice(avatar_choice)
    avatar_skin_tone = _validate_skin_tone(avatar_skin_tone)

    # Handedness is asked, never guessed from the video. Exactly "right" or
    # "left"; absent or null is allowed (the instructor tool then assumes
    # right-handed and logs that it did).
    handedness = body.handedness if body else None
    if handedness is not None and handedness not in ("right", "left"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid handedness. Must be 'right' or 'left'.",
        )

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
        db, current_user.id, club_type=club_type,
        avatar_skin_tone=avatar_skin_tone, avatar_choice=avatar_choice,
        handedness=handedness,
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
    summary="Upload the swing video (max 15 s, MP4/MOV/WebM). Upload images first.",
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

# The lineup is 14 avatars. The canonical stored form is always the id
# "avatar_N"; the app sends ordinal labels ("7th"), older clients and the
# Unity tool may send the id. Both are accepted, one form is stored, so the
# coach tool never has to guess which it is looking at.
AVATAR_COUNT = 14
_VALID_AVATAR_CHOICES = {f"avatar_{n}" for n in range(1, AVATAR_COUNT + 1)}

_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{_ORDINAL_SUFFIX.get(n % 10, 'th')}"


_ORDINAL_TO_ID = {_ordinal(n): f"avatar_{n}" for n in range(1, AVATAR_COUNT + 1)}

_AVATAR_ERROR = (
    "Invalid avatar_choice. Must be avatar_1 ... avatar_14 or 1st ... 14th."
)
_SKIN_TONE_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _normalise_avatar_choice(raw):
    """
    Accepts "avatar_7", "avatar7", "7" and "7th"; returns "avatar_7".
    Raises 400 on anything else. None passes through untouched so the
    field stays optional.
    """
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in _ORDINAL_TO_ID:
        return _ORDINAL_TO_ID[value]
    if value.startswith("avatar"):
        digits = value[len("avatar"):].lstrip("_")
    else:
        digits = value
    if digits.isdigit():
        n = int(digits)
        if 1 <= n <= AVATAR_COUNT:
            return f"avatar_{n}"
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=_AVATAR_ERROR,
    )


def _validate_skin_tone(raw):
    """Hex colour only. The coach tool applies the value directly."""
    if raw is None:
        return None
    value = str(raw).strip()
    if not _SKIN_TONE_RE.match(value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid avatar_skin_tone. Must be a hex colour like #C68863.",
        )
    return value


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
    normalised_choice = _normalise_avatar_choice(body.avatar_choice)

    result = await db.execute(
        select(Submission).where(Submission.id == submission_id)
    )
    submission: Submission = result.scalar_one_or_none()

    if submission is None or submission.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found.")

    submission.avatar_choice = normalised_choice
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

# ---------------------------------------------------------------------------
# POST /submissions/{id}/instructor  (instructor picker)
# ---------------------------------------------------------------------------

from datetime import timedelta as _timedelta  # noqa: E402
from pydantic import BaseModel as _BaseModel  # noqa: E402

from app.models.coach import Coach as _Coach  # noqa: E402
from app.services import app_settings as _app_settings  # noqa: E402
from app.utils.rate_limit import make_rate_limiter  # noqa: E402

# A user picking an instructor is a deliberate, low-frequency action.
# This stops a script hammering the endpoint to enumerate instructor ids.
_instructor_request_limiter = make_rate_limiter(20, 60)

# Review has not started yet in these states. Once an instructor has the
# submission in review, the pick is no longer meaningful.
_PICKABLE_STATUSES = (
    SubmissionStatus.PENDING,
    SubmissionStatus.UPLOADING,
    SubmissionStatus.ANALYZING,
    SubmissionStatus.READY_FOR_REVIEW,
)


class InstructorRequestBody(_BaseModel):
    coach_id: uuid.UUID


@router.post(
    "/{submission_id}/instructor",
    summary="Request a specific instructor for this submission.",
)
async def request_instructor(
    submission_id: uuid.UUID,
    body: InstructorRequestBody,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _instructor_request_limiter(request)

    # 1. Feature flag. Off means this route does not exist.
    enabled = await _app_settings.get_bool_setting(
        db, "INSTRUCTOR_PICKER_ENABLED", False
    )
    if not enabled:
        raise HTTPException(status_code=404, detail="Not found.")

    # 2. Ownership. Same generic 404 as corrections.py so this never
    #    reveals whether someone else's submission id exists.
    result = await db.execute(
        select(Submission).where(Submission.id == submission_id)
    )
    submission = result.scalar_one_or_none()
    if submission is None or submission.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Submission not found.")

    # 3. Review must not have started.
    if submission.status not in _PICKABLE_STATUSES or submission.coach_id is not None:
        raise HTTPException(
            status_code=409,
            detail="This submission is already being reviewed.",
        )

    # 4. Instructor must exist and be active.
    coach_result = await db.execute(
        select(_Coach).where(_Coach.id == body.coach_id, _Coach.is_active.is_(True))
    )
    coach = coach_result.scalar_one_or_none()
    if coach is None:
        raise HTTPException(status_code=400, detail="Instructor not available.")

    # 5. No request already pending on this submission.
    now = get_current_utc()
    has_pending = (
        submission.requested_coach_id is not None
        and submission.instructor_request_expires_at is not None
        and submission.instructor_request_expires_at > now
    )
    if has_pending:
        raise HTTPException(
            status_code=409,
            detail="An instructor has already been requested for this submission.",
        )

    window_hours = await _app_settings.get_int_setting(
        db, "INSTRUCTOR_ACCEPT_WINDOW_HOURS", 48
    )
    submission.requested_coach_id = coach.id
    submission.instructor_request_expires_at = now + _timedelta(hours=window_hours)
    await db.commit()

    logger.info(
        "User %s requested instructor %s for submission %s (window %sh)",
        current_user.id, coach.id, submission.id, window_hours,
    )

    return {
        "status": "success",
        "message": "Instructor requested.",
        "data": {
            "requested_coach_id": str(coach.id),
            "instructor_request_expires_at": (
                submission.instructor_request_expires_at.isoformat()
            ),
            "accept_window_hours": window_hours,
        },
    }
