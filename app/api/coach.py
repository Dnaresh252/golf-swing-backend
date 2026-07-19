import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.dependencies import get_current_user
from app.models.coach import Coach
from app.models.user import User
from app.models.coach_notes import CoachNotes
from app.models.submission import Submission, SubmissionStatus
from app.schemas.coach import (
    CoachActionRequest,
    CoachNotesResponse,
    CoachNotesUpdate,
    CoachQueueItem,
    CoachQueueResponse,
    CorrectedVideoUpload,
    UnityCorrectionsRequest,
)
from app.services.coach_service import coach_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Coach authentication dependency
# ---------------------------------------------------------------------------

async def get_current_coach(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> tuple[User, Coach]:
    """
    Verify the authenticated user has an active coach profile.
    Returns (User, Coach).  Raises 403 otherwise.
    """
    result = await db.execute(
        select(Coach).where(Coach.user_id == current_user.id)
    )
    coach: Optional[Coach] = result.scalar_one_or_none()

    if coach is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Coach account required.",
        )
    if not coach.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Coach account is inactive.",
        )
    return current_user, coach


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


# ---------------------------------------------------------------------------
# GET /coach/queue
# ---------------------------------------------------------------------------

@router.get("/queue", summary="Get paginated review queue")
async def get_queue(
    request: Request,
    status_filter: Optional[str] = Query(None, description="READY_FOR_REVIEW or IN_REVIEW"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    _, coach = auth
    result = await coach_service.get_queue(
        db, coach.id, status_filter, page, limit
    )
    return {
        "status": "success",
        "message": "Queue retrieved.",
        "data": CoachQueueResponse(
            items=[CoachQueueItem(**item) for item in result["items"]],
            total=result["total"],
            page=result["page"],
            pages=result["pages"],
            limit=result["limit"],
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# GET /coach/queue/{submission_id}
# ---------------------------------------------------------------------------

@router.get(
    "/queue/{submission_id}",
    summary="Open a submission for review (acquires lock)",
)
async def get_submission_for_review(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    _, coach = auth
    data = await coach_service.get_submission_for_review(db, submission_id, coach.id)
    sub = data["submission"]

    return {
        "status": "success",
        "message": "Submission locked for review.",
        "data": {
            "id": str(sub.id),
            "status": sub.status.value,
            "user_name": sub.user.name if sub.user else None,
            "created_at": sub.created_at.isoformat(),
            "files": data["files"],
            "avatar": {
                "status": data["avatar_status"],
                "glb_url": data["avatar_glb_url"],
                "angles": data["angle_views"],
                "skeleton_json": data["skeleton_json"],
            },
            "notes": CoachNotesResponse.model_validate(data["notes"]).model_dump()
            if data["notes"]
            else None,
        },
    }


# ---------------------------------------------------------------------------
# POST /coach/queue/{submission_id}/notes
# ---------------------------------------------------------------------------

@router.post(
    "/queue/{submission_id}/notes",
    summary="Save (auto-save) coach notes for a submission",
)
async def save_notes(
    submission_id: uuid.UUID,
    payload: CoachNotesUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    _, coach = auth
    notes = await coach_service.save_notes(
        db,
        submission_id,
        coach.id,
        payload.model_dump(exclude_none=True),
    )
    return {
        "status": "success",
        "message": "Notes saved.",
        "data": CoachNotesResponse.model_validate(notes).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /coach/queue/{submission_id}/upload-corrected-video
# ---------------------------------------------------------------------------

@router.post(
    "/queue/{submission_id}/upload-corrected-video",
    summary="Upload a corrected angle video for a submission",
)
async def upload_corrected_video(
    submission_id: uuid.UUID,
    request: Request,
    file: UploadFile,
    angle: str = Query(..., description="One of: top, front, left, right, back"),
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    rid = _request_id(request)
    _, coach = auth

    # Validate angle via schema
    try:
        CorrectedVideoUpload(angle=angle)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "Invalid angle. Use: top, front, left, right, back",
                "request_id": rid,
            },
        )

    video = await coach_service.upload_corrected_video(
        db, submission_id, coach.id, angle.lower(), file
    )
    logger.info("Corrected video uploaded: %s angle: %s", submission_id, angle)
    return {
        "status": "success",
        "message": f"Corrected video for angle '{angle}' uploaded.",
        "data": {
            "id": str(video.id),
            "submission_id": str(video.submission_id),
            "angle": video.angle.value,
            "video_url": video.video_url,
            "created_at": video.created_at.isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# POST /coach/queue/{submission_id}/approve
# ---------------------------------------------------------------------------

@router.post(
    "/queue/{submission_id}/approve",
    summary="Approve a submission after review",
)
async def approve_submission(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    _, coach = auth
    await coach_service.approve_submission(db, submission_id, coach.id)
    logger.info("Submission approved by coach: %s", coach.id)
    return {
        "status": "success",
        "message": "Submission approved. User has been notified.",
        "data": None,
    }


# ---------------------------------------------------------------------------
# POST /coach/queue/{submission_id}/reject
# ---------------------------------------------------------------------------

@router.post(
    "/queue/{submission_id}/reject",
    summary="Reject a submission with a reason",
)
async def reject_submission(
    submission_id: uuid.UUID,
    payload: CoachActionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    rid = _request_id(request)
    _, coach = auth

    if not payload.reason or not payload.reason.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "Rejection reason required.",
                "request_id": rid,
            },
        )

    await coach_service.reject_submission(
        db, submission_id, coach.id, payload.reason.strip()
    )
    logger.info("Submission rejected by coach: %s", coach.id)
    return {
        "status": "success",
        "message": "Submission rejected. User has been notified.",
        "data": None,
    }


# ---------------------------------------------------------------------------
# POST /coach/queue/{submission_id}/corrections  (Unity coach tool)
# ---------------------------------------------------------------------------

@router.post(
    "/queue/{submission_id}/corrections",
    summary="Receive corrected skeleton frames from the Unity coach tool",
)
async def save_unity_corrections(
    submission_id: uuid.UUID,
    payload: UnityCorrectionsRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    _, coach = auth

    # Verify submission exists
    sub_result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = sub_result.scalar_one_or_none()
    if submission is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found.")

    # Build the corrections blob to store
    corrections_data = {
        "corrected_frames": [f.model_dump() for f in payload.corrected_frames],
        "coach_notes": payload.coach_notes,
        "submitted_at": payload.created_at,
        "frame_count": len(payload.corrected_frames),
    }

    # Upsert into coach_notes (unique per submission_id)
    notes_result = await db.execute(
        select(CoachNotes).where(CoachNotes.submission_id == submission_id)
    )
    notes = notes_result.scalar_one_or_none()

    if notes is None:
        notes = CoachNotes(
            submission_id=submission_id,
            coach_id=coach.id,
            notes_text=payload.coach_notes,
            corrected_skeleton_json=corrections_data,
        )
        db.add(notes)
    else:
        notes.corrected_skeleton_json = corrections_data
        if payload.coach_notes:
            notes.notes_text = payload.coach_notes

    # Advance submission status to CORRECTIONS_MADE if it was IN_REVIEW
    if submission.status in (SubmissionStatus.IN_REVIEW, SubmissionStatus.READY_FOR_REVIEW):
        submission.status = SubmissionStatus.CORRECTIONS_MADE

    await db.commit()

    logger.info(
        "Unity corrections saved for submission %s by coach %s — %d frame(s)",
        submission_id, coach.id, len(payload.corrected_frames),
    )
    return {
        "status": "success",
        "message": "Corrections saved.",
        "data": {
            "submission_id": str(submission_id),
            "frames_saved": len(payload.corrected_frames),
        },
    }


# ---------------------------------------------------------------------------
# GET /coach/results-queue
# ---------------------------------------------------------------------------

@router.get("/results-queue", summary="Get pending social media approval queue")
async def get_results_queue(
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    _, coach = auth
    result = await coach_service.get_results_queue(db, coach.id, page, limit)

    items = []
    for sharing in result["items"]:
        sub = sharing.submission
        items.append({
            "id": str(sharing.id),
            "submission_id": str(sharing.submission_id),
            "user_name": sub.user.name if sub and sub.user else "Unknown",
            "platforms": sharing.platforms,
            "opt_in": sharing.opt_in,
            "created_at": sharing.created_at.isoformat(),
        })

    return {
        "status": "success",
        "message": "Results queue retrieved.",
        "data": {
            "items": items,
            "total": result["total"],
            "page": result["page"],
            "pages": result["pages"],
            "limit": result["limit"],
        },
    }


# ---------------------------------------------------------------------------
# POST /coach/results-queue/{submission_id}/approve-for-posting
# ---------------------------------------------------------------------------

@router.post(
    "/results-queue/{submission_id}/approve-for-posting",
    summary="Approve a results video for social media posting",
)
async def approve_for_posting(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    _, coach = auth
    result = await coach_service.approve_for_posting(db, submission_id, coach.id)
    logger.info("Results approved for posting: %s", submission_id)
    return {
        "status": "success",
        "message": "Video approved for posting. Discount code sent to user.",
        "data": result,
    }


# ---------------------------------------------------------------------------
# POST /coach/results-queue/{submission_id}/reject-posting
# ---------------------------------------------------------------------------

@router.post(
    "/results-queue/{submission_id}/reject-posting",
    summary="Reject a results video from social media posting",
)
async def reject_posting(
    submission_id: uuid.UUID,
    payload: CoachActionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(get_current_coach),
):
    rid = _request_id(request)
    _, coach = auth

    if not payload.reason or not payload.reason.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "Rejection reason required.",
                "request_id": rid,
            },
        )

    await coach_service.reject_posting(
        db, submission_id, coach.id, payload.reason.strip()
    )
    logger.info("Results posting rejected: %s", submission_id)
    return {
        "status": "success",
        "message": "Posting rejected. User has been notified.",
        "data": None,
    }
