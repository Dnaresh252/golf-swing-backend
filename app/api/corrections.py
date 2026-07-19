import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import get_current_user
from app.models.coach_notes import CoachNotes
from app.models.correction import CorrectedVideo, CorrectionAngle
from app.models.submission import Submission
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

_VALID_ANGLES = {"top", "front", "left", "right", "back"}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def _get_owned_submission(
    db: AsyncSession,
    submission_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Submission:
    result = await db.execute(
        select(Submission).where(Submission.id == submission_id)
    )
    sub = result.scalar_one_or_none()
    if sub is None or sub.user_id != user_id:
        raise HTTPException(status_code=404, detail="Submission not found.")
    return sub


# ---------------------------------------------------------------------------
# GET /submissions/{id}/corrections
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}/corrections",
    summary="Get all corrected videos and coach notes for a submission",
)
async def get_corrections(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    await _get_owned_submission(db, submission_id, current_user.id)

    # Load corrected videos
    cv_result = await db.execute(
        select(CorrectedVideo)
        .where(CorrectedVideo.submission_id == submission_id)
        .order_by(CorrectedVideo.created_at.asc())
    )
    corrected_videos = cv_result.scalars().all()

    # Load coach notes (contains Unity corrections too)
    notes_result = await db.execute(
        select(CoachNotes).where(CoachNotes.submission_id == submission_id)
    )
    notes = notes_result.scalar_one_or_none()

    # 404 only if neither corrected videos nor Unity corrections exist
    unity_corrections = notes.corrected_skeleton_json if notes else None
    if not corrected_videos and not unity_corrections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": "error",
                "message": "No corrections available yet.",
                "request_id": rid,
            },
        )

    # Build angles dict — key is lowercase angle name
    angles: dict = {}
    for video in corrected_videos:
        angle_key = video.angle.value.lower()
        angles[angle_key] = {
            "video_url": video.video_url,
            "created_at": video.created_at.isoformat(),
        }

    coach_notes_data = None
    if notes:
        coach_notes_data = {
            "notes_text": notes.notes_text,
            "recommendations": notes.recommendations,
        }

    logger.info("Corrections retrieved for: %s", submission_id)
    return {
        "status": "success",
        "message": "Corrections retrieved.",
        "data": {
            "submission_id": str(submission_id),
            "angles": angles,
            "coach_notes": coach_notes_data,
            "total_angles": len(angles),
            "unity_corrections": unity_corrections,
        },
    }


# ---------------------------------------------------------------------------
# GET /submissions/{id}/corrections/{angle}
# ---------------------------------------------------------------------------

@router.get(
    "/{submission_id}/corrections/{angle}",
    summary="Get a specific angle corrected video",
)
async def get_correction_by_angle(
    submission_id: uuid.UUID,
    angle: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    angle_lower = angle.lower().strip()

    if angle_lower not in _VALID_ANGLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": f"Invalid angle '{angle}'. Valid options: top, front, left, right, back.",
                "request_id": rid,
            },
        )

    await _get_owned_submission(db, submission_id, current_user.id)

    angle_enum = CorrectionAngle[angle_lower.upper()]
    cv_result = await db.execute(
        select(CorrectedVideo).where(
            CorrectedVideo.submission_id == submission_id,
            CorrectedVideo.angle == angle_enum,
        )
    )
    video = cv_result.scalar_one_or_none()

    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": "error",
                "message": f"Correction for '{angle_lower}' not ready yet.",
                "request_id": rid,
            },
        )

    logger.info("Correction angle %s retrieved: %s", angle_lower, submission_id)
    return {
        "status": "success",
        "message": f"Correction for angle '{angle_lower}' retrieved.",
        "data": {
            "submission_id": str(submission_id),
            "angle": angle_lower,
            "video_url": video.video_url,
            "b2_file_id": video.b2_file_id,
            "created_at": video.created_at.isoformat(),
        },
    }
