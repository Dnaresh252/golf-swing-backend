import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.integrations.backblaze import b2_service
from app.models.avatar import Avatar, AvatarStatus
from app.models.coach_notes import CoachNotes
from app.models.results import FacePrivacy, ResultsVideo
from app.models.submission import Submission
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def _get_owned_submission(
    db: AsyncSession, submission_id: uuid.UUID, user_id: uuid.UUID
) -> Submission:
    result = await db.execute(
        select(Submission).where(Submission.id == submission_id)
    )
    sub = result.scalar_one_or_none()
    if sub is None or sub.user_id != user_id:
        raise HTTPException(status_code=404, detail="Submission not found.")
    return sub


# ---------------------------------------------------------------------------
# GET /submissions/{id}/practice-mode
# ---------------------------------------------------------------------------

@router.get("/{submission_id}/practice-mode", summary="Get practice mode comparison data")
async def get_practice_mode(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    await _get_owned_submission(db, submission_id, current_user.id)

    # Load coach notes (corrected skeleton)
    notes_result = await db.execute(
        select(CoachNotes).where(CoachNotes.submission_id == submission_id)
    )
    notes = notes_result.scalar_one_or_none()

    if notes is None or notes.corrected_skeleton_json is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "No corrections available yet.", "request_id": rid},
        )

    # Load avatar (original skeleton)
    avatar_result = await db.execute(
        select(Avatar).where(Avatar.submission_id == submission_id)
    )
    avatar = avatar_result.scalar_one_or_none()
    original_skeleton = avatar.skeleton_json if avatar else None

    corrected = notes.corrected_skeleton_json
    joints = corrected.get("joints", [])

    # Build key frames from corrected skeleton
    # These indices correspond to approximate golf swing phases
    # based on MediaPipe BlazePose joint topology
    key_frames = {
        "address": _extract_key_frame(joints, phase="address"),
        "top":     _extract_key_frame(joints, phase="top"),
        "impact":  _extract_key_frame(joints, phase="impact"),
        "follow_through": _extract_key_frame(joints, phase="follow_through"),
    }

    # Accuracy zones: each joint has an acceptable deviation range (in normalised units)
    accuracy_zones = [
        {
            "joint_id": j.get("id"),
            "joint_name": j.get("name"),
            "acceptable_range": 0.05,  # ±5% in normalised coordinate space
            "target_x": j.get("x"),
            "target_y": j.get("y"),
            "target_z": j.get("z"),
        }
        for j in joints
    ]

    logger.info("Practice mode accessed: %s", submission_id)
    return {
        "status": "success",
        "message": "Practice mode data retrieved.",
        "data": {
            "submission_id": str(submission_id),
            "corrected_skeleton_json": corrected,
            "original_skeleton_json": original_skeleton,
            "key_frames": key_frames,
            "accuracy_zones": accuracy_zones,
        },
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/practice-mode/record
# ---------------------------------------------------------------------------

@router.post(
    "/{submission_id}/practice-mode/record",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a practice session recording",
)
async def record_practice_session(
    submission_id: uuid.UUID,
    request: Request,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    await _get_owned_submission(db, submission_id, current_user.id)

    if file.content_type not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Invalid video format. Allowed: MP4, MOV.", "request_id": rid},
        )

    content = await file.read()
    if len(content) > settings.max_video_size_bytes:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Video exceeds {settings.MAX_VIDEO_SIZE_MB} MB.", "request_id": rid},
        )

    dest = b2_service.build_destination_path(
        str(current_user.id),
        str(submission_id),
        "PRACTICE_RECORDING",
        file.filename or "practice.mp4",
    )
    try:
        upload_result = b2_service.upload_file(content, dest, file.content_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    recording = ResultsVideo(
        submission_id=submission_id,
        video_url=upload_result["file_url"],
        b2_file_id=upload_result["b2_file_id"],
        face_privacy=FacePrivacy.SHOW,
    )
    db.add(recording)
    await db.flush()
    await db.refresh(recording)

    logger.info("Practice recording saved: %s", submission_id)
    return {
        "status": "success",
        "message": "Practice recording saved.",
        "data": {
            "id": str(recording.id),
            "submission_id": str(submission_id),
            "video_url": recording.video_url,
            "created_at": recording.created_at.isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# GET /submissions/{id}/practice-sessions
# ---------------------------------------------------------------------------

@router.get("/{submission_id}/practice-sessions", summary="List all practice recordings")
async def list_practice_sessions(
    submission_id: uuid.UUID,
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    import math
    from sqlalchemy import func

    await _get_owned_submission(db, submission_id, current_user.id)

    page = max(1, page)
    limit = max(1, min(limit, 100))
    offset = (page - 1) * limit

    count_result = await db.execute(
        select(func.count(ResultsVideo.id)).where(
            ResultsVideo.submission_id == submission_id
        )
    )
    total: int = count_result.scalar_one()

    result = await db.execute(
        select(ResultsVideo)
        .where(ResultsVideo.submission_id == submission_id)
        .order_by(ResultsVideo.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    recordings = result.scalars().all()

    items = [
        {
            "id": str(r.id),
            "video_url": r.video_url,
            "face_privacy": r.face_privacy.value,
            "created_at": r.created_at.isoformat(),
        }
        for r in recordings
    ]

    logger.info("Practice sessions retrieved: %s", submission_id)
    return {
        "status": "success",
        "message": "Practice sessions retrieved.",
        "data": {
            "items": items,
            "total": total,
            "page": page,
            "pages": math.ceil(total / limit) if total else 0,
            "limit": limit,
        },
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_key_frame(joints: list, phase: str) -> dict:
    """
    Extract a simplified pose snapshot for a named swing phase.
    In a real implementation the phase detector would identify the
    frame index from the video; here we return the full joint set
    with a phase label so the frontend can build the comparison view.
    """
    # Key joint subsets per phase (by landmark name)
    _PHASE_JOINTS = {
        "address":        ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"],
        "top":            ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "impact":         ["left_wrist", "right_wrist", "left_hip", "right_hip", "nose"],
        "follow_through": ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "nose"],
    }
    include = set(_PHASE_JOINTS.get(phase, []))
    filtered = [j for j in joints if j.get("name") in include] if include else joints
    return {"phase": phase, "joints": filtered}
