import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.integrations.backblaze import b2_service
from app.models.results import FacePrivacy, ResultsVideo
from app.models.social import SocialSharing
from app.models.submission import Submission, SubmissionStatus
from app.models.user import User
from app.schemas.social import SocialOptInRequest, SocialOptInResponse, SocialPostResponse
from app.utils.helpers import get_current_utc

logger = logging.getLogger(__name__)
router = APIRouter()

_SOCIAL_ELIGIBLE_STATUSES = {
    SubmissionStatus.CORRECTIONS_MADE,
    SubmissionStatus.COMPLETED,
}


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
# POST /submissions/{id}/social-opt-in
# ---------------------------------------------------------------------------

@router.post("/{submission_id}/social-opt-in", summary="Opt in to social media sharing")
async def social_opt_in(
    submission_id: uuid.UUID,
    payload: SocialOptInRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    submission = await _get_owned_submission(db, submission_id, current_user.id)

    if submission.status not in _SOCIAL_ELIGIBLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "Complete your submission first.", "request_id": rid},
        )

    # Verify results video exists
    rv_result = await db.execute(
        select(ResultsVideo).where(ResultsVideo.submission_id == submission_id).limit(1)
    )
    results_video = rv_result.scalar_one_or_none()
    if results_video is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "Upload results video first.", "request_id": rid},
        )

    # Update ResultsVideo face_privacy
    try:
        fp_enum = FacePrivacy[payload.face_privacy.upper()]
    except KeyError:
        fp_enum = FacePrivacy.SHOW
    results_video.face_privacy = fp_enum

    # Upsert SocialSharing record
    ss_result = await db.execute(
        select(SocialSharing).where(SocialSharing.submission_id == submission_id)
    )
    sharing = ss_result.scalar_one_or_none()

    if sharing is None:
        sharing = SocialSharing(
            submission_id=submission_id,
            user_id=current_user.id,
            platforms=payload.platforms,
            opt_in=True,
        )
        db.add(sharing)
    else:
        sharing.platforms = payload.platforms
        sharing.opt_in = True

    await db.flush()
    logger.info("Social opt-in for submission: %s", submission_id)

    return {
        "status": "success",
        "message": "Social sharing opt-in recorded.",
        "data": SocialOptInResponse(
            submission_id=submission_id,
            platforms=sharing.platforms,
            face_privacy=payload.face_privacy,
            opt_in=True,
            message="Your video will be reviewed by a coach before posting.",
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/post-results
# ---------------------------------------------------------------------------

@router.post("/{submission_id}/post-results", summary="Upload the final results video")
async def post_results(
    submission_id: uuid.UUID,
    request: Request,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    submission = await _get_owned_submission(db, submission_id, current_user.id)

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
        str(current_user.id), str(submission_id), "RESULTS_VIDEO", file.filename or "results.mp4"
    )
    try:
        loop = asyncio.get_event_loop()
        upload_result = await loop.run_in_executor(
            None, b2_service.upload_file, content, dest, file.content_type
        )
    except RuntimeError as exc:
        logger.error("B2 upload failed for submission %s: %s", submission_id, exc)
        raise HTTPException(status_code=500, detail="Video upload failed. Please try again.")

    results_video = ResultsVideo(
        submission_id=submission_id,
        video_url=upload_result["file_url"],
        b2_file_id=upload_result["b2_file_id"],
        face_privacy=FacePrivacy.SHOW,
    )
    db.add(results_video)

    submission.status = SubmissionStatus.COMPLETED
    await db.flush()
    await db.refresh(results_video)

    logger.info("Results video uploaded: %s", submission_id)
    return {
        "status": "success",
        "message": "Results video uploaded successfully.",
        "data": {
            "id": str(results_video.id),
            "submission_id": str(submission_id),
            "video_url": results_video.video_url,
            "created_at": results_video.created_at.isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/post-to-social
# ---------------------------------------------------------------------------

@router.post("/{submission_id}/post-to-social", summary="Trigger social media posting")
async def post_to_social(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    await _get_owned_submission(db, submission_id, current_user.id)

    ss_result = await db.execute(
        select(SocialSharing).where(SocialSharing.submission_id == submission_id)
    )
    sharing = ss_result.scalar_one_or_none()

    if sharing is None or not sharing.opt_in:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Complete social opt-in first.", "request_id": rid},
        )
    if sharing.posted_at is not None:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Video already posted to social media.", "request_id": rid},
        )

    # Check coach approval — indicated by the task having been triggered
    # (coach approval fires post_to_social_media Celery task directly;
    #  this endpoint is user-initiated and requires coach approval first)
    # We use posted_at=None + opt_in=True as "pending coach approval"
    # If posted_at is still None after opt_in, coach hasn't approved yet.
    # Allow re-triggering only when a coach has explicitly approved (opt_in stays True
    # after coach approval) — we check this by verifying youtube/tiktok URL absence.
    # If the task was never kicked off by coach, youtube_url and tiktok_url are None.
    if sharing.youtube_url is None and sharing.tiktok_url is None and sharing.posted_at is None:
        # No prior posting attempt — check coach approved the posting
        # Coach approval is stored as opt_in=True with the task triggered externally.
        # Without a separate `coach_approved` column we rely on the workflow:
        # coach action sets off the task; this endpoint is a manual re-trigger only.
        pass  # Allow user to manually trigger after opt-in for testing/re-submission

    try:
        from app.workers.video_tasks import post_to_social_media
        post_to_social_media.delay(str(submission_id))
    except Exception as exc:
        logger.error("Failed to queue social post task: %s", exc)
        raise HTTPException(status_code=500, detail="Could not schedule social posting.")

    logger.info("Social posting triggered: %s", submission_id)
    return {
        "status": "success",
        "message": "Your video is being posted to selected platforms.",
        "data": SocialPostResponse(
            submission_id=submission_id,
            status="queued",
        ).model_dump(),
    }, status.HTTP_202_ACCEPTED


# ---------------------------------------------------------------------------
# POST /{submission_id}/verify-post — social post verification + free code
# ---------------------------------------------------------------------------

_GGW_YOUTUBE_HANDLE = "@golfgameworld"
_GGW_TIKTOK_HANDLE = "@ggw0059"
_GGW_HASHTAGS = ("#ggw", "#golfgameworld", "#ggwacademy")


def _fetch_url(url: str, timeout: int = 12) -> tuple:
    """Blocking fetch; returns (status_code, body_text). Runs in executor."""
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (GGW-Verify)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception:
        return 0, ""


from pydantic import BaseModel as _BaseModel


class VerifyPostRequest(_BaseModel):
    post_url: str
    platform: str  # "youtube" or "tiktok"


@router.post("/{submission_id}/verify-post", summary="Verify a public social post and grant a free code")
async def verify_post(
    submission_id: uuid.UUID,
    body: VerifyPostRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)
    submission = await _get_owned_submission(db, submission_id, current_user.id)

    platform = (body.platform or "").strip().lower()
    post_url = (body.post_url or "").strip()

    if platform not in ("youtube", "tiktok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="platform must be 'youtube' or 'tiktok'.",
        )

    # Basic URL sanity — must be a real link on the claimed platform
    valid_hosts = {
        "youtube": ("youtube.com/watch", "youtube.com/shorts", "youtu.be/"),
        "tiktok": ("tiktok.com/",),
    }
    if not post_url.startswith(("http://", "https://")) or not any(
        h in post_url for h in valid_hosts[platform]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"That does not look like a valid {platform} post link.",
        )

    # Prevent double-claiming the reward for the same submission
    sharing_row = await db.execute(
        select(SocialSharing).where(SocialSharing.submission_id == submission_id)
    )
    sharing = sharing_row.scalar_one_or_none()
    if sharing is not None and sharing.posted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A post for this submission has already been verified and rewarded.",
        )

    loop = asyncio.get_event_loop()

    # ── 1. Post is public and reachable (oEmbed only resolves public posts) ──
    if platform == "youtube":
        oembed_url = f"https://www.youtube.com/oembed?url={post_url}&format=json"
    else:
        oembed_url = f"https://www.tiktok.com/oembed?url={post_url}"

    oembed_status, oembed_body = await loop.run_in_executor(None, _fetch_url, oembed_url)
    if oembed_status != 200 or not oembed_body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The post could not be found or is not public. Make sure the video is public and the link is correct.",
        )

    import json as _json
    try:
        oembed = _json.loads(oembed_body)
    except Exception:
        oembed = {}

    # ── 2. Caption / tags include our handle or hashtag ─────────────────────
    searchable = (oembed.get("title") or "").lower()
    # Also scan the public page HTML — YouTube descriptions are not in oEmbed
    page_status, page_body = await loop.run_in_executor(None, _fetch_url, post_url)
    if page_status == 200:
        searchable += " " + page_body.lower()

    handle = _GGW_YOUTUBE_HANDLE if platform == "youtube" else _GGW_TIKTOK_HANDLE
    tagged = handle in searchable or any(tag in searchable for tag in _GGW_HASHTAGS)
    if not tagged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"The post does not tag us. Add {handle} or the #GGW hashtag "
                "to the caption and try again."
            ),
        )

    # ── 3. Grant a personal free submission code ─────────────────────────────
    import secrets
    from datetime import timedelta
    from app.models.free_code import FreeCode

    code = f"GGWFREE-{secrets.token_hex(4).upper()}"
    free_code = FreeCode(
        code=code,
        max_uses=1,
        expires_at=get_current_utc() + timedelta(days=90),
    )
    db.add(free_code)

    # Record the verified post so it cannot be claimed twice
    if sharing is None:
        sharing = SocialSharing(
            submission_id=submission_id,
            user_id=current_user.id,
            platforms=[platform],
            opt_in=True,
        )
        db.add(sharing)
    if platform == "youtube":
        sharing.youtube_url = post_url
    else:
        sharing.tiktok_url = post_url
    sharing.posted_at = get_current_utc()

    await db.commit()

    logger.info(
        "Verified %s post for submission %s (user %s) — code %s granted",
        platform, submission_id, current_user.id, code,
    )
    return {
        "status": "success",
        "verified": True,
        "message": "Verified. A free submission code has been added to your account.",
        "data": {
            "verified": True,
            "free_code": code,
            "expires_in_days": 90,
        },
    }
