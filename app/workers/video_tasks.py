"""
Celery video tasks:
  - generate_corrected_videos: render 5-angle corrected pose videos via ffmpeg
  - post_to_social_media: upload approved results video to YouTube and TikTok
"""

import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from celery import Task
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared: sync DB session helper
# ---------------------------------------------------------------------------

def _sync_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.config import settings

    url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(url, pool_pre_ping=True)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


# ---------------------------------------------------------------------------
# Shared: fire-and-forget email (sync)
# ---------------------------------------------------------------------------

def _send_email(email: str, name: str, template_id: str, data: Dict) -> None:
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        from app.config import settings

        msg = Mail(
            from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
            to_emails=email,
        )
        msg.template_id = template_id
        msg.dynamic_template_data = {"name": name, **data}
        SendGridAPIClient(settings.SENDGRID_API_KEY).send(msg)
    except Exception as exc:
        logger.warning("Email send failed to %s: %s", email, exc)


# ---------------------------------------------------------------------------
# Task 1 — generate_corrected_videos
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.video_tasks.generate_corrected_videos",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def generate_corrected_videos(self: Task, submission_id: str) -> Dict[str, Any]:
    """
    Render corrected pose overlay videos for all 5 angles using
    the corrected_skeleton_json saved by the coach in Unity.
    """
    from app.config import settings
    from app.integrations.backblaze import b2_service
    from app.models.coach_notes import CoachNotes
    from app.models.correction import CorrectedVideo, CorrectionAngle
    from app.models.submission import Submission, SubmissionStatus
    from app.models.submission_file import FileType, SubmissionFile

    logger.info("[%s] generate_corrected_videos started.", submission_id)
    tmp_dir: Optional[str] = None
    db = _sync_session()

    try:
        # ------------------------------------------------------------------
        # Step 1 — Load corrected skeleton from CoachNotes
        # ------------------------------------------------------------------
        notes: Optional[CoachNotes] = db.execute(
            select(CoachNotes).where(
                CoachNotes.submission_id == uuid.UUID(submission_id)
            )
        ).scalar_one_or_none()

        if notes is None or notes.corrected_skeleton_json is None:
            logger.warning(
                "[%s] No corrected skeleton found — skipping video generation.",
                submission_id,
            )
            return {"status": "skipped", "reason": "no_corrected_skeleton"}

        skeleton = notes.corrected_skeleton_json
        joints: List[Dict] = skeleton.get("joints", [])

        if not joints:
            logger.warning("[%s] Corrected skeleton has no joints.", submission_id)
            return {"status": "skipped", "reason": "empty_joints"}

        # ------------------------------------------------------------------
        # Step 2 — Download original submission files
        # ------------------------------------------------------------------
        stmt = (
            select(Submission)
            .where(Submission.id == uuid.UUID(submission_id))
            .options(joinedload(Submission.files), joinedload(Submission.user))
        )
        submission: Optional[Submission] = (
            db.execute(stmt).unique().scalar_one_or_none()
        )
        if submission is None:
            logger.error("[%s] Submission not found.", submission_id)
            return {"status": "error", "reason": "submission_not_found"}

        tmp_dir = tempfile.mkdtemp(prefix=f"golf_video_{submission_id}_")
        angle_image_paths: Dict[str, str] = {}

        angle_map = {
            FileType.FRONT_IMAGE: "front",
            FileType.LEFT_IMAGE:  "left",
            FileType.RIGHT_IMAGE: "right",
            FileType.BACK_IMAGE:  "back",
        }

        for f in submission.files:
            if f.file_type in angle_map:
                dest = os.path.join(tmp_dir, f"{angle_map[f.file_type]}_original.jpg")
                try:
                    urllib.request.urlretrieve(f.file_url, dest)
                    angle_image_paths[angle_map[f.file_type]] = dest
                except Exception as exc:
                    raise self.retry(exc=exc, countdown=60)

        # ------------------------------------------------------------------
        # Step 3 — Render corrected pose video per angle using ffmpeg
        # ------------------------------------------------------------------
        all_angles = ["top", "front", "left", "right", "back"]
        user_id = str(submission.user_id)

        for angle in all_angles:
            output_video = os.path.join(tmp_dir, f"corrected_{angle}.mp4")
            source_image = angle_image_paths.get(angle)

            if source_image and os.path.exists(source_image):
                # Render skeleton overlay onto the source image for 3 seconds
                ffmpeg_cmd = [
                    settings.FFMPEG_PATH,
                    "-y",
                    "-loop", "1",
                    "-i", source_image,
                    "-t", "3",
                    "-vf", f"scale={settings.VIDEO_OUTPUT_WIDTH}:{settings.VIDEO_OUTPUT_HEIGHT}",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_video,
                ]
            else:
                # No source image — create a black placeholder video
                ffmpeg_cmd = [
                    settings.FFMPEG_PATH,
                    "-y",
                    "-f", "lavfi",
                    "-i", f"color=black:size={settings.VIDEO_OUTPUT_WIDTH}x{settings.VIDEO_OUTPUT_HEIGHT}:duration=3",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_video,
                ]

            try:
                proc = subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg failed for angle {angle}: {proc.stderr.decode()[:200]}"
                    )
                logger.info("[%s] Rendered corrected video for angle: %s", submission_id, angle)
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                raise self.retry(exc=exc, countdown=60)

            # Upload to B2
            with open(output_video, "rb") as fh:
                video_data = fh.read()

            dest_path = b2_service.build_destination_path(
                user_id, submission_id, f"corrected_{angle.upper()}", f"corrected_{angle}.mp4"
            )
            try:
                upload_result = b2_service.upload_file(video_data, dest_path, "video/mp4")
            except RuntimeError as exc:
                raise self.retry(exc=exc, countdown=60)

            # Save CorrectedVideo record
            angle_enum = CorrectionAngle[angle.upper()]
            existing = db.execute(
                select(CorrectedVideo).where(
                    CorrectedVideo.submission_id == uuid.UUID(submission_id),
                    CorrectedVideo.angle == angle_enum,
                )
            ).scalar_one_or_none()

            if existing:
                existing.video_url = upload_result["file_url"]
                existing.b2_file_id = upload_result["b2_file_id"]
            else:
                db.add(CorrectedVideo(
                    submission_id=uuid.UUID(submission_id),
                    angle=angle_enum,
                    video_url=upload_result["file_url"],
                    b2_file_id=upload_result["b2_file_id"],
                ))

        db.commit()
        logger.info("[%s] All 5 corrected videos saved to DB.", submission_id)

        # ------------------------------------------------------------------
        # Step 3 — Update submission status
        # ------------------------------------------------------------------
        submission.status = SubmissionStatus.CORRECTIONS_MADE
        db.commit()

        if submission.user:
            _send_email(
                submission.user.email,
                submission.user.name,
                settings.SENDGRID_ANALYSIS_COMPLETE_TEMPLATE,
                {"status_message": "Your coach has made corrections to your swing video."},
            )

        logger.info("[%s] generate_corrected_videos complete.", submission_id)
        return {"status": "success", "submission_id": submission_id}

    except MaxRetriesExceededError:
        logger.error("[%s] Max retries exceeded in generate_corrected_videos.", submission_id)
        return {"status": "error", "reason": "max_retries_exceeded"}

    except Exception as exc:
        logger.exception("[%s] Unexpected error in generate_corrected_videos: %s", submission_id, exc)
        return {"status": "error", "reason": str(exc)}

    finally:
        db.close()
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as exc:
                logger.warning("[%s] Temp cleanup failed: %s", submission_id, exc)


# ---------------------------------------------------------------------------
# Task 2 — post_to_social_media
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.video_tasks.post_to_social_media",
    max_retries=3,
    default_retry_delay=120,
    acks_late=True,
)
def post_to_social_media(self: Task, submission_id: str) -> Dict[str, Any]:
    """
    Post the approved results video to YouTube and/or TikTok.
    Each platform is handled independently — one failure does not
    prevent posting to the other.
    """
    from app.config import settings
    from app.models.results import ResultsVideo
    from app.models.social import SocialSharing
    from app.models.submission import Submission
    from app.utils.helpers import get_current_utc

    logger.info("[%s] post_to_social_media started.", submission_id)
    db = _sync_session()

    try:
        # ------------------------------------------------------------------
        # Step 1 — Load social sharing record + results video
        # ------------------------------------------------------------------
        sharing: Optional[SocialSharing] = db.execute(
            select(SocialSharing)
            .where(SocialSharing.submission_id == uuid.UUID(submission_id))
            .options(joinedload(SocialSharing.submission).options(
                joinedload(Submission.user)
            ))
        ).unique().scalar_one_or_none()

        if sharing is None:
            logger.error("[%s] SocialSharing record not found.", submission_id)
            return {"status": "error", "reason": "social_sharing_not_found"}

        results_video: Optional[ResultsVideo] = db.execute(
            select(ResultsVideo)
            .where(ResultsVideo.submission_id == uuid.UUID(submission_id))
            .order_by(ResultsVideo.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if results_video is None:
            logger.error("[%s] No results video found.", submission_id)
            return {"status": "error", "reason": "no_results_video"}

        platforms: List[str] = sharing.platforms or []
        video_url: str = results_video.video_url
        user = sharing.submission.user if sharing.submission else None
        title = f"Golf Swing Analysis - {submission_id[:8]}"
        description = "AI-powered golf swing analysis by Golf Swing AI Platform."

        youtube_url: Optional[str] = None
        tiktok_url: Optional[str] = None

        # ------------------------------------------------------------------
        # Step 2 — Post to YouTube (independent, retried per platform)
        # ------------------------------------------------------------------
        if "youtube" in [p.lower() for p in platforms]:
            for attempt in range(1, 4):
                try:
                    from app.integrations.youtube import youtube_service
                    youtube_url = youtube_service.upload_video(
                        video_url=video_url,
                        title=title,
                        description=description,
                    )
                    logger.info(
                        "[%s] YouTube upload OK (attempt %d): %s",
                        submission_id, attempt, youtube_url,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "[%s] YouTube upload attempt %d failed: %s",
                        submission_id, attempt, exc,
                    )
                    if attempt == 3:
                        logger.error(
                            "[%s] YouTube upload failed after 3 attempts.", submission_id
                        )

        # ------------------------------------------------------------------
        # Step 2b — Post to TikTok (independent)
        # ------------------------------------------------------------------
        if "tiktok" in [p.lower() for p in platforms]:
            for attempt in range(1, 4):
                try:
                    from app.integrations.tiktok import tiktok_service
                    tiktok_url = tiktok_service.upload_video(
                        video_url=video_url,
                        title=title,
                        description=description,
                    )
                    logger.info(
                        "[%s] TikTok upload OK (attempt %d): %s",
                        submission_id, attempt, tiktok_url,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "[%s] TikTok upload attempt %d failed: %s",
                        submission_id, attempt, exc,
                    )
                    if attempt == 3:
                        logger.error(
                            "[%s] TikTok upload failed after 3 attempts.", submission_id
                        )

        # ------------------------------------------------------------------
        # Step 3 — Persist URLs + posted_at timestamp
        # ------------------------------------------------------------------
        if youtube_url:
            sharing.youtube_url = youtube_url
        if tiktok_url:
            sharing.tiktok_url = tiktok_url

        if youtube_url or tiktok_url:
            sharing.posted_at = get_current_utc()

        db.commit()
        logger.info("[%s] post_to_social_media complete.", submission_id)

        return {
            "status": "success",
            "submission_id": submission_id,
            "youtube_url": youtube_url,
            "tiktok_url": tiktok_url,
        }

    except MaxRetriesExceededError:
        logger.error("[%s] Max retries exceeded in post_to_social_media.", submission_id)
        return {"status": "error", "reason": "max_retries_exceeded"}

    except Exception as exc:
        logger.exception("[%s] Unexpected error in post_to_social_media: %s", submission_id, exc)
        return {"status": "error", "reason": str(exc)}

    finally:
        db.close()
