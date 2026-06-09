"""
Celery task: full avatar generation pipeline.

Pipeline steps:
  1. Download submission files from B2
  2. Run Specialist 3 ML pipeline (skeleton + avatar + renders in one call)
  3. Upload avatar model files to B2
  4. Upload angle-view PNGs to B2
  5. Persist URLs + skeleton JSON, advance statuses
  6. Cleanup temp files (always, in finally)
"""

import logging
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional

from celery import Task
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import select

from app.integrations.backblaze import b2_service
from app.integrations.mediapipe_client import mediapipe_client
from app.models.avatar import Avatar, AvatarStatus
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_file import FileType, SubmissionFile
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: synchronous DB session (Celery tasks are sync)
# ---------------------------------------------------------------------------

def _sync_session():
    """Return a synchronous SQLAlchemy session for use inside Celery tasks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.config import settings

    # Celery workers need a sync engine — strip asyncpg driver if present
    url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return Session()


# ---------------------------------------------------------------------------
# Helper: send status email (fire-and-forget, never raises)
# ---------------------------------------------------------------------------

def _send_email_sync(user_email: str, user_name: str, template_id: str, data: Dict) -> None:
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        from app.config import settings

        msg = Mail(
            from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
            to_emails=user_email,
        )
        msg.template_id = template_id
        msg.dynamic_template_data = {"name": user_name, **data}
        SendGridAPIClient(settings.SENDGRID_API_KEY).send(msg)
        logger.info("Email sent to %s (template=%s)", user_email, template_id)
    except Exception as exc:
        logger.warning("Failed to send email to %s: %s", user_email, exc)


# ---------------------------------------------------------------------------
# Helper: mark avatar FAILED and notify user
# ---------------------------------------------------------------------------

def _fail_avatar(
    db,
    submission_id: str,
    error_message: str,
) -> None:
    from app.config import settings

    try:
        avatar = db.execute(
            select(Avatar).where(Avatar.submission_id == uuid.UUID(submission_id))
        ).scalar_one_or_none()

        if avatar:
            avatar.status = AvatarStatus.FAILED
            avatar.error_message = error_message
            db.commit()

        # Notify user
        sub = db.execute(
            select(Submission).where(Submission.id == uuid.UUID(submission_id))
        ).scalar_one_or_none()
        if sub and sub.user:
            _send_email_sync(
                sub.user.email,
                sub.user.name,
                settings.SENDGRID_ANALYSIS_COMPLETE_TEMPLATE,
                {"status_message": "We encountered an issue analysing your swing. Please re-submit."},
            )
    except Exception as exc:
        logger.error("_fail_avatar error for %s: %s", submission_id, exc)


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.avatar_tasks.process_avatar_generation",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def process_avatar_generation(self: Task, submission_id: str) -> Dict[str, Any]:
    """
    Full avatar generation pipeline for a single submission.
    Retries (up to 3 ×) only on network-related errors.
    """
    from app.config import settings
    from app.models.user import User
    from sqlalchemy.orm import joinedload

    logger.info("[%s] Avatar pipeline started.", submission_id)
    tmp_dir: Optional[str] = None
    db = _sync_session()

    try:
        # ----------------------------------------------------------------
        # Step 1 — Load submission + files from DB, download from B2
        # ----------------------------------------------------------------
        stmt = (
            select(Submission)
            .where(Submission.id == uuid.UUID(submission_id))
            .options(
                joinedload(Submission.files),
                joinedload(Submission.avatar),
                joinedload(Submission.user),
            )
        )
        submission: Optional[Submission] = db.execute(stmt).unique().scalar_one_or_none()

        if submission is None:
            logger.error("[%s] Submission not found — aborting.", submission_id)
            return {"status": "error", "reason": "submission_not_found"}

        # Ensure avatar record exists
        avatar: Optional[Avatar] = submission.avatar
        if avatar is None:
            avatar = Avatar(
                submission_id=uuid.UUID(submission_id),
                status=AvatarStatus.PROCESSING,
            )
            db.add(avatar)
        else:
            avatar.status = AvatarStatus.PROCESSING
        db.commit()
        db.refresh(avatar)

        tmp_dir = tempfile.mkdtemp(prefix=f"golf_{submission_id}_")
        logger.info("[%s] Temp dir: %s", submission_id, tmp_dir)

        image_paths: Dict[str, str] = {}
        video_path: Optional[str] = None

        for f in submission.files:
            dest = os.path.join(tmp_dir, f"{f.file_type.value}_{f.id}")
            try:
                file_bytes = b2_service.download_file_bytes(f.file_url)
                with open(dest, "wb") as fh:
                    fh.write(file_bytes)
            except Exception as exc:
                # Network error — eligible for retry
                raise self.retry(exc=exc, countdown=60)

            if f.file_type == FileType.SWING_VIDEO:
                video_path = dest
            else:
                angle_map = {
                    FileType.FRONT_IMAGE: "front",
                    FileType.LEFT_IMAGE:  "left",
                    FileType.RIGHT_IMAGE: "right",
                    FileType.BACK_IMAGE:  "back",
                }
                image_paths[angle_map[f.file_type]] = dest

        logger.info(
            "[%s] Step 1 complete — %d images, video=%s",
            submission_id, len(image_paths), video_path is not None,
        )

        # ----------------------------------------------------------------
        # Step 2 — Run full ML pipeline (skeleton + avatar + renders)
        # ----------------------------------------------------------------
        try:
            pipeline_result = mediapipe_client.run_full_pipeline(
                submission_id,
                image_paths,
                video_path or "",
            )
        except ValueError as exc:
            logger.error("[%s] Step 2 ML pipeline failed: %s", submission_id, exc)
            _fail_avatar(db, submission_id, str(exc))
            return {"status": "error", "reason": "ml_pipeline_failed"}

        skeleton_data  = pipeline_result["skeleton_json"]
        avatar_paths   = pipeline_result["avatar_paths"]   # {obj, glb, fbx}
        render_paths   = pipeline_result["render_paths"]   # {top, front, left, right, back}
        warnings       = pipeline_result.get("analysis_warnings", [])

        if warnings:
            logger.warning("[%s] Pipeline warnings: %s", submission_id, warnings)

        logger.info(
            "[%s] Step 2 complete — pipeline finished, %d warning(s).",
            submission_id, len(warnings),
        )

        # ----------------------------------------------------------------
        # Step 3 — Upload avatar model files to B2
        # ----------------------------------------------------------------
        user_id = str(submission.user_id)
        model_urls: Dict[str, str] = {}

        for fmt, local_path in avatar_paths.items():
            if not local_path or not os.path.exists(local_path):
                continue
            with open(local_path, "rb") as fh:
                data = fh.read()
            ext = os.path.splitext(local_path)[1]
            dest = b2_service.build_destination_path(
                user_id, submission_id, f"avatar_{fmt}", f"avatar{ext}"
            )
            try:
                result = b2_service.upload_file(data, dest, f"model/{ext.lstrip('.')}")
                model_urls[fmt] = result["file_url"]
            except RuntimeError as exc:
                raise self.retry(exc=exc, countdown=60)

        logger.info("[%s] Step 3 complete — avatar model files uploaded.", submission_id)

        # ----------------------------------------------------------------
        # Step 4 — Upload angle-view PNGs to B2
        # ----------------------------------------------------------------
        angle_urls: Dict[str, str] = {}

        for angle, local_path in render_paths.items():
            if not local_path or not os.path.exists(local_path):
                continue
            with open(local_path, "rb") as fh:
                data = fh.read()
            dest = b2_service.build_destination_path(
                user_id, submission_id, f"view_{angle}", "view.png"
            )
            try:
                result = b2_service.upload_file(data, dest, "image/png")
                angle_urls[angle] = result["file_url"]
            except RuntimeError as exc:
                raise self.retry(exc=exc, countdown=60)

        logger.info("[%s] Step 4 complete — angle-view PNGs uploaded.", submission_id)

        # ----------------------------------------------------------------
        # Step 5 — Persist everything
        # ----------------------------------------------------------------
        avatar.skeleton_json  = skeleton_data
        avatar.avatar_obj_url = model_urls.get("obj")
        avatar.avatar_fbx_url = model_urls.get("fbx")
        avatar.avatar_glb_url = model_urls.get("glb")
        avatar.view_top_url   = angle_urls.get("top")
        avatar.view_front_url = angle_urls.get("front")
        avatar.view_left_url  = angle_urls.get("left")
        avatar.view_right_url = angle_urls.get("right")
        avatar.view_back_url  = angle_urls.get("back")
        avatar.status         = AvatarStatus.COMPLETED
        avatar.error_message  = None

        submission.status = SubmissionStatus.READY_FOR_REVIEW
        db.commit()

        logger.info("[%s] Step 5 complete — DB updated, status=READY_FOR_REVIEW.", submission_id)

        # Send success email
        if submission.user:
            _send_email_sync(
                submission.user.email,
                submission.user.name,
                settings.SENDGRID_ANALYSIS_COMPLETE_TEMPLATE,
                {"status_message": "Your avatar is ready for coach review."},
            )

        logger.info("[%s] Avatar generation complete.", submission_id)
        return {"status": "success", "submission_id": submission_id}

    except MaxRetriesExceededError:
        logger.error("[%s] Max retries exceeded — marking avatar FAILED.", submission_id)
        _fail_avatar(db, submission_id, "Processing failed after multiple retries.")
        return {"status": "error", "reason": "max_retries_exceeded"}

    except Exception as exc:
        logger.exception("[%s] Unexpected error in avatar pipeline: %s", submission_id, exc)
        _fail_avatar(db, submission_id, f"Unexpected error: {exc}")
        return {"status": "error", "reason": str(exc)}

    finally:
        # Step 6 — always clean up temp files
        db.close()
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
                logger.info("[%s] Temp dir cleaned up.", submission_id)
            except Exception as exc:
                logger.warning("[%s] Failed to clean temp dir: %s", submission_id, exc)
