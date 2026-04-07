import logging
import math
import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.config import settings
from app.integrations.backblaze import b2_service
from app.models.coach import Coach
from app.models.coach_notes import CoachNotes
from app.models.correction import CorrectedVideo, CorrectionAngle
from app.models.discount import DiscountCode
from app.models.social import SocialSharing
from app.models.submission import Submission, SubmissionStatus
from app.models.user import User

logger = logging.getLogger(__name__)

_LOCK_TTL_SECONDS = 1800  # 30 minutes
_REVIEWABLE_STATUSES = {SubmissionStatus.READY_FOR_REVIEW, SubmissionStatus.IN_REVIEW}

_redis: Optional[aioredis.Redis] = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
    return _redis


def _lock_key(submission_id: str) -> str:
    return f"submission_lock:{submission_id}"


class CoachService:

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    async def get_queue(
        self,
        db: AsyncSession,
        coach_id: uuid.UUID,
        status_filter: Optional[str],
        page: int = 1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        page = max(1, page)
        limit = max(1, min(limit, 100))
        offset = (page - 1) * limit

        if status_filter:
            try:
                statuses = [SubmissionStatus(status_filter.upper())]
            except ValueError:
                statuses = list(_REVIEWABLE_STATUSES)
        else:
            statuses = list(_REVIEWABLE_STATUSES)

        base_filter = Submission.status.in_(statuses)

        count_result = await db.execute(
            select(func.count(Submission.id)).where(base_filter)
        )
        total: int = count_result.scalar_one()

        stmt = (
            select(Submission)
            .where(base_filter)
            .options(
                joinedload(Submission.user),
                selectinload(Submission.files),
            )
            .order_by(Submission.created_at.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        submissions = result.unique().scalars().all()

        items = []
        for sub in submissions:
            items.append({
                "id": sub.id,
                "user_name": sub.user.name if sub.user else "Unknown",
                "status": sub.status.value,
                "created_at": sub.created_at,
                "files_count": len(sub.files),
            })

        logger.info("Coach queue accessed by: %s", coach_id)
        return {
            "items": items,
            "total": total,
            "page": page,
            "pages": math.ceil(total / limit) if total else 0,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Submission lock / unlock
    # ------------------------------------------------------------------

    async def lock_submission(
        self, submission_id: str, coach_id: str
    ) -> bool:
        """
        Acquire a Redis lock for this submission.
        Returns True if lock acquired or already held by this coach.
        Returns False if held by a different coach.
        """
        r = _get_redis()
        key = _lock_key(submission_id)

        current = await r.get(key)
        if current is not None and current != coach_id:
            return False  # locked by someone else

        # SET NX — only sets if not already present
        await r.set(key, coach_id, ex=_LOCK_TTL_SECONDS, nx=False)
        return True

    async def is_submission_locked_by_coach(
        self, submission_id: str, coach_id: str
    ) -> bool:
        r = _get_redis()
        current = await r.get(_lock_key(submission_id))
        return current == coach_id

    async def release_lock(self, submission_id: str) -> None:
        r = _get_redis()
        await r.delete(_lock_key(submission_id))

    async def _assert_locked_by(
        self, submission_id: uuid.UUID, coach_id: uuid.UUID
    ) -> None:
        """Raise 423 if submission not locked by this coach."""
        if not await self.is_submission_locked_by_coach(
            str(submission_id), str(coach_id)
        ):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="You do not hold the lock on this submission. Open it from the queue first.",
            )

    # ------------------------------------------------------------------
    # Get submission for review (with lock)
    # ------------------------------------------------------------------

    async def get_submission_for_review(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        coach_id: uuid.UUID,
    ) -> Dict[str, Any]:
        # Check if already locked by a different coach
        r = _get_redis()
        current_holder = await r.get(_lock_key(str(submission_id)))
        if current_holder is not None and current_holder != str(coach_id):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Submission currently being reviewed by another coach.",
            )

        stmt = (
            select(Submission)
            .where(Submission.id == submission_id)
            .options(
                joinedload(Submission.user),
                selectinload(Submission.files),
                joinedload(Submission.avatar),
                joinedload(Submission.coach_notes),
            )
        )
        result = await db.execute(stmt)
        submission: Optional[Submission] = result.unique().scalar_one_or_none()

        if submission is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found.",
            )
        if submission.status not in _REVIEWABLE_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Submission is not available for review.",
            )

        # Acquire lock + assign coach + set IN_REVIEW
        acquired = await self.lock_submission(str(submission_id), str(coach_id))
        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Submission currently being reviewed by another coach.",
            )

        submission.coach_id = coach_id
        submission.status = SubmissionStatus.IN_REVIEW
        await db.flush()
        await db.refresh(submission)

        logger.info("Submission %s locked by coach %s", submission_id, coach_id)

        avatar = submission.avatar
        angle_views = {}
        skeleton_json = None
        if avatar:
            angle_views = {
                "top":   avatar.view_top_url,
                "front": avatar.view_front_url,
                "left":  avatar.view_left_url,
                "right": avatar.view_right_url,
                "back":  avatar.view_back_url,
            }
            skeleton_json = avatar.skeleton_json

        files_by_type = {f.file_type.value: f.file_url for f in submission.files}
        notes = submission.coach_notes

        return {
            "submission": submission,
            "files": files_by_type,
            "avatar_status": avatar.status.value if avatar else None,
            "avatar_glb_url": avatar.avatar_glb_url if avatar else None,
            "angle_views": angle_views,
            "skeleton_json": skeleton_json,
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # Notes (upsert)
    # ------------------------------------------------------------------

    async def save_notes(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        coach_id: uuid.UUID,
        notes_data: Dict[str, Any],
    ) -> CoachNotes:
        await self._assert_locked_by(submission_id, coach_id)

        result = await db.execute(
            select(CoachNotes).where(CoachNotes.submission_id == submission_id)
        )
        notes: Optional[CoachNotes] = result.scalar_one_or_none()

        now = datetime.now(timezone.utc)

        if notes is None:
            notes = CoachNotes(
                submission_id=submission_id,
                coach_id=coach_id,
                auto_saved_at=now,
            )
            db.add(notes)

        if notes_data.get("notes_text") is not None:
            notes.notes_text = notes_data["notes_text"]
        if notes_data.get("recommendations") is not None:
            notes.recommendations = notes_data["recommendations"]
        if notes_data.get("corrected_skeleton_json") is not None:
            notes.corrected_skeleton_json = notes_data["corrected_skeleton_json"]

        notes.auto_saved_at = now

        await db.flush()
        await db.refresh(notes)
        logger.info("Notes saved for submission: %s", submission_id)
        return notes

    # ------------------------------------------------------------------
    # Upload corrected video
    # ------------------------------------------------------------------

    async def upload_corrected_video(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        coach_id: uuid.UUID,
        angle: str,
        file: UploadFile,
    ) -> CorrectedVideo:
        await self._assert_locked_by(submission_id, coach_id)

        if file.content_type not in settings.ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid video format. Allowed: MP4, MOV.",
            )

        content = await file.read()
        if len(content) > settings.max_video_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video exceeds the maximum size of {settings.MAX_VIDEO_SIZE_MB} MB.",
            )

        # Load submission to get user_id for path building
        sub_result = await db.execute(
            select(Submission).where(Submission.id == submission_id)
        )
        submission = sub_result.scalar_one_or_none()
        if submission is None:
            raise HTTPException(status_code=404, detail="Submission not found.")

        dest = b2_service.build_destination_path(
            str(submission.user_id),
            str(submission_id),
            f"corrected_{angle.upper()}",
            file.filename or "corrected.mp4",
        )
        try:
            upload_result = b2_service.upload_file(content, dest, file.content_type)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video upload failed. Please try again.",
            )

        angle_enum = CorrectionAngle[angle.upper()]
        video = CorrectedVideo(
            submission_id=submission_id,
            angle=angle_enum,
            video_url=upload_result["file_url"],
            b2_file_id=upload_result["b2_file_id"],
        )
        db.add(video)
        await db.flush()
        await db.refresh(video)

        logger.info(
            "Corrected video uploaded: %s angle: %s", submission_id, angle
        )
        return video

    # ------------------------------------------------------------------
    # Approve / reject
    # ------------------------------------------------------------------

    async def approve_submission(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        coach_id: uuid.UUID,
    ) -> bool:
        await self._assert_locked_by(submission_id, coach_id)

        # Verify notes or corrected video exists
        notes_result = await db.execute(
            select(CoachNotes).where(CoachNotes.submission_id == submission_id)
        )
        notes = notes_result.scalar_one_or_none()
        has_notes = notes is not None and (
            bool(notes.notes_text) or bool(notes.recommendations)
        )

        from sqlalchemy import exists
        video_exists = await db.execute(
            select(CorrectedVideo.id).where(
                CorrectedVideo.submission_id == submission_id
            ).limit(1)
        )
        has_video = video_exists.scalar_one_or_none() is not None

        if not has_notes and not has_video:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Add notes or corrected video before approving.",
            )

        sub_result = await db.execute(
            select(Submission)
            .where(Submission.id == submission_id)
            .options(joinedload(Submission.user))
        )
        submission = sub_result.unique().scalar_one_or_none()
        if submission is None:
            raise HTTPException(status_code=404, detail="Submission not found.")

        submission.status = SubmissionStatus.CORRECTIONS_MADE
        await db.flush()
        await self.release_lock(str(submission_id))

        if submission.user:
            _send_email_nowait(
                submission.user.email,
                submission.user.name,
                settings.SENDGRID_COACH_APPROVAL_TEMPLATE,
                {"status_message": "Coach has reviewed your swing and made corrections."},
            )

        logger.info("Submission approved by coach: %s", coach_id)
        return True

    async def reject_submission(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        coach_id: uuid.UUID,
        reason: str,
    ) -> bool:
        await self._assert_locked_by(submission_id, coach_id)

        sub_result = await db.execute(
            select(Submission)
            .where(Submission.id == submission_id)
            .options(joinedload(Submission.user))
        )
        submission = sub_result.unique().scalar_one_or_none()
        if submission is None:
            raise HTTPException(status_code=404, detail="Submission not found.")

        submission.status = SubmissionStatus.REJECTED
        await db.flush()
        await self.release_lock(str(submission_id))

        if submission.user:
            _send_email_nowait(
                submission.user.email,
                submission.user.name,
                settings.SENDGRID_COACH_APPROVAL_TEMPLATE,
                {
                    "status_message": "Your submission has been rejected.",
                    "rejection_reason": reason,
                },
            )

        logger.info("Submission rejected by coach: %s", coach_id)
        return True

    # ------------------------------------------------------------------
    # Results queue
    # ------------------------------------------------------------------

    async def get_results_queue(
        self,
        db: AsyncSession,
        coach_id: uuid.UUID,
        page: int = 1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        page = max(1, page)
        limit = max(1, min(limit, 100))
        offset = (page - 1) * limit

        base_filter = (
            (SocialSharing.opt_in == True) &  # noqa: E712
            (SocialSharing.posted_at == None)  # noqa: E711
        )

        count_result = await db.execute(
            select(func.count(SocialSharing.id)).where(base_filter)
        )
        total: int = count_result.scalar_one()

        stmt = (
            select(SocialSharing)
            .where(base_filter)
            .options(
                joinedload(SocialSharing.submission).options(
                    joinedload(Submission.user)
                )
            )
            .order_by(SocialSharing.created_at.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        sharings = result.unique().scalars().all()

        logger.info("Results queue accessed by: %s", coach_id)
        return {
            "items": sharings,
            "total": total,
            "page": page,
            "pages": math.ceil(total / limit) if total else 0,
            "limit": limit,
        }

    # ------------------------------------------------------------------
    # Approve / reject social posting
    # ------------------------------------------------------------------

    async def approve_for_posting(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        coach_id: uuid.UUID,
    ) -> Dict[str, Any]:
        sharing_result = await db.execute(
            select(SocialSharing)
            .where(SocialSharing.submission_id == submission_id)
            .options(joinedload(SocialSharing.submission).options(
                joinedload(Submission.user)
            ))
        )
        sharing: Optional[SocialSharing] = sharing_result.unique().scalar_one_or_none()

        if sharing is None or not sharing.opt_in:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No pending social sharing request for this submission.",
            )

        # Trigger social posting task
        try:
            from app.workers.video_tasks import post_to_social_media
            post_to_social_media.delay(str(submission_id))
        except Exception as exc:
            logger.error("Failed to queue social post task for %s: %s", submission_id, exc)

        # Generate discount code for user
        user = sharing.submission.user if sharing.submission else None
        discount_code = None
        if user:
            discount_code = await self._create_discount_code(db, user.id, submission_id)

        # Send email
        if user:
            _send_email_nowait(
                user.email,
                user.name,
                settings.SENDGRID_DISCOUNT_EARNED_TEMPLATE,
                {
                    "status_message": "Your video has been approved for posting!",
                    "discount_code": discount_code,
                    "discount_percent": settings.DISCOUNT_PERCENTAGE,
                },
            )

        logger.info("Results approved for posting: %s", submission_id)
        return {
            "message": "Approved for social posting.",
            "discount_code": discount_code,
        }

    async def reject_posting(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        coach_id: uuid.UUID,
        reason: str,
    ) -> bool:
        sharing_result = await db.execute(
            select(SocialSharing)
            .where(SocialSharing.submission_id == submission_id)
            .options(joinedload(SocialSharing.submission).options(
                joinedload(Submission.user)
            ))
        )
        sharing: Optional[SocialSharing] = sharing_result.unique().scalar_one_or_none()

        if sharing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Social sharing record not found.",
            )

        sharing.opt_in = False
        await db.flush()

        user = sharing.submission.user if sharing.submission else None
        if user:
            _send_email_nowait(
                user.email,
                user.name,
                settings.SENDGRID_COACH_APPROVAL_TEMPLATE,
                {
                    "status_message": "Your video was not approved for social media posting.",
                    "rejection_reason": reason,
                },
            )

        logger.info("Results posting rejected: %s", submission_id)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _create_discount_code(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        submission_id: uuid.UUID,
    ) -> str:
        """Generate a unique discount code and persist it."""
        for _ in range(10):
            suffix = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=5)
            )
            code = f"{settings.DISCOUNT_CODE_PREFIX}-{suffix}"
            existing = await db.execute(
                select(DiscountCode.id).where(DiscountCode.code == code).limit(1)
            )
            if existing.scalar_one_or_none() is None:
                break

        valid_until = datetime.now(timezone.utc) + timedelta(
            days=settings.DISCOUNT_VALIDITY_DAYS
        )
        discount = DiscountCode(
            user_id=user_id,
            submission_id=submission_id,
            code=code,
            discount_percent=settings.DISCOUNT_PERCENTAGE,
            valid_until=valid_until,
        )
        db.add(discount)
        await db.flush()
        return code


# ---------------------------------------------------------------------------
# Module-level email helper (fire-and-forget, never raises)
# ---------------------------------------------------------------------------

def _send_email_nowait(
    email: str, name: str, template_id: str, data: Dict[str, Any]
) -> None:
    import asyncio

    async def _send():
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            msg = Mail(
                from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
                to_emails=email,
            )
            msg.template_id = template_id
            msg.dynamic_template_data = {"name": name, **data}
            SendGridAPIClient(settings.SENDGRID_API_KEY).send(msg)
            logger.info("Email sent to %s (template=%s)", email, template_id)
        except Exception as exc:
            logger.warning("Failed to send email to %s: %s", email, exc)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_send())
        else:
            loop.run_until_complete(_send())
    except Exception as exc:
        logger.warning("_send_email_nowait scheduling failed: %s", exc)


coach_service = CoachService()
