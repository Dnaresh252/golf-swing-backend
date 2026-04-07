import asyncio
import logging
import math
import os
import subprocess
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.integrations.backblaze import b2_service
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_file import FileType, SubmissionFile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_MESSAGES: Dict[SubmissionStatus, str] = {
    SubmissionStatus.PENDING:          "Waiting for file uploads",
    SubmissionStatus.UPLOADING:        "Images uploaded, waiting for video",
    SubmissionStatus.ANALYZING:        "AI is analyzing your golf swing",
    SubmissionStatus.READY_FOR_REVIEW: "Ready for coach review",
    SubmissionStatus.IN_REVIEW:        "Coach is reviewing your swing",
    SubmissionStatus.CORRECTIONS_MADE: "Coach has made corrections",
    SubmissionStatus.COMPLETED:        "Analysis complete",
    SubmissionStatus.REJECTED:         "Submission was rejected",
}

_REQUIRED_IMAGE_TYPES = {
    FileType.FRONT_IMAGE,
    FileType.LEFT_IMAGE,
    FileType.RIGHT_IMAGE,
    FileType.BACK_IMAGE,
}

_DELETABLE_STATUSES = {SubmissionStatus.PENDING, SubmissionStatus.REJECTED}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SubmissionService:

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_submission(
        self, db: AsyncSession, user_id: uuid.UUID
    ) -> Submission:
        submission = Submission(user_id=user_id, status=SubmissionStatus.PENDING)
        db.add(submission)
        await db.flush()
        await db.refresh(submission)
        logger.info("Submission created: %s by user: %s", submission.id, user_id)
        return submission

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    async def get_submission(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Submission:
        """
        Fetch submission with files.  Returns 404 if not found or not owned —
        never leaks existence to a third party.
        """
        stmt = (
            select(Submission)
            .where(Submission.id == submission_id)
            .options(selectinload(Submission.files))
        )
        result = await db.execute(stmt)
        submission: Optional[Submission] = result.scalar_one_or_none()

        if submission is None or submission.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found.",
            )
        return submission

    async def get_user_submissions(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        page: int = 1,
        limit: int = 20,
    ) -> Dict:
        page = max(1, page)
        limit = max(1, min(limit, 100))
        offset = (page - 1) * limit

        count_result = await db.execute(
            select(func.count(Submission.id)).where(Submission.user_id == user_id)
        )
        total: int = count_result.scalar_one()

        stmt = (
            select(Submission)
            .where(Submission.user_id == user_id)
            .options(selectinload(Submission.files))
            .order_by(Submission.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        items = result.scalars().all()

        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": math.ceil(total / limit) if total else 0,
        }

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_status_message(self, sub_status: SubmissionStatus) -> str:
        return STATUS_MESSAGES.get(sub_status, "Unknown status")

    async def update_status(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        new_status: SubmissionStatus,
    ) -> Submission:
        result = await db.execute(
            select(Submission).where(Submission.id == submission_id)
        )
        submission: Optional[Submission] = result.scalar_one_or_none()
        if submission is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found.",
            )
        submission.status = new_status
        await db.flush()
        await db.refresh(submission)
        logger.info("Submission %s status → %s", submission_id, new_status.value)
        return submission

    # ------------------------------------------------------------------
    # Upload images  (all 4 at once, atomic B2 rollback on failure)
    # ------------------------------------------------------------------

    async def upload_images(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
        files: Dict[FileType, UploadFile],
    ) -> Submission:
        """
        Validate and upload all 4 swing images atomically.
        If any B2 upload fails, already-uploaded files are deleted and
        a 500 is raised — nothing is saved to the database.
        """
        submission = await self.get_submission(db, submission_id, user_id)

        if submission.status != SubmissionStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot upload to this submission in its current status.",
            )

        # ---- Phase 1: Validate ALL files before touching B2 ----
        contents: Dict[FileType, bytes] = {}
        for ft, upload in files.items():
            if upload.content_type not in settings.ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid format for {ft.value}. Allowed: JPEG, PNG.",
                )
            data = await upload.read()
            if len(data) > settings.max_image_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{ft.value} exceeds the maximum size of {settings.MAX_IMAGE_SIZE_MB} MB.",
                )
            contents[ft] = data

        # ---- Phase 2: Upload all 4 to B2; roll back on any failure ----
        uploaded: List[Tuple[str, str, str]] = []  # (b2_file_id, file_url, dest_path)

        try:
            for ft, data in contents.items():
                dest = b2_service.build_destination_path(
                    str(user_id),
                    str(submission_id),
                    ft.value,
                    files[ft].filename or "image",
                )
                result = b2_service.upload_file(data, dest, files[ft].content_type)
                uploaded.append((result["b2_file_id"], result["file_url"], dest))

        except RuntimeError as exc:
            # Roll back every file that made it to B2
            for b2_id, url, _ in uploaded:
                b2_service.delete_file(b2_id, url)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            )

        # ---- Phase 3: Persist records only after all uploads succeed ----
        for (b2_id, file_url, _), (ft, data) in zip(uploaded, contents.items()):
            db.add(SubmissionFile(
                submission_id=submission_id,
                file_type=ft,
                file_url=file_url,
                b2_file_id=b2_id,
                file_size_bytes=len(data),
                mime_type=files[ft].content_type,
            ))

        submission.status = SubmissionStatus.UPLOADING
        await db.flush()
        await db.refresh(submission)

        logger.info("Images uploaded for submission: %s", submission_id)
        return submission

    # ------------------------------------------------------------------
    # Upload video
    # ------------------------------------------------------------------

    async def upload_video(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
        file: UploadFile,
    ) -> Submission:
        submission = await self.get_submission(db, submission_id, user_id)

        if submission.status != SubmissionStatus.UPLOADING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload images first before uploading the video.",
            )

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

        self._check_video_duration(content)

        dest = b2_service.build_destination_path(
            str(user_id), str(submission_id), "SWING_VIDEO", file.filename or "video"
        )
        try:
            result = b2_service.upload_file(content, dest, file.content_type)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Upload failed, please try again.",
            )

        db.add(SubmissionFile(
            submission_id=submission_id,
            file_type=FileType.SWING_VIDEO,
            file_url=result["file_url"],
            b2_file_id=result["b2_file_id"],
            file_size_bytes=len(content),
            mime_type=file.content_type,
        ))
        await db.flush()
        await db.refresh(submission)

        logger.info("Video uploaded for submission: %s", submission_id)
        return submission

    # ------------------------------------------------------------------
    # Submit for analysis
    # ------------------------------------------------------------------

    async def submit_for_analysis(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Submission:
        submission = await self.get_submission(db, submission_id, user_id)
        uploaded_types = {f.file_type for f in submission.files}

        missing: List[str] = []
        for ft in _REQUIRED_IMAGE_TYPES:
            if ft not in uploaded_types:
                missing.append(ft.value)
        if FileType.SWING_VIDEO not in uploaded_types:
            missing.append(FileType.SWING_VIDEO.value)

        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing files: {', '.join(missing)}",
            )

        submission.status = SubmissionStatus.ANALYZING
        await db.flush()
        await db.refresh(submission)

        # Queue Celery task
        try:
            from app.workers.avatar_tasks import process_avatar_generation
            process_avatar_generation.delay(str(submission_id))
        except Exception as exc:
            logger.error("Failed to queue avatar task for %s: %s", submission_id, exc)

        # Non-blocking email
        asyncio.create_task(self._send_status_email(submission, "Your swing is being analyzed"))

        logger.info("Analysis triggered for submission: %s", submission_id)
        return submission

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_submission(
        self,
        db: AsyncSession,
        submission_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        submission = await self.get_submission(db, submission_id, user_id)

        if submission.status not in _DELETABLE_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete submission in current status.",
            )

        for f in submission.files:
            b2_service.delete_file(f.b2_file_id, f.file_url)

        await db.delete(submission)
        await db.flush()
        logger.info("Submission deleted: %s by user: %s", submission_id, user_id)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_video_duration(self, content: bytes) -> None:
        """Run ffprobe on a temp file; raise 400 if duration exceeds the limit."""
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            ffprobe = settings.FFMPEG_PATH.replace("ffmpeg", "ffprobe")
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    tmp_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            raw = proc.stdout.decode().strip()
            if not raw:
                raise ValueError("ffprobe returned no duration.")
            duration = float(raw)

        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not verify video duration. Please use a different file.",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if duration > settings.MAX_VIDEO_DURATION_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Video must be {settings.MAX_VIDEO_DURATION_SECONDS} seconds or less "
                    f"(yours is {duration:.1f}s)."
                ),
            )

    async def _send_status_email(self, submission: Submission, subject: str) -> None:
        """Send a status-change email via SendGrid.  Never raises."""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            from app.database import AsyncSessionLocal
            from app.models.user import User

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(User).where(User.id == submission.user_id)
                )
                user = result.scalar_one_or_none()
                if user is None:
                    return

            message = Mail(
                from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
                to_emails=user.email,
            )
            message.template_id = settings.SENDGRID_ANALYSIS_COMPLETE_TEMPLATE
            message.dynamic_template_data = {
                "name": user.name,
                "submission_id": str(submission.id),
                "status_message": subject,
            }
            SendGridAPIClient(settings.SENDGRID_API_KEY).send(message)
            logger.info("Status email sent to %s: %s", user.email, subject)
        except Exception as exc:
            logger.warning(
                "Failed to send status email for submission %s: %s",
                submission.id, exc,
            )


submission_service = SubmissionService()
