"""
Account deletion: a 7-day grace period, then the person is removed.

Agreed with Stan, 2026-09-14:
  - Asking to delete schedules it seven days out. Logging back in cancels it.
  - Not allowed while a swing is with an instructor; it has to be released
    first. There is no refund path.
  - After the grace period everything that describes the person goes: name,
    email, password, profile, and every photo, video, 3D model and skeleton
    of them, in Backblaze and in the local outputs directory.
  - Payment and submission rows stay, anonymised, so instructor payouts and
    the books still reconcile. They can stay indefinitely because nothing
    personal is left on them.

The users row itself is kept as a tombstone rather than deleted. Payments
and submissions cascade from users.id, so deleting the row would delete the
very records accounting needs.
"""
import logging
import secrets
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.backblaze import b2_service
from app.models.audit_log import AuditLog
from app.models.avatar import Avatar
from app.models.coach_notes import CoachNotes
from app.models.correction import CorrectedVideo
from app.models.discount import DiscountCode
from app.models.free_code import FreeCode
from app.models.results import ResultsVideo
from app.models.social import SocialSharing
from app.models.submission import Submission, SubmissionStatus
from app.models.submission_file import SubmissionFile
from app.models.user import User
from app.utils.security import hash_password

logger = logging.getLogger(__name__)

GRACE_PERIOD = timedelta(days=7)

# A swing in any of these states is being analysed or is with an instructor.
# Deleting then would pull the photos and video out from under the review.
BLOCKING_STATUSES = (
    SubmissionStatus.ANALYZING,
    SubmissionStatus.READY_FOR_REVIEW,
    SubmissionStatus.IN_REVIEW,
    SubmissionStatus.PGA_APPROVAL,
    SubmissionStatus.CORRECTIONS_MADE,
)

BLOCKED_MESSAGE = (
    "One of your swings is still being analyzed or reviewed by an instructor. "
    "You can delete your account once it has been released to you."
)

# The analysis engine writes one directory per submission here.
OUTPUTS_ROOT = Path(__file__).resolve().parents[2] / "outputs"

_AVATAR_URL_COLUMNS = (
    "avatar_obj_url", "avatar_fbx_url", "avatar_glb_url",
    "view_top_url", "view_front_url", "view_left_url", "view_right_url", "view_back_url",
)


class StorageCleanupError(Exception):
    """Some stored files could not be deleted. Nothing in the database was changed."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def scheduled_for(user: User) -> Optional[datetime]:
    if user.deletion_requested_at is None:
        return None
    return user.deletion_requested_at + GRACE_PERIOD


def tombstone_email(user_id: uuid.UUID) -> str:
    return f"deleted-{user_id.hex}@deleted.invalid"


async def blocking_submission_count(db: AsyncSession, user_id: uuid.UUID) -> int:
    return (
        await db.scalar(
            select(func.count(Submission.id)).where(
                Submission.user_id == user_id,
                Submission.status.in_(BLOCKING_STATUSES),
            )
        )
    ) or 0


async def request_deletion(db: AsyncSession, user: User) -> datetime:
    """Start the grace period. Asking twice keeps the original date."""
    if user.deletion_requested_at is None:
        user.deletion_requested_at = _now()
        db.add(AuditLog(action="account_deletion_requested", detail=f"user_id={user.id}"))
        await db.flush()
        logger.info("Account deletion requested: %s", user.id)
    return scheduled_for(user)


def cancel_deletion(db: AsyncSession, user: User) -> None:
    user.deletion_requested_at = None
    db.add(AuditLog(action="account_deletion_cancelled", detail=f"user_id={user.id}"))
    logger.info("Account deletion cancelled by login: %s", user.id)


def _file_id_from_url(url: Optional[str]) -> Optional[str]:
    # Stored URLs are b2_download_file_by_id links: the id is the fileId param.
    if not url:
        return None
    try:
        return parse_qs(urlparse(url).query).get("fileId", [None])[0]
    except Exception:
        return None


async def purge_user(db: AsyncSession, user: User, *, reason: str) -> dict:
    """
    Remove the person and anonymise what has to be kept.

    Stored files are deleted first. If any deletion fails, StorageCleanupError
    is raised before the database is touched, so the rows that point at the
    remaining files survive and a retry can finish the job. Deleting a file
    that is already gone counts as success, which makes a retry safe.

    The caller commits.
    """
    sub_ids = list(
        (await db.execute(select(Submission.id).where(Submission.user_id == user.id))).scalars()
    )

    file_ids: set[str] = set()
    if sub_ids:
        for model, url_col in (
            (SubmissionFile, SubmissionFile.file_url),
            (ResultsVideo, ResultsVideo.video_url),
            (CorrectedVideo, CorrectedVideo.video_url),
        ):
            rows = await db.execute(
                select(model.b2_file_id, url_col).where(model.submission_id.in_(sub_ids))
            )
            for b2_id, url in rows:
                for fid in (b2_id, _file_id_from_url(url)):
                    if fid:
                        file_ids.add(fid)
        avatars = (
            await db.execute(select(Avatar).where(Avatar.submission_id.in_(sub_ids)))
        ).scalars()
        for avatar in avatars:
            for col in _AVATAR_URL_COLUMNS:
                fid = _file_id_from_url(getattr(avatar, col))
                if fid:
                    file_ids.add(fid)
    fid = _file_id_from_url(user.profile_picture_url)
    if fid:
        file_ids.add(fid)

    storage = {"deleted": 0, "missing": 0, "failed": 0}
    for fid in sorted(file_ids):
        storage[b2_service.delete_file_by_id(fid)] += 1
    if storage["failed"]:
        logger.error("Account purge %s: %d stored file(s) could not be deleted", user.id, storage["failed"])
        raise StorageCleanupError(f"{storage['failed']} stored file(s) could not be deleted")

    local_dirs = 0
    for sid in sub_ids:
        path = OUTPUTS_ROOT / str(sid)
        if path.is_dir():
            shutil.rmtree(path)
            local_dirs += 1

    if sub_ids:
        for model in (SubmissionFile, ResultsVideo, CorrectedVideo, Avatar):
            await db.execute(delete(model).where(model.submission_id.in_(sub_ids)))
        # Notes and payout records stay; the corrected skeleton is the
        # golfer's own movement, so it goes.
        await db.execute(
            update(CoachNotes)
            .where(CoachNotes.submission_id.in_(sub_ids))
            .values(corrected_skeleton_json=None)
        )
        await db.execute(
            update(Submission).where(Submission.id.in_(sub_ids)).values(avatar_skin_tone=None)
        )

    await db.execute(delete(SocialSharing).where(SocialSharing.user_id == user.id))
    await db.execute(
        delete(DiscountCode).where(DiscountCode.user_id == user.id, DiscountCode.used.is_(False))
    )
    await db.execute(update(FreeCode).where(FreeCode.user_id == user.id).values(active=False))

    now = _now()
    user.email = tombstone_email(user.id)
    user.name = "Deleted user"
    user.password_hash = hash_password(secrets.token_urlsafe(48))
    user.profile_picture_url = None
    user.bio = None
    user.home_club = None
    user.handicap_index = None
    user.public_profile_enabled = False
    user.is_active = False
    user.is_verified = False
    user.last_login_at = None
    user.deleted_at = now

    summary = {
        "submissions_kept": len(sub_ids),
        "files_deleted": storage["deleted"],
        "files_already_gone": storage["missing"],
        "local_dirs_removed": local_dirs,
    }
    db.add(AuditLog(
        action="account_deleted",
        detail=f"user_id={user.id} reason={reason} "
               + " ".join(f"{k}={v}" for k, v in summary.items()),
    ))
    await db.flush()
    logger.info("Account purged: %s (%s) %s", user.id, reason, summary)
    return summary
