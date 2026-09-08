"""
Instructor picker background work.

One job: when a user asks for a specific instructor, that request carries a
deadline. If the instructor has not started the review by then, the request
lapses and the submission goes back to the general queue so it is not stuck
waiting on someone who is not coming.

Deliberately quiet: no notification, no refund, no status change. The user
was told upfront that another instructor may pick the swing up.
"""
import asyncio
import logging

from sqlalchemy import select, update

from app.database import AsyncSessionLocal
from app.models.submission import Submission, SubmissionStatus
from app.utils.helpers import get_current_utc
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Statuses where a review has NOT started yet. Once an instructor is
# actually working on it, the request has been honoured and the deadline
# stops mattering.
_NOT_STARTED = (
    SubmissionStatus.READY_FOR_REVIEW,
    SubmissionStatus.ANALYZING,
    SubmissionStatus.PENDING,
    SubmissionStatus.UPLOADING,
)


async def _expire() -> int:
    now = get_current_utc()
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Submission.id).where(
                Submission.requested_coach_id.isnot(None),
                Submission.instructor_request_expires_at.isnot(None),
                Submission.instructor_request_expires_at < now,
                Submission.status.in_(_NOT_STARTED),
                Submission.coach_id.is_(None),
            )
        )
        ids = [row[0] for row in result.all()]
        if not ids:
            return 0

        await db.execute(
            update(Submission)
            .where(Submission.id.in_(ids))
            .values(requested_coach_id=None, instructor_request_expires_at=None)
        )
        await db.commit()
        return len(ids)


@celery_app.task(name="app.workers.instructor_tasks.expire_instructor_requests")
def expire_instructor_requests() -> dict:
    try:
        count = asyncio.run(_expire())
    except Exception as exc:  # noqa: BLE001 - a beat task must never die silently
        logger.exception("expire_instructor_requests failed: %s", exc)
        return {"expired": 0, "error": str(exc)}

    if count:
        logger.info("Instructor requests expired and returned to queue: %d", count)
    return {"expired": count}
