"""
Celery notification tasks.

send_email_notification  — one-off transactional email by template name.
send_bulk_reminder_emails — scheduled every 6 h; reminds coaches of stale reviews.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Map human-readable template names to settings attributes
_TEMPLATE_MAP = {
    "welcome":            "SENDGRID_WELCOME_TEMPLATE",
    "upload_confirm":     "SENDGRID_UPLOAD_CONFIRM_TEMPLATE",
    "analysis_complete":  "SENDGRID_ANALYSIS_COMPLETE_TEMPLATE",
    "coach_approval":     "SENDGRID_COACH_APPROVAL_TEMPLATE",
    "discount_earned":    "SENDGRID_DISCOUNT_EARNED_TEMPLATE",
}


def _sync_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.config import settings

    url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(url, pool_pre_ping=True)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


# ---------------------------------------------------------------------------
# Task 1: send_email_notification
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_email_notification",
    bind=False,
    acks_late=True,
    max_retries=3,
    default_retry_delay=30,
)
def send_email_notification(
    user_id: str,
    template_name: str,
    template_data: Dict[str, Any],
) -> None:
    """
    Load user by *user_id*, resolve the SendGrid template ID for
    *template_name*, and deliver the email.  Never raises.
    """
    from sqlalchemy import select
    from app.config import settings
    from app.integrations.sendgrid import sendgrid_service
    from app.models.user import User

    db = _sync_session()
    try:
        user = db.execute(
            select(User).where(User.id == user_id)
        ).scalar_one_or_none()

        if user is None:
            logger.warning(
                "send_email_notification: user %s not found — skipping.", user_id
            )
            return

        template_attr = _TEMPLATE_MAP.get(template_name)
        if not template_attr:
            logger.warning(
                "send_email_notification: unknown template_name '%s'.", template_name
            )
            return

        template_id: str = getattr(settings, template_attr, "")
        if not template_id:
            logger.warning(
                "send_email_notification: template ID not configured for '%s'.",
                template_name,
            )
            return

        merged_data = {"name": user.name, **template_data}
        sendgrid_service._send_email(
            to_email=user.email,
            template_id=template_id,
            dynamic_data=merged_data,
        )
        logger.info(
            "Email delivered — user=%s template=%s", user_id, template_name
        )

    except Exception as exc:
        logger.error(
            "send_email_notification failed for user=%s template=%s: %s",
            user_id, template_name, exc,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 2: send_bulk_reminder_emails
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_bulk_reminder_emails",
    bind=False,
    acks_late=True,
)
def send_bulk_reminder_emails() -> None:
    """
    Scheduled task (every 6 hours).

    Finds all submissions that have been in READY_FOR_REVIEW status for
    more than 48 hours and sends a reminder email to their assigned coach.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import joinedload
    from app.config import settings
    from app.integrations.sendgrid import sendgrid_service
    from app.models.coach import Coach
    from app.models.submission import Submission, SubmissionStatus
    from app.models.user import User

    db = _sync_session()
    reminder_count = 0

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

        stale_submissions = db.execute(
            select(Submission)
            .where(
                Submission.status == SubmissionStatus.READY_FOR_REVIEW,
                Submission.updated_at < cutoff,
            )
            .options(
                joinedload(Submission.user),
                joinedload(Submission.coach),
            )
        ).unique().scalars().all()

        for submission in stale_submissions:
            coach: Coach | None = submission.coach
            if coach is None:
                continue

            # Load the coach's user record to get their email
            coach_user = db.execute(
                select(User).where(User.id == coach.user_id)
            ).scalar_one_or_none()

            if coach_user is None:
                continue

            sendgrid_service._send_email(
                to_email=coach_user.email,
                template_id=settings.SENDGRID_COACH_APPROVAL_TEMPLATE,
                dynamic_data={
                    "name": coach_user.name,
                    "submission_id": str(submission.id),
                    "user_name": submission.user.name if submission.user else "A golfer",
                    "status_message": (
                        "A swing submission has been waiting for your review for over 48 hours."
                    ),
                    "hours_waiting": int(
                        (datetime.now(timezone.utc) - submission.updated_at.replace(tzinfo=timezone.utc)).total_seconds() // 3600
                    ),
                },
            )
            reminder_count += 1
            logger.info(
                "Reminder sent to coach %s for submission %s",
                coach_user.email,
                submission.id,
            )

        logger.info("send_bulk_reminder_emails: %d reminders sent.", reminder_count)

    except Exception as exc:
        logger.error("send_bulk_reminder_emails failed: %s", exc)
    finally:
        db.close()
