"""
Celery entry-point.

Run the worker:
    celery -A celery_worker worker --loglevel=info

Run the beat scheduler (periodic tasks):
    celery -A celery_worker beat --loglevel=info

Run both combined (development only):
    celery -A celery_worker worker --beat --loglevel=info
"""

from celery.schedules import crontab

from app.workers.celery_app import celery_app

# ---------------------------------------------------------------------------
# Beat schedule
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    "send-bulk-reminders": {
        "task": "app.workers.notification_tasks.send_bulk_reminder_emails",
        "schedule": crontab(minute=0, hour="*/6"),  # every 6 hours
        "options": {"expires": 3600},
    },
}

celery_app.conf.timezone = "UTC"
celery_app.conf.enable_utc = True

# ---------------------------------------------------------------------------
# Required: expose as "celery" so Celery CLI auto-discovery works
# ---------------------------------------------------------------------------
celery = celery_app
