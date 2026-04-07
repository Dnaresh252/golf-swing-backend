from celery import Celery

from app.config import settings

celery_app = Celery(
    "golf_swing_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.avatar_tasks",
        "app.workers.video_tasks",
        "app.workers.notification_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=int(settings.CELERY_TASK_TIMEOUT * 0.9),
    task_time_limit=settings.CELERY_TASK_TIMEOUT,
    task_max_retries=settings.CELERY_MAX_RETRIES,
    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
)
