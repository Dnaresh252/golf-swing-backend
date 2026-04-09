web: alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
worker: celery -A celery_worker worker --loglevel=info --concurrency=2
