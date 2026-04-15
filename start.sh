#!/bin/sh
# Railway startup script.
# Run alembic with a timeout so a stuck lock never blocks uvicorn.
# Migrations are already applied in production — this is a safety check only.

echo "Running database migrations..."
# timeout 30s: if alembic hangs (lock contention during rolling deploy), skip it
timeout 30 alembic upgrade head || echo "WARNING: alembic timed out or failed — continuing startup"

echo "Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
