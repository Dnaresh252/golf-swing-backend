"""
Purge accounts whose 7-day deletion grace period has ended.

Run by the golf-account-purge systemd timer (hourly), not by Celery beat,
which is not running on this server:

    venv/bin/python -m app.workers.account_purge            # purge what is due
    venv/bin/python -m app.workers.account_purge --dry-run  # list only

Each account is purged in its own transaction, so one failure (for example
Backblaze being unreachable) leaves that account pending for the next run
without holding up the rest.
"""
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone

from sqlalchemy import select

import app.models  # noqa: F401  (registers every mapper before first use)
from app.database import AsyncSessionLocal, engine
from app.models.user import User
from app.services import account_deletion

logger = logging.getLogger("account_purge")


async def purge_due_accounts(dry_run: bool = False) -> dict:
    cutoff = datetime.now(timezone.utc) - account_deletion.GRACE_PERIOD
    async with AsyncSessionLocal() as db:
        due = list(
            (
                await db.execute(
                    select(User.id).where(
                        User.deletion_requested_at.is_not(None),
                        User.deletion_requested_at <= cutoff,
                        User.deleted_at.is_(None),
                    )
                )
            ).scalars()
        )

    report = {"due": len(due), "purged": 0, "skipped_blocked": 0, "failed": 0, "dry_run": dry_run}
    if dry_run:
        report["ids"] = [str(i) for i in due]
        return report

    for user_id in due:
        async with AsyncSessionLocal() as db:
            try:
                user = await db.get(User, user_id)
                if user is None or user.deleted_at is not None or user.deletion_requested_at is None:
                    continue
                if await account_deletion.blocking_submission_count(db, user.id):
                    # Cannot normally happen (the account is locked during the
                    # grace period), but never pull files from a live review.
                    logger.warning("Purge skipped, submission with instructor: %s", user.id)
                    report["skipped_blocked"] += 1
                    continue
                await account_deletion.purge_user(db, user, reason="grace_period_ended")
                await db.commit()
                report["purged"] += 1
            except Exception as exc:  # noqa: BLE001 - one account must not stop the rest
                await db.rollback()
                logger.error("Purge failed for %s: %s", user_id, exc)
                report["failed"] += 1
    return report


async def _main(dry_run: bool) -> dict:
    try:
        return await purge_due_accounts(dry_run=dry_run)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from app.utils.error_tracking import init_error_tracking
    init_error_tracking("account_purge")
    result = asyncio.run(_main("--dry-run" in sys.argv))
    print(json.dumps(result))
    sys.exit(1 if result.get("failed") else 0)
