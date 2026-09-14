import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.coach import Coach
from app.models.submission import Submission
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

# Public, unauthenticated counts are cheap to serve but not free to
# compute, and the landing page hits this on every cold visit. Cache
# the result in-process for an hour, which is the freshness the
# social proof block actually needs.
_CACHE_TTL_SECONDS = 3600
_cache: Dict[str, Any] = {"expires_at": 0.0, "payload": None}

# Internal test accounts must never be counted as real golfers.
_TEST_EMAIL_PATTERN = "%golftest.com"


async def _build_payload(db: AsyncSession) -> Dict[str, Any]:
    # Registered golfers. Excludes staff, deactivated and suspended
    # accounts, and internal test accounts, so the public number only
    # ever counts real people who are actually on the platform.
    users_count = await db.scalar(
        select(func.count(User.id)).where(
            User.is_active.is_(True),
            User.suspended.is_(False),
            User.is_admin.is_(False),
            ~User.email.ilike(_TEST_EMAIL_PATTERN),
        )
    )

    # Swings from internal test accounts are not real usage.
    total_submissions = await db.scalar(
        select(func.count(Submission.id))
        .join(User, User.id == Submission.user_id)
        .where(~User.email.ilike(_TEST_EMAIL_PATTERN))
    )

    # A deleted instructor is a soft delete: the account (users.is_active) is
    # switched off while the coaches row is kept for payout history, so
    # Coach.is_active alone still counted every removed, suspended and test
    # instructor. Count only instructors whose accounts are genuinely live.
    total_coaches = await db.scalar(
        select(func.count(Coach.id))
        .join(User, User.id == Coach.user_id)
        .where(
            Coach.is_active.is_(True),
            User.is_active.is_(True),
            User.suspended.is_(False),
            ~User.email.ilike(_TEST_EMAIL_PATTERN),
        )
    )

    # Only report a rating once real ratings exist. A default of 0
    # across every coach is not a rating, it is an empty table, and
    # publishing it as 0.0 would be a fabricated number.
    average_rating: Optional[float] = await db.scalar(
        select(func.avg(Coach.rating)).where(
            Coach.is_active.is_(True),
            Coach.rating > 0,
        )
    )

    # The date the platform actually opened: the first real account.
    platform_since = await db.scalar(
        select(func.min(User.created_at)).where(
            User.is_admin.is_(False),
            ~User.email.ilike(_TEST_EMAIL_PATTERN),
        )
    )

    users = int(users_count or 0)
    coaches = int(total_coaches or 0)

    # Served in BOTH shapes on purpose. The deployed landing page reads
    # the flat `users` field; the documented contract is the enveloped
    # `data` object. Returning one without the other silently breaks a
    # live page, so this endpoint answers to both.
    body = {
        "users": users,
        "coaches": coaches,
        "total_submissions": int(total_submissions or 0),
        "total_coaches": coaches,
        "average_rating": round(float(average_rating), 2) if average_rating else None,
        "platform_since": platform_since.isoformat() if platform_since else None,
    }
    return {"status": "success", **body, "data": dict(body)}


@router.get(
    "/public",
    summary="Public platform statistics. No authentication required.",
)
async def public_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    now = time.monotonic()
    if _cache["payload"] is not None and now < _cache["expires_at"]:
        return _cache["payload"]

    payload = await _build_payload(db)
    _cache["payload"] = payload
    _cache["expires_at"] = now + _CACHE_TTL_SECONDS
    return payload
