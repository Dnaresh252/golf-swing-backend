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


async def _build_payload(db: AsyncSession) -> Dict[str, Any]:
    # Registered golfers. Deactivated and suspended accounts are
    # excluded so the public number never counts people who are no
    # longer really on the platform.
    users_count = await db.scalar(
        select(func.count(User.id)).where(
            User.is_active.is_(True),
            User.suspended.is_(False),
        )
    )

    total_submissions = await db.scalar(select(func.count(Submission.id)))

    total_coaches = await db.scalar(
        select(func.count(Coach.id)).where(Coach.is_active.is_(True))
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

    # The date the platform actually opened: the first account created.
    platform_since = await db.scalar(select(func.min(User.created_at)))

    return {
        # `users` is the field the landing page reads. It renders
        # nothing unless this is a finite number greater than zero.
        "users": int(users_count or 0),
        "total_submissions": int(total_submissions or 0),
        "total_coaches": int(total_coaches or 0),
        "average_rating": round(float(average_rating), 2) if average_rating else None,
        "platform_since": platform_since.isoformat() if platform_since else None,
    }


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
