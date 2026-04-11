"""
Redis-based rate limiter implemented as FastAPI dependencies.

Uses a fixed-window counter in Redis — no slowapi dependency required.
Compatible with any starlette/FastAPI version.

Usage:
    from app.utils.rate_limit import login_rate_limiter, register_rate_limiter

    @router.post("/login")
    async def login(..., _: None = Depends(login_rate_limiter)):
        ...
"""
import time
import logging

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status

from app.config import settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
    return _redis


def make_rate_limiter(max_requests: int, window_seconds: int):
    """
    Returns a FastAPI dependency that enforces a fixed-window rate limit
    keyed by (endpoint path, client IP).

    Raises HTTP 429 when the caller exceeds *max_requests* within
    *window_seconds* seconds.
    """
    async def _check(request: Request) -> None:
        ip = (request.client.host if request.client else "unknown")
        # Bucket key: resets every window_seconds
        bucket = int(time.time()) // window_seconds
        path   = request.url.path.replace("/", "_")
        key    = f"ratelimit:{path}:{ip}:{bucket}"

        r = _get_redis()
        count = await r.incr(key)
        if count == 1:
            # First hit in this window — set TTL so the key self-expires
            await r.expire(key, window_seconds)

        if count > max_requests:
            logger.warning(
                "Rate limit exceeded | path=%s | ip=%s | count=%d | limit=%d",
                request.url.path, ip, count, max_requests,
            )
            rid = getattr(request.state, "request_id", "unknown")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "status": "error",
                    "message": "Too many requests. Please wait before trying again.",
                    "request_id": rid,
                },
            )

    return _check


# Pre-built limiters — import these in route files
login_rate_limiter    = make_rate_limiter(max_requests=5,  window_seconds=60)
register_rate_limiter = make_rate_limiter(max_requests=10, window_seconds=3600)
