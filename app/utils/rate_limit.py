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


def make_rate_limiter(max_requests: int, window_seconds: int, fail_closed: bool = False):
    """
    Returns a FastAPI dependency that enforces a fixed-window rate limit
    keyed by (endpoint path, client IP).

    Raises HTTP 429 when the caller exceeds *max_requests* within
    *window_seconds* seconds.
    """
    async def _check(request: Request) -> None:
        # Use X-Forwarded-For to get the real client IP.
        # Railway (and most cloud proxies) append the actual client IP there.
        # request.client.host is the proxy's internal IP (100.64.x.x), not the caller.
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
            request.client.host if request.client else "unknown"
        )
        # Bucket key: resets every window_seconds
        bucket = int(time.time()) // window_seconds
        path   = request.url.path.replace("/", "_")
        key    = f"ratelimit:{path}:{ip}:{bucket}"

        r = _get_redis()
        try:
            count = await r.incr(key)
            if count == 1:
                # First hit in this window — set TTL so the key self-expires
                await r.expire(key, window_seconds)
        except Exception as exc:
            # Redis unreachable — fail open (log and allow through)
            logger.error("Rate limiter Redis error: %s", exc)
            if fail_closed:
                # For auth endpoints, an unreachable Redis must not become an
                # unlimited-attempts window. Refusing logins briefly beats
                # leaving the door open while nobody is counting.
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service temporarily unavailable. Please try again shortly.",
                )
            return

        logger.info(
            "RateLimit | path=%s | ip=%s | count=%d | limit=%d | key=%s",
            request.url.path, ip, count, max_requests, key,
        )

        if count > max_requests:
            logger.warning(
                "Rate limit EXCEEDED | path=%s | ip=%s | count=%d",
                request.url.path, ip, count,
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
# Auth limiters fail closed: no Redis means no counting, and no counting on
# a login endpoint is worse than a short outage.
login_rate_limiter    = make_rate_limiter(max_requests=5,  window_seconds=60, fail_closed=True)
register_rate_limiter = make_rate_limiter(max_requests=10, window_seconds=3600, fail_closed=True)
forgot_password_limiter = make_rate_limiter(max_requests=3, window_seconds=60, fail_closed=True)
refresh_limiter       = make_rate_limiter(max_requests=10, window_seconds=60, fail_closed=True)

# Money and reward endpoints. These fail open: a Redis outage should not stop
# a paying customer checking out.
create_intent_limiter = make_rate_limiter(max_requests=10, window_seconds=60)
verify_post_limiter   = make_rate_limiter(max_requests=5,  window_seconds=60)
discount_limiter      = make_rate_limiter(max_requests=5,  window_seconds=60)
