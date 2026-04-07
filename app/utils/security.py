import functools
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import redis.asyncio as aioredis
from jose import JWTError, jwt

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing (bcrypt, cost factor 12)
# passlib 1.7.4 is incompatible with bcrypt >= 4.0; use bcrypt directly.
# ---------------------------------------------------------------------------

_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_access_token(data: Dict[str, Any]) -> str:
    """
    Create a short-lived JWT access token.
    *data* must contain at least a 'sub' (subject) key.
    """
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload.update({"exp": expire, "type": "access"})
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a long-lived JWT refresh token."""
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    payload.update({"exp": expire, "type": "refresh"})
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.
    Returns the payload dict on success, or None if the token is invalid/expired.
    """
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as exc:
        logger.debug("Token verification failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Account lockout (backed by Redis)
# ---------------------------------------------------------------------------

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30

_redis_client: Optional[aioredis.Redis] = None


def _get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
    return _redis_client


def _lockout_key(email: str) -> str:
    return f"lockout:{email}"


def _attempts_key(email: str) -> str:
    return f"login_attempts:{email}"


async def check_account_locked(email: str) -> bool:
    """Return True if the account is currently locked out."""
    r = _get_redis()
    locked = await r.exists(_lockout_key(email))
    return bool(locked)


async def record_failed_attempt(email: str) -> None:
    """Increment the failed-login counter; lock the account when threshold is reached."""
    r = _get_redis()
    key = _attempts_key(email)
    attempts = await r.incr(key)
    # Keep the counter alive for the lockout window
    await r.expire(key, LOCKOUT_DURATION_MINUTES * 60)

    if attempts >= MAX_LOGIN_ATTEMPTS:
        lock_key = _lockout_key(email)
        await r.set(lock_key, 1, ex=LOCKOUT_DURATION_MINUTES * 60)
        logger.warning("Account locked after %d failed attempts: %s", attempts, email)


async def reset_failed_attempts(email: str) -> None:
    """Clear the failed-login counter and any lockout flag after a successful login."""
    r = _get_redis()
    await r.delete(_attempts_key(email), _lockout_key(email))


# ---------------------------------------------------------------------------
# Role / status helpers
# ---------------------------------------------------------------------------

def is_coach(user) -> bool:
    """Return True if the user has an active coach profile."""
    return (
        getattr(user, "coach_profile", None) is not None
        and getattr(user.coach_profile, "is_active", False)
    )


def is_active_user(user) -> bool:
    """Return True if the user account is active."""
    return bool(getattr(user, "is_active", False))


# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------

def require_coach(func):
    """
    Dependency-style decorator.  Wrap an async route handler so that it raises
    403 when the current user is not a coach.

    Usage::

        @router.get("/coach-only")
        @require_coach
        async def endpoint(current_user = Depends(get_current_user)):
            ...
    """
    from fastapi import HTTPException, status

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        user = kwargs.get("current_user")
        if user is None or not is_coach(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Coach account required.",
            )
        return await func(*args, **kwargs)

    return wrapper


def require_active_user(func):
    """Raise 403 when the current user's account is not active."""
    from fastapi import HTTPException, status

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        user = kwargs.get("current_user")
        if user is None or not is_active_user(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive.",
            )
        return await func(*args, **kwargs)

    return wrapper
