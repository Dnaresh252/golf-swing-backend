import logging
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.coach import Coach
from app.models.user import User
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency — resolves the authenticated user from the Bearer token.

    Checks (in order):
      1. Token present
      2. Token not blacklisted in Redis
      3. JWT valid and not expired
      4. User exists in database
      5. User account is active

    Raises HTTP 401 on any failure.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Authentication token is required.",
                "request_id": request_id,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user = await auth_service.get_current_user(db, credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": str(exc),
                "request_id": request_id,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# ---------------------------------------------------------------------------
# get_current_coach
# ---------------------------------------------------------------------------

async def get_current_coach(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> tuple[User, Coach]:
    """
    FastAPI dependency — verifies the current user has an active coach profile.

    Returns a (User, Coach) tuple.
    Raises HTTP 403 if the user is not a coach or their profile is inactive.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    result = await db.execute(
        select(Coach).where(Coach.user_id == current_user.id)
    )
    coach: Optional[Coach] = result.scalar_one_or_none()

    if coach is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "status": "error",
                "message": "Coach account required.",
                "request_id": request_id,
            },
        )
    if not coach.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "status": "error",
                "message": "Coach account is inactive.",
                "request_id": request_id,
            },
        )

    return current_user, coach


# ---------------------------------------------------------------------------
# get_pagination
# ---------------------------------------------------------------------------

async def get_pagination(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> dict:
    """
    FastAPI dependency — returns a validated pagination dict.

    Returns:
        {"page": int, "limit": int, "offset": int}
    """
    return {
        "page": page,
        "limit": limit,
        "offset": (page - 1) * limit,
    }


# ---------------------------------------------------------------------------
# get_request_id
# ---------------------------------------------------------------------------

def get_request_id(request: Request) -> str:
    """
    FastAPI dependency — returns the X-Request-ID for the current request.

    Reads from request.state (set by RequestIDMiddleware) or falls back
    to the X-Request-ID header, then to a newly generated UUID.
    """
    # Prefer the value injected by middleware (already normalised)
    rid = getattr(request.state, "request_id", None)
    if rid:
        return rid

    # Fall back to header
    rid = request.headers.get("X-Request-ID")
    if rid:
        return rid

    # Last resort — generate a new one
    return str(uuid.uuid4())
