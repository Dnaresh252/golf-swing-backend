import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.auth import (
    AuthResponse,
    CoachLoginResponse,
    CoachProfileResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    CoachLogin,
)
from app.services.auth_service import auth_service
from app.utils.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error(message: str, request_id: str, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"status": "error", "message": message, "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
@limiter.limit("10/hour")
async def register(
    payload: UserRegister,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    rid = _request_id(request)
    try:
        user = await auth_service.register_user(db, payload)
    except ValueError as exc:
        raise _error(str(exc), rid, status.HTTP_409_CONFLICT)

    tokens = auth_service.create_tokens(user.id)

    # Fire-and-forget welcome email — do not block or fail the request
    import asyncio
    asyncio.create_task(auth_service.send_welcome_email(user))

    return {
        "status": "success",
        "message": "Registration successful. Please verify your email.",
        "data": AuthResponse(
            tokens=TokenResponse(**tokens),
            user=UserResponse.model_validate(user),
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------

@router.post("/login", summary="Authenticate a user")
@limiter.limit("5/minute")
async def login(
    payload: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    rid = _request_id(request)
    try:
        user = await auth_service.authenticate_user(db, payload.email, payload.password)
    except PermissionError as exc:
        raise _error(str(exc), rid, status.HTTP_423_LOCKED)
    except ValueError as exc:
        raise _error(str(exc), rid, status.HTTP_401_UNAUTHORIZED)

    tokens = auth_service.create_tokens(user.id)
    logger.info("User logged in: %s", user.email)

    return {
        "status": "success",
        "message": "Logged in successfully.",
        "data": AuthResponse(
            tokens=TokenResponse(**tokens),
            user=UserResponse.model_validate(user),
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /coach-login
# ---------------------------------------------------------------------------

@router.post("/coach-login", summary="Authenticate a coach")
async def coach_login(
    payload: CoachLogin,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    rid = _request_id(request)
    try:
        user, coach = await auth_service.authenticate_coach(
            db, payload.email, payload.password
        )
    except PermissionError as exc:
        raise _error(str(exc), rid, status.HTTP_403_FORBIDDEN)
    except ValueError as exc:
        raise _error(str(exc), rid, status.HTTP_401_UNAUTHORIZED)

    tokens = auth_service.create_tokens(user.id)
    logger.info("Coach logged in: %s", user.email)

    return {
        "status": "success",
        "message": "Coach logged in successfully.",
        "data": CoachLoginResponse(
            tokens=TokenResponse(**tokens),
            user=UserResponse.model_validate(user),
            coach=CoachProfileResponse.model_validate(coach),
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /refresh
# ---------------------------------------------------------------------------

@router.post("/refresh", summary="Refresh access token")
async def refresh_token(
    payload: RefreshTokenRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    rid = _request_id(request)
    try:
        new_access_token = await auth_service.refresh_access_token(payload.refresh_token)
    except ValueError as exc:
        raise _error(str(exc), rid, status.HTTP_401_UNAUTHORIZED)

    return {
        "status": "success",
        "message": "Token refreshed.",
        "data": {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": __import__("app.config", fromlist=["settings"]).settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        },
    }


# ---------------------------------------------------------------------------
# POST /logout
# ---------------------------------------------------------------------------

@router.post("/logout", summary="Invalidate current session")
async def logout(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    rid = _request_id(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise _error("Authentication token is required.", rid, status.HTTP_401_UNAUTHORIZED)

    token = auth_header.split(" ", 1)[1]

    try:
        current_user = await auth_service.get_current_user(db, token)
    except ValueError as exc:
        raise _error(str(exc), rid, status.HTTP_401_UNAUTHORIZED)

    await auth_service.logout_user(token)
    logger.info("User logged out: %s", current_user.id)

    return {
        "status": "success",
        "message": "Logged out successfully.",
        "data": None,
    }
