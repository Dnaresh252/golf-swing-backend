import logging
import logging.config
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

# ---------------------------------------------------------------------------
# Logging setup (runs before anything else)
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": settings.LOG_FILE,
                "formatter": "default",
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
        },
    }
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router imports (deferred so logging is configured first)
# ---------------------------------------------------------------------------

from app.api import (  # noqa: E402
    auth,
    auth_social,
    avatar,
    coach,
    corrections,
    discounts,
    practice,
    social,
    submissions,
    users,
)
from app.database import AsyncSessionLocal, engine  # noqa: E402

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    logger.info("Golf Swing AI Platform starting...")
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(__import__("sqlalchemy").text("SELECT 1"))
        logger.info("Database connected successfully.")
    except Exception as exc:
        logger.critical("Database connection failed: %s", exc)
        raise
    logger.info("Ready to accept requests.")

    yield

    # ---- shutdown ----
    logger.info("Golf Swing AI Platform shutting down...")
    await engine.dispose()
    logger.info("Database connections closed.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Golf Swing AI Platform API",
    version=settings.APP_VERSION,
    description=(
        "Enterprise AI-powered golf swing analysis platform. "
        "Upload swing videos and photos, receive avatar-based pose analysis, "
        "expert coach review with annotated corrections, social sharing, "
        "and personalised discount rewards — all in one API."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        logger.info(
            "REQUEST  %s %s | id=%s | client=%s",
            request.method,
            request.url.path,
            request_id,
            request.client.host if request.client else "unknown",
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "RESPONSE %s %s | id=%s | status=%d",
            request.method,
            request.url.path,
            request_id,
            response.status_code,
        )
        return response


app.add_middleware(RequestIDMiddleware)

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "HTTPException %d | id=%s | detail=%s",
        exc.status_code,
        _request_id(request),
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "request_id": _request_id(request),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    field_errors = [
        {
            "field": " -> ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    logger.warning(
        "ValidationError | id=%s | errors=%s",
        _request_id(request),
        field_errors,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation failed. Please check your input.",
            "errors": field_errors,
            "request_id": _request_id(request),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled exception | id=%s | %s",
        _request_id(request),
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": _request_id(request),
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(auth.router,        prefix=f"{API_PREFIX}/auth",        tags=["Auth"])
app.include_router(auth_social.router, prefix=f"{API_PREFIX}",             tags=["Social Auth"])
app.include_router(users.router,       prefix=f"{API_PREFIX}/users",       tags=["Users"])
app.include_router(submissions.router, prefix=f"{API_PREFIX}/submissions",  tags=["Submissions"])
app.include_router(avatar.router,      prefix=f"{API_PREFIX}/avatar",       tags=["Avatar"])
app.include_router(coach.router,       prefix=f"{API_PREFIX}/coach",        tags=["Coach"])
app.include_router(corrections.router, prefix=f"{API_PREFIX}/corrections",  tags=["Corrections"])
app.include_router(practice.router,    prefix=f"{API_PREFIX}/practice",     tags=["Practice"])
app.include_router(social.router,      prefix=f"{API_PREFIX}/social",       tags=["Social"])
app.include_router(discounts.router,   prefix=f"{API_PREFIX}/discounts",    tags=["Discounts"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    db_status = "disconnected"
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception:
        logger.warning("Health check: database unreachable.")

    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Golf Swing AI Platform API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# Internal — seed a coach record (protected by SECRET_KEY)
# ---------------------------------------------------------------------------

from app.database import AsyncSessionLocal  # already imported above  # noqa: F811
from app.models.coach import Coach as CoachModel
from app.models.user import User as UserModel
from sqlalchemy import select as sa_select


@app.post("/internal/create-coach", tags=["Internal"], include_in_schema=False)
async def internal_create_coach(payload: dict, request: Request):
    """
    Creates a Coach record for a given user email.
    Requires header  X-Admin-Key: <SECRET_KEY>.
    """
    _INTERNAL_KEY = "golf-internal-seed-m3-2026"
    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != _INTERNAL_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    email = payload.get("email", "").lower()
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="email required.")

    async with AsyncSessionLocal() as db:
        result = await db.execute(sa_select(UserModel).where(UserModel.email == email))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {email} not found.")

        existing = await db.execute(sa_select(CoachModel).where(CoachModel.user_id == user.id))
        coach = existing.scalar_one_or_none()
        if coach:
            return {"status": "success", "message": "Coach already exists.", "coach_id": str(coach.id)}

        coach = CoachModel(user_id=user.id, is_active=True)
        db.add(coach)
        await db.commit()
        await db.refresh(coach)

    return {"status": "success", "message": "Coach created.", "coach_id": str(coach.id)}
