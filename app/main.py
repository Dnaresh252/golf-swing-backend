import logging
import logging.config
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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
    admin,
    auth,
    auth_social,
    avatar,
    coach,
    corrections,
    discounts,
    payments,
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

# GZip: compress any response >= 1000 bytes for clients that send Accept-Encoding: gzip
# Must be added after CORS so it wraps the outermost layer and compresses final responses.
app.add_middleware(GZipMiddleware, minimum_size=1000)

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
    # When detail is a dict (e.g. video quality gate), spread it directly into the
    # response so structured fields like `rejection_reason` surface at the top level.
    if isinstance(exc.detail, dict):
        content = {"status": "error", "request_id": _request_id(request), **exc.detail}
    else:
        # Keep `detail` too — the frontend reads response.data.detail for
        # user-facing messages (e.g. the suspended-account popup)
        content = {
            "status": "error",
            "message": exc.detail,
            "detail": exc.detail,
            "request_id": _request_id(request),
        }
    return JSONResponse(status_code=exc.status_code, content=content)


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
app.include_router(payments.router,    prefix=f"{API_PREFIX}/payments",     tags=["Payments"])
app.include_router(admin.router,       prefix=f"{API_PREFIX}/admin",        tags=["Admin"])


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
# Public announcement banner — no auth required
# ---------------------------------------------------------------------------

@app.get("/api/v1/announcement", tags=["Announcement"])
async def get_public_announcement():
    async with AsyncSessionLocal() as db:
        result = await db.execute(sa_select(AnnouncementModel).limit(1))
        ann = result.scalar_one_or_none()
    if ann is None:
        return {"status": "success", "data": {"message": "", "active": False}}
    return {"status": "success", "data": {"message": ann.message, "active": ann.active}}


# ---------------------------------------------------------------------------
# NOTE: internal bootstrap endpoints (/internal/create-coach,
# /internal/seed-test-submission, /internal/upload-avatar-fbx,
# /internal/update-avatar-fbx-urls, /internal/create-admin) removed
# 2026-07-25 — security fix. Use the admin panel instead.
# ---------------------------------------------------------------------------

from app.models.announcement import Announcement as AnnouncementModel
from sqlalchemy import select as sa_select
