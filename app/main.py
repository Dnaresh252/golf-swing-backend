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
# Internal — seed a coach record (protected by SECRET_KEY)
# ---------------------------------------------------------------------------

from app.database import AsyncSessionLocal  # already imported above  # noqa: F811
from app.models.announcement import Announcement as AnnouncementModel
from app.models.avatar import Avatar as AvatarModel, AvatarStatus as AvatarStatusEnum
from app.models.coach import Coach as CoachModel
from app.models.submission import Submission as SubmissionModel, SubmissionStatus
from app.models.user import User as UserModel
from sqlalchemy import select as sa_select

_INTERNAL_KEY = "golf-internal-seed-m3-2026"


@app.post("/internal/create-coach", tags=["Internal"], include_in_schema=False)
async def internal_create_coach(payload: dict, request: Request):
    """Creates a Coach record for a given user email. Requires X-Admin-Key header."""
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


@app.post("/internal/seed-test-submission", tags=["Internal"], include_in_schema=False)
async def internal_seed_test_submission(payload: dict, request: Request):
    """
    Creates a test user + submission + avatar record with realistic golf skeleton data.
    Requires X-Admin-Key header.
    Returns submission_id for Specialist 4 Unity testing.
    """
    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != _INTERNAL_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    import uuid as _uuid
    from app.utils.security import hash_password

    email = payload.get("email", "specialist4_testuser@golftest.com").lower()
    password = payload.get("password", "Test@1234")

    # 33-joint MediaPipe golf pose (address position — realistic T-pose rotated)
    GOLF_SKELETON = {
        "frames": [
            {
                "frame_num": 0,
                "timestamp": 0.0,
                "joints": [
                    # Head/face landmarks (0-10)
                    {"id": 0,  "name": "nose",              "x": 0.50, "y": 0.12, "z": 0.00, "confidence": 0.99},
                    {"id": 1,  "name": "left_eye_inner",    "x": 0.52, "y": 0.11, "z":-0.01, "confidence": 0.98},
                    {"id": 2,  "name": "left_eye",          "x": 0.54, "y": 0.11, "z":-0.01, "confidence": 0.98},
                    {"id": 3,  "name": "left_eye_outer",    "x": 0.56, "y": 0.11, "z":-0.01, "confidence": 0.97},
                    {"id": 4,  "name": "right_eye_inner",   "x": 0.48, "y": 0.11, "z":-0.01, "confidence": 0.98},
                    {"id": 5,  "name": "right_eye",         "x": 0.46, "y": 0.11, "z":-0.01, "confidence": 0.98},
                    {"id": 6,  "name": "right_eye_outer",   "x": 0.44, "y": 0.11, "z":-0.01, "confidence": 0.97},
                    {"id": 7,  "name": "left_ear",          "x": 0.57, "y": 0.12, "z":-0.03, "confidence": 0.95},
                    {"id": 8,  "name": "right_ear",         "x": 0.43, "y": 0.12, "z":-0.03, "confidence": 0.95},
                    {"id": 9,  "name": "mouth_left",        "x": 0.52, "y": 0.14, "z":-0.01, "confidence": 0.97},
                    {"id": 10, "name": "mouth_right",       "x": 0.48, "y": 0.14, "z":-0.01, "confidence": 0.97},
                    # Shoulders
                    {"id": 11, "name": "left_shoulder",     "x": 0.60, "y": 0.25, "z": 0.00, "confidence": 0.99},
                    {"id": 12, "name": "right_shoulder",    "x": 0.40, "y": 0.25, "z": 0.00, "confidence": 0.99},
                    # Elbows
                    {"id": 13, "name": "left_elbow",        "x": 0.65, "y": 0.38, "z": 0.05, "confidence": 0.98},
                    {"id": 14, "name": "right_elbow",       "x": 0.35, "y": 0.38, "z": 0.05, "confidence": 0.98},
                    # Wrists
                    {"id": 15, "name": "left_wrist",        "x": 0.62, "y": 0.50, "z": 0.10, "confidence": 0.97},
                    {"id": 16, "name": "right_wrist",       "x": 0.38, "y": 0.50, "z": 0.10, "confidence": 0.97},
                    # Hands
                    {"id": 17, "name": "left_pinky",        "x": 0.61, "y": 0.53, "z": 0.11, "confidence": 0.90},
                    {"id": 18, "name": "right_pinky",       "x": 0.39, "y": 0.53, "z": 0.11, "confidence": 0.90},
                    {"id": 19, "name": "left_index",        "x": 0.63, "y": 0.53, "z": 0.12, "confidence": 0.91},
                    {"id": 20, "name": "right_index",       "x": 0.37, "y": 0.53, "z": 0.12, "confidence": 0.91},
                    {"id": 21, "name": "left_thumb",        "x": 0.64, "y": 0.52, "z": 0.11, "confidence": 0.89},
                    {"id": 22, "name": "right_thumb",       "x": 0.36, "y": 0.52, "z": 0.11, "confidence": 0.89},
                    # Hips
                    {"id": 23, "name": "left_hip",          "x": 0.56, "y": 0.55, "z": 0.00, "confidence": 0.99},
                    {"id": 24, "name": "right_hip",         "x": 0.44, "y": 0.55, "z": 0.00, "confidence": 0.99},
                    # Knees
                    {"id": 25, "name": "left_knee",         "x": 0.57, "y": 0.72, "z": 0.02, "confidence": 0.98},
                    {"id": 26, "name": "right_knee",        "x": 0.43, "y": 0.72, "z": 0.02, "confidence": 0.98},
                    # Ankles
                    {"id": 27, "name": "left_ankle",        "x": 0.57, "y": 0.88, "z": 0.01, "confidence": 0.97},
                    {"id": 28, "name": "right_ankle",       "x": 0.43, "y": 0.88, "z": 0.01, "confidence": 0.97},
                    # Feet
                    {"id": 29, "name": "left_heel",         "x": 0.56, "y": 0.90, "z":-0.01, "confidence": 0.95},
                    {"id": 30, "name": "right_heel",        "x": 0.44, "y": 0.90, "z":-0.01, "confidence": 0.95},
                    {"id": 31, "name": "left_foot_index",   "x": 0.58, "y": 0.92, "z": 0.03, "confidence": 0.94},
                    {"id": 32, "name": "right_foot_index",  "x": 0.42, "y": 0.92, "z": 0.03, "confidence": 0.94},
                ]
            }
        ],
        "swing_angles": {
            "left_elbow":    145.2,
            "right_elbow":   148.7,
            "left_knee":     162.4,
            "right_knee":    164.1,
            "left_shoulder":  82.3,
            "right_shoulder": 79.8,
            "hip_rotation":   18.5,
            "spine_tilt":     12.3,
        },
        "analysis_warnings": [],
        "frame_count": 1,
        "source": "seed_data_specialist4",
    }

    PLACEHOLDER_BASE = "https://f005.backblazeb2.com/file/golf-swing-test"

    async with AsyncSessionLocal() as db:
        # Get or create test user
        result = await db.execute(sa_select(UserModel).where(UserModel.email == email))
        user = result.scalar_one_or_none()
        if user is None:
            user = UserModel(
                email=email,
                password_hash=hash_password(password),
                name=payload.get("name", "Specialist 4 Test User"),
            )
            db.add(user)
            await db.flush()
            await db.refresh(user)

        # Create submission in READY_FOR_REVIEW so coach can pull it from queue
        submission = SubmissionModel(
            user_id=user.id,
            status=SubmissionStatus.READY_FOR_REVIEW,
        )
        db.add(submission)
        await db.flush()
        await db.refresh(submission)

        # Create avatar with full sample data
        avatar = AvatarModel(
            submission_id=submission.id,
            skeleton_json=GOLF_SKELETON,
            avatar_obj_url=f"{PLACEHOLDER_BASE}/avatar_{submission.id}.obj",
            avatar_fbx_url=f"{PLACEHOLDER_BASE}/avatar_{submission.id}.fbx",
            avatar_glb_url=f"{PLACEHOLDER_BASE}/avatar_{submission.id}.glb",
            view_top_url=f"{PLACEHOLDER_BASE}/render_{submission.id}_top.png",
            view_front_url=f"{PLACEHOLDER_BASE}/render_{submission.id}_front.png",
            view_left_url=f"{PLACEHOLDER_BASE}/render_{submission.id}_left.png",
            view_right_url=f"{PLACEHOLDER_BASE}/render_{submission.id}_right.png",
            view_back_url=f"{PLACEHOLDER_BASE}/render_{submission.id}_back.png",
            status=AvatarStatusEnum.COMPLETED,
        )
        db.add(avatar)
        await db.commit()

    return {
        "status": "success",
        "message": "Test submission created with skeleton + avatar data.",
        "submission_id": str(submission.id),
        "avatar_id": str(avatar.id),
        "user_email": email,
        "submission_status": submission.status.value,
        "avatar_status": avatar.status.value,
        "glb_url": avatar.avatar_glb_url,
        "fbx_url": avatar.avatar_fbx_url,
        "obj_url": avatar.avatar_obj_url,
        "joint_count": len(GOLF_SKELETON["frames"][0]["joints"]),
    }


@app.post("/internal/upload-avatar-fbx", tags=["Internal"], include_in_schema=False)
async def internal_upload_avatar_fbx(payload: dict, request: Request):
    """
    Upload a base64-encoded FBX file to Backblaze B2 via the Railway server.
    Requires X-Admin-Key header.

    Body:
      b2_path     (str)  — destination path in B2, e.g. assets/avatars/male_skinny/avatar.fbx
      file_base64 (str)  — base64-encoded file bytes
    """
    import base64
    import asyncio

    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != _INTERNAL_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    b2_path      = payload.get("b2_path", "").strip()
    file_b64     = payload.get("file_base64", "").strip()
    content_type = payload.get("content_type", "model/fbx").strip() or "model/fbx"

    if not b2_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="b2_path required.")
    if not file_b64:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file_base64 required.")

    try:
        file_bytes = base64.b64decode(file_b64)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 in file_base64.")

    # b2_service.upload_file is synchronous — run in thread pool to avoid blocking event loop
    from app.integrations.backblaze import b2_service
    try:
        result = await asyncio.to_thread(
            b2_service.upload_file,
            file_bytes,
            b2_path,
            content_type,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"B2 upload failed: {exc}",
        )

    return {
        "status": "success",
        "message": f"File uploaded to B2 at '{b2_path}'.",
        "b2_path": b2_path,
        "file_url": result["file_url"],
        "b2_file_id": result["b2_file_id"],
        "file_size_bytes": result["file_size"],
    }


@app.post("/internal/update-avatar-fbx-urls", tags=["Internal"], include_in_schema=False)
async def internal_update_avatar_fbx_urls(payload: dict, request: Request):
    """
    Update FBX/GLB/OBJ URLs on an existing avatar record.
    Requires X-Admin-Key header.

    Body:
      submission_id  (str)           — UUID of the submission
      avatar_fbx_url (str)           — required
      avatar_glb_url (str, optional)
      avatar_obj_url (str, optional)
    """
    import uuid as _uuid

    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != _INTERNAL_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    submission_id_str = payload.get("submission_id", "").strip()
    fbx_url = payload.get("avatar_fbx_url", "").strip()

    if not submission_id_str:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="submission_id required.")
    if not fbx_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="avatar_fbx_url required.")

    try:
        submission_uuid = _uuid.UUID(submission_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="submission_id is not a valid UUID.")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            sa_select(AvatarModel).where(AvatarModel.submission_id == submission_uuid)
        )
        avatar = result.scalar_one_or_none()
        if avatar is None:
            raise HTTPException(status_code=404, detail=f"No avatar found for submission {submission_id_str}.")

        avatar.avatar_fbx_url = fbx_url
        if payload.get("avatar_glb_url"):
            avatar.avatar_glb_url = payload["avatar_glb_url"].strip()
        if payload.get("avatar_obj_url"):
            avatar.avatar_obj_url = payload["avatar_obj_url"].strip()

        await db.commit()
        await db.refresh(avatar)

    return {
        "status": "success",
        "message": "Avatar URLs updated.",
        "submission_id": submission_id_str,
        "avatar_id": str(avatar.id),
        "avatar_fbx_url": avatar.avatar_fbx_url,
        "avatar_glb_url": avatar.avatar_glb_url,
        "avatar_obj_url": avatar.avatar_obj_url,
        "avatar_status": avatar.status.value,
    }


@app.post("/internal/create-admin", tags=["Internal"], include_in_schema=False)
async def internal_create_admin(payload: dict, request: Request):
    """
    Create the one-time owner admin account.
    Body: { "email": "...", "password": "..." }
    Requires X-Admin-Key header.
    ONLY creates admin if no admin account exists yet.
    """
    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != _INTERNAL_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    from app.utils.security import hash_password as _hash_pw

    email = payload.get("email", "").lower().strip()
    password = payload.get("password", "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required.")

    async with AsyncSessionLocal() as db:
        # Ensure no admin exists yet
        existing_admin = await db.execute(
            sa_select(UserModel).where(UserModel.is_admin == True)  # noqa: E712
        )
        if existing_admin.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="An admin account already exists.")

        existing_email = await db.execute(sa_select(UserModel).where(UserModel.email == email))
        if existing_email.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email already registered.")

        admin_user = UserModel(
            email=email,
            password_hash=_hash_pw(password),
            name="GGW Owner",
            is_admin=True,
            is_active=True,
            is_verified=True,
        )
        db.add(admin_user)
        await db.commit()
        await db.refresh(admin_user)

    logger.info("Owner admin account created for %s", email)
    return {
        "status": "success",
        "message": "Admin account created.",
        "email": email,
        "user_id": str(admin_user.id),
    }
