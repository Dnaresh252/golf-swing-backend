import re
import shlex
import subprocess
from typing import Optional

from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.utils.constants import DISCOUNT_CODE_PATTERN, ErrorMessage

# ---------------------------------------------------------------------------
# File validators
# ---------------------------------------------------------------------------

async def validate_image_file(file: UploadFile) -> None:
    """
    Validate that *file* is an allowed image type and within the size limit.
    Raises HTTPException(400) on failure.
    """
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorMessage.INVALID_IMAGE_TYPE,
        )
    content = await file.read()
    await file.seek(0)
    if len(content) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorMessage.FILE_TOO_LARGE_IMAGE,
        )


async def validate_video_file(file: UploadFile) -> None:
    """
    Validate that *file* is an allowed video type and within the size limit.
    Raises HTTPException(400) on failure.
    """
    if file.content_type not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorMessage.INVALID_VIDEO_TYPE,
        )
    content = await file.read()
    await file.seek(0)
    if len(content) > settings.max_video_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorMessage.FILE_TOO_LARGE_VIDEO,
        )


def validate_video_duration(file_path: str) -> None:
    """
    Check that the video at *file_path* does not exceed the configured max duration.
    Uses ffprobe (part of the FFmpeg suite) to read duration metadata.
    Raises HTTPException(400) on failure.
    """
    try:
        ffprobe_path = settings.FFMPEG_PATH.replace("ffmpeg", "ffprobe")
        cmd = [
            ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        duration_str = result.stdout.decode().strip()
        if not duration_str:
            raise ValueError("ffprobe returned no duration.")
        duration = float(duration_str)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine video duration. Please try a different file.",
        )

    if duration > settings.MAX_VIDEO_DURATION_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorMessage.VIDEO_TOO_LONG,
        )


# ---------------------------------------------------------------------------
# Input validators
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_PASSWORD_RE = re.compile(
    r"^(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]).{8,}$"
)
_DISCOUNT_RE = re.compile(DISCOUNT_CODE_PATTERN)


def validate_email_format(email: str) -> bool:
    """Return True if *email* is a syntactically valid email address."""
    return bool(_EMAIL_RE.match(email.strip()))


def validate_password_strength(password: str) -> bool:
    """
    Return True when *password* meets complexity requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one digit
    - At least one special character
    """
    return bool(_PASSWORD_RE.match(password))


def validate_discount_code_format(code: str) -> bool:
    """Return True if *code* matches the GOLF2024-XXXXX pattern."""
    return bool(_DISCOUNT_RE.match(code.strip().upper()))


# ---------------------------------------------------------------------------
# Raising helpers (used by API layer for cleaner route code)
# ---------------------------------------------------------------------------

def assert_valid_email(email: str) -> None:
    if not validate_email_format(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email address format.",
        )


def assert_strong_password(password: str) -> None:
    if not validate_password_strength(password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Password must be at least 8 characters and include "
                "an uppercase letter, a number, and a special character."
            ),
        )


def assert_valid_discount_code(code: str) -> None:
    if not validate_discount_code_format(code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorMessage.DISCOUNT_INVALID_FORMAT,
        )
