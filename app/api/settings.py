"""
Public, unauthenticated settings.

Only the handful of flags the frontend needs before a user has logged in
belong here. Nothing else goes in this file without an explicit request:
it is served to anyone on the internet.
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services import app_settings

logger = logging.getLogger(__name__)
router = APIRouter()

INSTRUCTOR_PICKER_ENABLED = "INSTRUCTOR_PICKER_ENABLED"
INSTRUCTOR_ACCEPT_WINDOW_HOURS = "INSTRUCTOR_ACCEPT_WINDOW_HOURS"
DEFAULT_ACCEPT_WINDOW_HOURS = 48


@router.get(
    "/public",
    summary="Public feature flags. No authentication required.",
)
async def public_settings(db: AsyncSession = Depends(get_db)):
    enabled = await app_settings.get_bool_setting(db, INSTRUCTOR_PICKER_ENABLED, False)
    window = await app_settings.get_int_setting(
        db, INSTRUCTOR_ACCEPT_WINDOW_HOURS, DEFAULT_ACCEPT_WINDOW_HOURS
    )
    return {
        "status": "success",
        "data": {
            "instructor_picker_enabled": enabled,
            "instructor_accept_window_hours": window,
        },
    }
