import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, model_validator

from app.dependencies import get_current_user
from app.models.user import User
from app.services import admin_settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


# ---------------------------------------------------------------------------
# Dependency: require is_admin == True on the JWT user
# ---------------------------------------------------------------------------

async def _require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class AdminSettingsUpdate(BaseModel):
    SUBMISSION_PRICE_CENTS: Optional[int] = None
    COACH_PAYOUT_CENTS: Optional[int] = None

    @model_validator(mode="after")
    def validate_values(self) -> "AdminSettingsUpdate":
        if self.SUBMISSION_PRICE_CENTS is None and self.COACH_PAYOUT_CENTS is None:
            raise ValueError(
                "Provide at least one of SUBMISSION_PRICE_CENTS or COACH_PAYOUT_CENTS."
            )
        if self.SUBMISSION_PRICE_CENTS is not None and self.SUBMISSION_PRICE_CENTS <= 0:
            raise ValueError("SUBMISSION_PRICE_CENTS must be a positive integer.")
        if self.COACH_PAYOUT_CENTS is not None and self.COACH_PAYOUT_CENTS <= 0:
            raise ValueError("COACH_PAYOUT_CENTS must be a positive integer.")
        return self


# ---------------------------------------------------------------------------
# GET /admin/settings
# ---------------------------------------------------------------------------

@router.get("/settings", summary="Get current effective pricing settings")
async def get_settings(
    admin_user: User = Depends(_require_admin),
):
    logger.info("Admin %s fetched settings.", admin_user.id)
    return {
        "status": "success",
        "message": "Current effective settings.",
        "data": admin_settings.get_effective_settings(),
    }


# ---------------------------------------------------------------------------
# POST /admin/settings
# ---------------------------------------------------------------------------

@router.post("/settings", summary="Update runtime pricing settings (admin only)")
async def update_settings(
    body: AdminSettingsUpdate,
    request: Request,
    admin_user: User = Depends(_require_admin),
):
    rid = _request_id(request)

    if body.SUBMISSION_PRICE_CENTS is not None:
        admin_settings.set_override("SUBMISSION_PRICE_CENTS", body.SUBMISSION_PRICE_CENTS)
        logger.info(
            "Admin %s set SUBMISSION_PRICE_CENTS=%d",
            admin_user.id, body.SUBMISSION_PRICE_CENTS,
        )

    if body.COACH_PAYOUT_CENTS is not None:
        admin_settings.set_override("COACH_PAYOUT_CENTS", body.COACH_PAYOUT_CENTS)
        logger.info(
            "Admin %s set COACH_PAYOUT_CENTS=%d",
            admin_user.id, body.COACH_PAYOUT_CENTS,
        )

    return {
        "status": "success",
        "message": "Settings updated. Note: overrides reset on process restart.",
        "data": admin_settings.get_effective_settings(),
        "request_id": rid,
    }
