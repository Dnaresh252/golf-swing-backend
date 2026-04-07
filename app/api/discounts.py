import logging
import random
import string
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.integrations.sendgrid import sendgrid_service
from app.models.discount import DiscountCode
from app.models.submission import Submission, SubmissionStatus
from app.models.user import User
from app.schemas.discount import DiscountCodeResponse, DiscountCodesListResponse
from app.utils.helpers import get_current_utc

logger = logging.getLogger(__name__)
router = APIRouter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


# ---------------------------------------------------------------------------
# GET /user/discount-codes
# ---------------------------------------------------------------------------

@router.get("/discount-codes", summary="List all discount codes for the current user")
async def list_discount_codes(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(DiscountCode)
        .where(DiscountCode.user_id == current_user.id)
        .order_by(DiscountCode.created_at.desc())
    )
    codes = result.scalars().all()
    now = get_current_utc()

    items = []
    active_count = 0
    used_count = 0

    for code in codes:
        is_expired = code.valid_until.replace(tzinfo=timezone.utc) < now
        is_effectively_used = code.used or is_expired

        if code.used:
            used_count += 1
        elif not is_expired:
            active_count += 1

        items.append(DiscountCodeResponse.model_validate(code))

    logger.info("Discount codes retrieved for user: %s", current_user.id)
    return {
        "status": "success",
        "message": "Discount codes retrieved.",
        "data": DiscountCodesListResponse(
            items=items,
            total=len(items),
            active_count=active_count,
            used_count=used_count,
        ).model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /submissions/{id}/generate-discount
# ---------------------------------------------------------------------------

@router.post(
    "/{submission_id}/generate-discount",
    status_code=status.HTTP_201_CREATED,
    summary="Generate a discount code for a completed submission",
)
async def generate_discount(
    submission_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)

    # Verify ownership
    sub_result = await db.execute(
        select(Submission).where(Submission.id == submission_id)
    )
    submission = sub_result.scalar_one_or_none()
    if submission is None or submission.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Submission not found.")

    if submission.status != SubmissionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Submission must be completed to generate a discount.",
                "request_id": rid,
            },
        )

    # Return existing unused code if present
    existing_result = await db.execute(
        select(DiscountCode).where(
            DiscountCode.submission_id == submission_id,
            DiscountCode.used == False,  # noqa: E712
        ).limit(1)
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        logger.info("Returning existing discount code for submission: %s", submission_id)
        return {
            "status": "success",
            "message": "Existing discount code returned.",
            "data": DiscountCodeResponse.model_validate(existing).model_dump(),
        }

    # Generate unique code
    code_str = await _generate_unique_code(db)
    valid_until = get_current_utc() + timedelta(days=settings.DISCOUNT_VALIDITY_DAYS)

    discount = DiscountCode(
        user_id=current_user.id,
        submission_id=submission_id,
        code=code_str,
        discount_percent=settings.DISCOUNT_PERCENTAGE,
        valid_until=valid_until,
    )
    db.add(discount)
    await db.flush()
    await db.refresh(discount)

    # Fire-and-forget email
    import asyncio
    asyncio.create_task(_send_discount_email(current_user, code_str, settings.DISCOUNT_PERCENTAGE, valid_until))

    logger.info("Discount generated for submission: %s", submission_id)
    return {
        "status": "success",
        "message": "Discount code generated.",
        "data": DiscountCodeResponse.model_validate(discount).model_dump(),
    }


async def _generate_unique_code(db: AsyncSession, max_attempts: int = 10) -> str:
    prefix = settings.DISCOUNT_CODE_PREFIX
    for _ in range(max_attempts):
        suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        candidate = f"{prefix}-{suffix}"
        result = await db.execute(
            select(DiscountCode.id).where(DiscountCode.code == candidate).limit(1)
        )
        if result.scalar_one_or_none() is None:
            return candidate
    raise RuntimeError(f"Could not generate unique discount code after {max_attempts} attempts.")


async def _send_discount_email(
    user: User, code: str, percent: int, valid_until: datetime
) -> None:
    sendgrid_service.send_discount_earned(
        user=user,
        discount_code=code,
        percent=percent,
        expiry_date=valid_until.strftime("%B %d, %Y"),
    )
