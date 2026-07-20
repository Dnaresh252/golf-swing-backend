import logging
import uuid
from datetime import timezone
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.discount import DiscountCode
from app.models.free_code import FreeCode
from app.models.payment import Payment, PaymentStatus
from app.models.social import SocialSharing
from app.models.submission import Submission, SubmissionStatus
from app.models.user import User
from app.services import admin_settings
from app.utils.helpers import get_current_utc

logger = logging.getLogger(__name__)
router = APIRouter()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _configure_stripe() -> None:
    """Set stripe.api_key; raises 503 if STRIPE_SECRET_KEY not set."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing is not configured on this server.",
        )
    stripe.api_key = settings.STRIPE_SECRET_KEY


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateIntentRequest(BaseModel):
    submission_id: Optional[uuid.UUID] = None
    discount_code: Optional[str] = None
    free_code: Optional[str] = None
    skip_free_option: Optional[bool] = False


class PaymentRecord(BaseModel):
    id: str
    submission_id: Optional[str]
    amount_cents: int
    currency: str
    status: str
    stripe_payment_intent_id: Optional[str]
    is_free: bool
    created_at: str

    @classmethod
    def from_payment(cls, p: Payment) -> "PaymentRecord":
        meta = p.payment_metadata or {}
        return cls(
            id=str(p.id),
            submission_id=str(p.submission_id) if p.submission_id else None,
            amount_cents=p.amount_cents,
            currency=p.currency,
            status=p.status.value,
            stripe_payment_intent_id=p.stripe_payment_intent_id,
            is_free=meta.get("is_free", False),
            created_at=p.created_at.isoformat(),
        )


# ---------------------------------------------------------------------------
# POST /payments/create-intent
# ---------------------------------------------------------------------------

@router.post("/create-intent", summary="Create a Stripe PaymentIntent for a submission")
async def create_payment_intent(
    body: CreateIntentRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rid = _request_id(request)

    # ── 1. Base price from runtime admin settings ──────────────────────────
    amount_cents: int = admin_settings.get_submission_price_cents()
    free_reason: Optional[str] = None

    # ── 2. First-submission-free (launch promotion) ─────────────────────────
    completed_sub_count_row = await db.execute(
        select(func.count()).select_from(Submission).where(
            Submission.user_id == current_user.id,
            Submission.status == SubmissionStatus.COMPLETED,
        )
    )
    completed_sub_count: int = completed_sub_count_row.scalar() or 0

    if completed_sub_count == 0 and not body.skip_free_option:
        amount_cents = 0
        free_reason = "first_submission_free"

    # ── 3. Free code redemption ─────────────────────────────────────────────
    free_code_record: Optional[FreeCode] = None
    if body.free_code and amount_cents > 0:
        code_upper = body.free_code.upper().strip()
        fc_result = await db.execute(
            select(FreeCode).where(
                FreeCode.code == code_upper,
                FreeCode.active == True,  # noqa: E712
            )
        )
        free_code_record = fc_result.scalar_one_or_none()
        if free_code_record is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"status": "error", "message": "Free code is invalid or inactive.", "request_id": rid},
            )
        now_utc = get_current_utc()
        if free_code_record.expires_at and free_code_record.expires_at.replace(tzinfo=timezone.utc if free_code_record.expires_at.tzinfo is None else free_code_record.expires_at.tzinfo) < now_utc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"status": "error", "message": "Free code has expired.", "request_id": rid},
            )
        if free_code_record.uses >= free_code_record.max_uses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"status": "error", "message": "Free code has reached its usage limit.", "request_id": rid},
            )
        amount_cents = 0
        free_reason = f"free_code:{code_upper}"

    # ── 4. Discount code ────────────────────────────────────────────────────
    discount_record: Optional[DiscountCode] = None
    if body.discount_code and amount_cents > 0:
        code_upper = body.discount_code.upper().strip()
        dc_result = await db.execute(
            select(DiscountCode).where(
                DiscountCode.code == code_upper,
                DiscountCode.used == False,  # noqa: E712
            )
        )
        discount_record = dc_result.scalar_one_or_none()

        if discount_record is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": "Discount code is invalid or has already been used.",
                    "request_id": rid,
                },
            )

        now = get_current_utc()
        expires = discount_record.valid_until
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": "Discount code has expired.",
                    "request_id": rid,
                },
            )

        multiplier = 1.0 - (discount_record.discount_percent / 100.0)
        amount_cents = max(0, round(amount_cents * multiplier))
        if amount_cents == 0:
            free_reason = "discount_100pct"

    # ── 6. Shared metadata ──────────────────────────────────────────────────
    meta = {
        "user_id": str(current_user.id),
        "submission_id": str(body.submission_id) if body.submission_id else None,
        "discount_code": body.discount_code.upper().strip() if body.discount_code else None,
        "discount_percent": discount_record.discount_percent if discount_record else 0,
        "free_code": body.free_code.upper().strip() if body.free_code else None,
        "is_free": amount_cents == 0,
        "free_reason": free_reason,
    }

    # ── 7. Free path — no Stripe required ──────────────────────────────────
    if amount_cents == 0:
        payment = Payment(
            user_id=current_user.id,
            submission_id=body.submission_id,
            amount_cents=0,
            currency=settings.STRIPE_CURRENCY,
            status=PaymentStatus.COMPLETED,
            payment_metadata=meta,
        )
        db.add(payment)

        if discount_record is not None:
            discount_record.used = True
            discount_record.used_at = get_current_utc()

        if free_code_record is not None:
            free_code_record.uses += 1

        await db.commit()
        await db.refresh(payment)

        logger.info(
            "Free submission granted for user %s reason=%s", current_user.id, free_reason
        )
        return {
            "status": "success",
            "message": "Submission is free — no payment required.",
            "data": {
                "is_free": True,
                "client_secret": None,
                "payment_intent_id": None,
                "payment_id": str(payment.id),
                "amount_cents": 0,
                "currency": settings.STRIPE_CURRENCY,
                "free_reason": free_reason,
            },
        }

    # ── 8. Paid path — create Stripe PaymentIntent ──────────────────────────
    _configure_stripe()
    try:
        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency=settings.STRIPE_CURRENCY,
            metadata={
                "user_id": str(current_user.id),
                "submission_id": str(body.submission_id) if body.submission_id else "",
                "discount_code": body.discount_code or "",
                "platform": "golf_swing_ai",
            },
        )
    except stripe.StripeError as exc:
        logger.error(
            "Stripe PaymentIntent creation failed for user %s: %s", current_user.id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "status": "error",
                "message": f"Payment processor error: {getattr(exc, 'user_message', None) or str(exc)}",
                "request_id": rid,
            },
        )

    # ── 7. Persist PENDING payment ──────────────────────────────────────────
    payment = Payment(
        user_id=current_user.id,
        submission_id=body.submission_id,
        stripe_payment_intent_id=intent["id"],
        amount_cents=amount_cents,
        currency=settings.STRIPE_CURRENCY,
        status=PaymentStatus.PENDING,
        payment_metadata=meta,
    )
    db.add(payment)
    await db.commit()
    await db.refresh(payment)

    logger.info(
        "PaymentIntent %s created for user %s amount=%d cents",
        intent["id"], current_user.id, amount_cents,
    )
    return {
        "status": "success",
        "message": "Payment intent created.",
        "data": {
            "is_free": False,
            "client_secret": intent["client_secret"],
            "payment_intent_id": intent["id"],
            "payment_id": str(payment.id),
            "amount_cents": amount_cents,
            "currency": settings.STRIPE_CURRENCY,
            "free_reason": None,
        },
    }


# ---------------------------------------------------------------------------
# POST /payments/webhook
# ---------------------------------------------------------------------------

@router.post("/webhook", summary="Stripe webhook receiver (no JWT auth)")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Receives signed Stripe events. Verified via Stripe-Signature header.
    Always returns 200 after verification to prevent Stripe retry storms.
    """
    if not settings.STRIPE_SECRET_KEY:
        return {"status": "ignored", "reason": "stripe_not_configured"}

    stripe.api_key = settings.STRIPE_SECRET_KEY
    payload: bytes = await request.body()
    sig_header: str = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except stripe.SignatureVerificationError:
        logger.warning("Stripe webhook: invalid signature.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Stripe webhook signature.",
        )
    except Exception as exc:
        logger.error("Stripe webhook parse error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook payload error.",
        )

    event_type: str = event["type"]
    event_obj: dict = event["data"]["object"]
    intent_id: str = event_obj.get("id", "")

    logger.info("Stripe webhook: type=%s intent=%s", event_type, intent_id)

    if event_type == "payment_intent.succeeded":
        await _on_payment_succeeded(db, intent_id, event_obj)
    elif event_type == "payment_intent.payment_failed":
        await _on_payment_failed(db, intent_id, event_obj)
    else:
        logger.debug("Stripe event '%s' not handled — ignored.", event_type)

    return {"status": "ok", "event_type": event_type}


async def _on_payment_succeeded(
    db: AsyncSession, intent_id: str, event_obj: dict
) -> None:
    result = await db.execute(
        select(Payment).where(Payment.stripe_payment_intent_id == intent_id)
    )
    payment = result.scalar_one_or_none()
    if payment is None:
        logger.warning("payment_intent.succeeded: no Payment row for intent %s", intent_id)
        return
    if payment.status == PaymentStatus.COMPLETED:
        logger.info("Payment %s already COMPLETED — idempotent skip.", payment.id)
        return

    # Snapshot discount code before mutating metadata
    discount_code_str: Optional[str] = (payment.payment_metadata or {}).get("discount_code")

    payment.status = PaymentStatus.COMPLETED
    payment.payment_metadata = {
        **(payment.payment_metadata or {}),
        "stripe_event": "payment_intent.succeeded",
    }

    if discount_code_str:
        dc_result = await db.execute(
            select(DiscountCode).where(
                DiscountCode.code == discount_code_str,
                DiscountCode.used == False,  # noqa: E712
            )
        )
        dc = dc_result.scalar_one_or_none()
        if dc:
            dc.used = True
            dc.used_at = get_current_utc()

    await db.commit()
    logger.info("Payment %s → COMPLETED (intent %s).", payment.id, intent_id)


async def _on_payment_failed(
    db: AsyncSession, intent_id: str, event_obj: dict
) -> None:
    result = await db.execute(
        select(Payment).where(Payment.stripe_payment_intent_id == intent_id)
    )
    payment = result.scalar_one_or_none()
    if payment is None:
        logger.warning("payment_intent.payment_failed: no Payment row for intent %s", intent_id)
        return

    failure_msg: str = (event_obj.get("last_payment_error") or {}).get("message", "")
    payment.status = PaymentStatus.FAILED
    payment.payment_metadata = {
        **(payment.payment_metadata or {}),
        "stripe_event": "payment_intent.payment_failed",
        "failure_message": failure_msg,
    }
    await db.commit()
    logger.warning("Payment %s → FAILED (intent %s): %s", payment.id, intent_id, failure_msg)


# ---------------------------------------------------------------------------
# GET /payments/history
# ---------------------------------------------------------------------------

@router.get("/history", summary="Get current user's payment history")
async def payment_history(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Payment)
        .where(Payment.user_id == current_user.id)
        .order_by(Payment.created_at.desc())
        .limit(100)
    )
    payments = result.scalars().all()
    items = [PaymentRecord.from_payment(p).model_dump() for p in payments]

    logger.info("Payment history: user=%s count=%d", current_user.id, len(items))
    return {
        "status": "success",
        "message": "Payment history retrieved.",
        "data": {"items": items, "total": len(items)},
    }


# ---------------------------------------------------------------------------
# GET /payments/config
# ---------------------------------------------------------------------------

@router.get("/config", summary="Return Stripe publishable key and current pricing")
async def payment_config(
    current_user: User = Depends(get_current_user),
):
    price_cents = admin_settings.get_submission_price_cents()
    return {
        "status": "success",
        "message": "Payment configuration retrieved.",
        "data": {
            "publishable_key": settings.STRIPE_PUBLISHABLE_KEY,
            "submission_price": round(price_cents / 100, 2),
            "submission_price_cents": price_cents,
            "first_submission_free": True,
            "currency": settings.STRIPE_CURRENCY,
            "discount_percentage": settings.DISCOUNT_PERCENTAGE,
        },
    }
