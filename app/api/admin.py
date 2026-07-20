import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, field_validator
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.announcement import Announcement
from app.models.audit_log import AuditLog
from app.models.coach import Coach
from app.models.free_code import FreeCode
from app.models.payment import Payment, PaymentStatus
from app.models.submission import Submission, SubmissionStatus
from app.models.user import User
from app.services import admin_settings
from app.utils.helpers import get_current_utc
from app.utils.security import hash_password

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: require is_admin == True
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
# Audit log helper
# ---------------------------------------------------------------------------

async def _audit(db: AsyncSession, action: str, detail: str = "") -> None:
    entry = AuditLog(action=action, detail=detail)
    db.add(entry)
    # committed by the caller alongside the main change


# ---------------------------------------------------------------------------
# Task 2 — GET /admin/stats
# ---------------------------------------------------------------------------

@router.get("/stats", summary="Dashboard statistics")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    now = get_current_utc()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    async def _sum(since: datetime) -> float:
        row = await db.execute(
            select(func.coalesce(func.sum(Payment.amount_cents), 0)).where(
                Payment.status == PaymentStatus.COMPLETED,
                Payment.created_at >= since,
            )
        )
        return round((row.scalar() or 0) / 100, 2)

    async def _count_subs(since: Optional[datetime] = None) -> int:
        q = select(func.count()).select_from(Submission)
        if since:
            q = q.where(Submission.created_at >= since)
        row = await db.execute(q)
        return row.scalar() or 0

    revenue_today = await _sum(today_start)
    revenue_week = await _sum(week_start)
    revenue_month = await _sum(month_start)

    subs_today = await _count_subs(today_start)
    subs_month = await _count_subs(month_start)

    total_users_row = await db.execute(
        select(func.count()).select_from(User).where(
            User.is_admin == False,  # noqa: E712
        )
    )
    total_users = total_users_row.scalar() or 0

    active_coaches_row = await db.execute(
        select(func.count()).select_from(Coach).where(Coach.is_active == True)  # noqa: E712
    )
    active_coaches = active_coaches_row.scalar() or 0

    pending_row = await db.execute(
        select(func.count()).select_from(Submission).where(
            Submission.status.in_([SubmissionStatus.READY_FOR_REVIEW, SubmissionStatus.IN_REVIEW])
        )
    )
    pending_in_queue = pending_row.scalar() or 0

    stuck_cutoff = now - timedelta(minutes=30)
    stuck_row = await db.execute(
        select(func.count()).select_from(Submission).where(
            Submission.status == SubmissionStatus.ANALYZING,
            Submission.updated_at <= stuck_cutoff,
        )
    )
    stuck_processing = stuck_row.scalar() or 0

    return {
        "status": "success",
        "data": {
            "revenue_today": revenue_today,
            "revenue_week": revenue_week,
            "revenue_month": revenue_month,
            "submissions_today": subs_today,
            "submissions_month": subs_month,
            "total_users": total_users,
            "active_coaches": active_coaches,
            "pending_in_queue": pending_in_queue,
            "stuck_processing": stuck_processing,
        },
    }


# ---------------------------------------------------------------------------
# Task 3 — User Management
# ---------------------------------------------------------------------------

@router.get("/users", summary="List users with optional search and pagination")
async def list_users(
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    per_page = 50
    offset = (page - 1) * per_page

    q = (
        select(User, func.count(Submission.id).label("submissions_count"))
        .outerjoin(Submission, Submission.user_id == User.id)
        .where(User.is_admin == False)  # noqa: E712
        .group_by(User.id)
        .order_by(User.created_at.desc())
    )
    if search:
        term = f"%{search}%"
        q = q.where(
            or_(User.name.ilike(term), User.email.ilike(term))
        )
    q = q.offset(offset).limit(per_page)

    rows = (await db.execute(q)).all()
    items = []
    for user, sub_count in rows:
        items.append({
            "id": str(user.id),
            "full_name": user.name,
            "email": user.email,
            "suspended": user.suspended,
            "submissions_count": sub_count,
            "last_active": user.last_login_at.isoformat() if user.last_login_at else user.updated_at.isoformat(),
        })

    return {"status": "success", "data": {"items": items, "page": page}}


class SuspendRequest(BaseModel):
    suspended: bool


@router.put("/users/{user_id}/suspend", summary="Suspend or unsuspend a student account")
async def suspend_user(
    user_id: uuid.UUID,
    body: SuspendRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    result = await db.execute(
        select(User).where(User.id == user_id).options(selectinload(User.coach_profile))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.is_admin or user.coach_profile is not None:
        raise HTTPException(status_code=400, detail="Cannot suspend admin or coach accounts.")

    user.suspended = body.suspended
    action = "user_suspended" if body.suspended else "user_unsuspended"
    await _audit(db, action, f"user_id={user_id} email={user.email}")
    await db.commit()
    logger.info("Admin %s %s user %s", admin_user.id, action, user_id)
    return {"status": "success", "message": f"User {'suspended' if body.suspended else 'unsuspended'}."}


@router.delete("/users/{user_id}", summary="Permanently delete a student account")
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    result = await db.execute(
        select(User).where(User.id == user_id).options(selectinload(User.coach_profile))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.is_admin or user.coach_profile is not None:
        raise HTTPException(status_code=400, detail="Cannot delete admin or coach accounts.")

    email = user.email
    await _audit(db, "user_deleted", f"user_id={user_id} email={email}")
    await db.delete(user)
    await db.commit()
    logger.info("Admin %s deleted user %s (%s)", admin_user.id, user_id, email)
    return {"status": "success", "message": "User deleted."}


@router.get("/users/ghosts", summary="List ghost accounts (unverified, no submissions, inactive)")
async def list_ghosts(
    inactive_months: int = Query(6, ge=1),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    cutoff = get_current_utc() - timedelta(days=inactive_months * 30)

    q = (
        select(User, func.count(Submission.id).label("submissions_count"))
        .outerjoin(Submission, Submission.user_id == User.id)
        .outerjoin(Coach, Coach.user_id == User.id)
        .where(
            User.is_admin == False,  # noqa: E712
            Coach.id == None,  # noqa: E711
            or_(
                User.is_verified == False,  # noqa: E712
                User.last_login_at <= cutoff,
                User.last_login_at == None,  # noqa: E711
            ),
        )
        .group_by(User.id)
        .having(func.count(Submission.id) == 0)
    )

    rows = (await db.execute(q)).all()
    items = []
    for user, sub_count in rows:
        items.append({
            "id": str(user.id),
            "full_name": user.name,
            "email": user.email,
            "suspended": user.suspended,
            "submissions_count": sub_count,
            "last_active": user.last_login_at.isoformat() if user.last_login_at else None,
        })

    return {"status": "success", "data": {"items": items, "total": len(items)}}


class GhostDeleteRequest(BaseModel):
    user_ids: List[uuid.UUID]


@router.post("/users/ghosts/delete", summary="Bulk delete confirmed ghost accounts")
async def delete_ghosts(
    body: GhostDeleteRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    deleted = []
    for uid in body.user_ids:
        result = await db.execute(
            select(User).where(User.id == uid).options(selectinload(User.coach_profile))
        )
        user = result.scalar_one_or_none()
        if user is None or user.is_admin or user.coach_profile is not None:
            continue
        # Verify still a ghost: no submissions + (unverified OR inactive)
        sub_count_row = await db.execute(
            select(func.count()).select_from(Submission).where(Submission.user_id == uid)
        )
        if (sub_count_row.scalar() or 0) > 0:
            continue
        deleted.append(str(uid))
        await db.delete(user)

    await _audit(db, "ghosts_purged", f"count={len(deleted)} ids={deleted}")
    await db.commit()
    logger.info("Admin %s purged %d ghost accounts", admin_user.id, len(deleted))
    return {"status": "success", "data": {"deleted_count": len(deleted), "deleted_ids": deleted}}


# ---------------------------------------------------------------------------
# Task 4 — Coach Management
# ---------------------------------------------------------------------------

class CreateCoachRequest(BaseModel):
    full_name: str
    email: str
    password: str

    @field_validator("full_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("full_name must be at least 2 characters.")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        import re
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must have an uppercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must have a digit.")
        return v


@router.post("/coaches", summary="Create a coach account", status_code=status.HTTP_201_CREATED)
async def create_coach(
    body: CreateCoachRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    email = body.email.lower().strip()
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = User(
        email=email,
        password_hash=hash_password(body.password),
        name=body.full_name.strip(),
        is_admin=False,
        is_active=True,
    )
    db.add(user)
    await db.flush()

    coach = Coach(user_id=user.id, is_active=True)
    db.add(coach)

    await _audit(db, "coach_created", f"email={email} name={body.full_name}")
    await db.commit()
    await db.refresh(user)
    await db.refresh(coach)

    logger.info("Admin %s created coach %s (%s)", admin_user.id, user.id, email)
    return {
        "status": "success",
        "message": "Coach account created.",
        "data": {
            "coach_id": str(coach.id),
            "user_id": str(user.id),
            "email": email,
            "full_name": user.name,
        },
    }


@router.get("/coaches", summary="List all coach accounts")
async def list_coaches(
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    rows = (await db.execute(
        select(Coach, User)
        .join(User, User.id == Coach.user_id)
        .order_by(Coach.created_at.desc())
    )).all()

    items = [
        {
            "id": str(coach.id),
            "full_name": user.name,
            "email": user.email,
            "active": coach.is_active,
        }
        for coach, user in rows
    ]
    return {"status": "success", "data": {"items": items}}


class SetCoachActiveRequest(BaseModel):
    active: bool


@router.put("/coaches/{coach_id}/active", summary="Enable or disable a coach")
async def set_coach_active(
    coach_id: uuid.UUID,
    body: SetCoachActiveRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    result = await db.execute(
        select(Coach).where(Coach.id == coach_id).options(selectinload(Coach.user))
    )
    coach = result.scalar_one_or_none()
    if coach is None:
        raise HTTPException(status_code=404, detail="Coach not found.")

    coach.is_active = body.active
    action = "coach_enabled" if body.active else "coach_disabled"
    await _audit(db, action, f"coach_id={coach_id}")
    await db.commit()
    logger.info("Admin %s %s coach %s", admin_user.id, action, coach_id)
    return {"status": "success", "message": f"Coach {'enabled' if body.active else 'disabled'}."}


# ---------------------------------------------------------------------------
# Task 5 — Billing and Refunds
# ---------------------------------------------------------------------------

class BillingUpdate(BaseModel):
    submission_price: float

    @field_validator("submission_price")
    @classmethod
    def price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("submission_price must be positive.")
        return v


@router.get("/billing", summary="Get current submission price")
async def get_billing(admin_user: User = Depends(_require_admin)):
    price_cents = admin_settings.get_submission_price_cents()
    return {
        "status": "success",
        "data": {"submission_price": round(price_cents / 100, 2)},
    }


@router.put("/billing", summary="Update submission price")
async def update_billing(
    body: BillingUpdate,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    new_cents = round(body.submission_price * 100)
    admin_settings.set_override("SUBMISSION_PRICE_CENTS", new_cents)
    await _audit(db, "price_changed", f"new_price={body.submission_price}")
    await db.commit()
    logger.info("Admin %s updated price to %.2f", admin_user.id, body.submission_price)
    return {
        "status": "success",
        "message": "Price updated.",
        "data": {"submission_price": body.submission_price},
    }


@router.get("/payments", summary="List all payments with optional search")
async def list_payments(
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    per_page = 50
    offset = (page - 1) * per_page

    q = (
        select(Payment, User.email.label("user_email"))
        .join(User, User.id == Payment.user_id)
        .order_by(Payment.created_at.desc())
    )
    if search:
        term = f"%{search}%"
        q = q.where(User.email.ilike(term))
    q = q.offset(offset).limit(per_page)

    rows = (await db.execute(q)).all()
    items = []
    for payment, user_email in rows:
        meta = payment.payment_metadata or {}
        items.append({
            "id": str(payment.id),
            "user_email": user_email,
            "amount": round(payment.amount_cents / 100, 2),
            "date": payment.created_at.isoformat(),
            "description": meta.get("free_reason") or (
                f"Submission {meta.get('submission_id', '')}" if meta.get("submission_id") else "Payment"
            ),
            "refunded": payment.status == PaymentStatus.REFUNDED,
        })

    return {"status": "success", "data": {"items": items, "page": page}}


class RefundRequest(BaseModel):
    reason: str = ""


@router.post("/payments/{payment_id}/refund", summary="Refund a payment via Stripe")
async def refund_payment(
    payment_id: uuid.UUID,
    body: RefundRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    result = await db.execute(select(Payment).where(Payment.id == payment_id))
    payment = result.scalar_one_or_none()
    if payment is None:
        raise HTTPException(status_code=404, detail="Payment not found.")
    if payment.status == PaymentStatus.REFUNDED:
        raise HTTPException(status_code=400, detail="Payment already refunded.")
    if payment.status != PaymentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Only completed payments can be refunded.")

    meta = payment.payment_metadata or {}
    if meta.get("platform") in ("app_store", "google_play"):
        raise HTTPException(
            status_code=400,
            detail="App store payments are refunded in App Store Connect or Google Play Console, not here.",
        )

    if not payment.stripe_payment_intent_id:
        raise HTTPException(status_code=400, detail="No Stripe payment intent on this record.")

    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured.")

    stripe.api_key = settings.STRIPE_SECRET_KEY
    try:
        stripe.Refund.create(
            payment_intent=payment.stripe_payment_intent_id,
            reason="requested_by_customer" if not body.reason else None,
            metadata={"admin_reason": body.reason},
        )
    except stripe.StripeError as exc:
        raise HTTPException(status_code=502, detail=f"Stripe refund failed: {exc}")

    payment.status = PaymentStatus.REFUNDED
    payment.payment_metadata = {**meta, "refund_reason": body.reason, "refunded_by_admin": str(admin_user.id)}
    await _audit(db, "refund_issued", f"payment_id={payment_id} reason={body.reason}")
    await db.commit()
    logger.info("Admin %s refunded payment %s", admin_user.id, payment_id)
    return {"status": "success", "message": "Payment refunded."}


# ---------------------------------------------------------------------------
# Task 6 — Free Submission Codes
# ---------------------------------------------------------------------------

@router.get("/free-codes", summary="List free submission codes")
async def list_free_codes(
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    rows = (await db.execute(
        select(FreeCode).order_by(FreeCode.created_at.desc())
    )).scalars().all()

    items = [
        {
            "id": str(fc.id),
            "code": fc.code,
            "max_uses": fc.max_uses,
            "uses": fc.uses,
            "expires_at": fc.expires_at.isoformat() if fc.expires_at else None,
            "active": fc.active,
        }
        for fc in rows
    ]
    return {"status": "success", "data": {"items": items}}


class CreateFreeCodeRequest(BaseModel):
    code: str
    max_uses: int = 1
    expires_at: Optional[str] = None

    @field_validator("code")
    @classmethod
    def code_not_empty(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) < 3:
            raise ValueError("code must be at least 3 characters.")
        return v

    @field_validator("max_uses")
    @classmethod
    def max_uses_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_uses must be at least 1.")
        return v


@router.post("/free-codes", summary="Create a free submission code", status_code=status.HTTP_201_CREATED)
async def create_free_code(
    body: CreateFreeCodeRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    existing = await db.execute(select(FreeCode).where(FreeCode.code == body.code))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Code already exists.")

    expires: Optional[datetime] = None
    if body.expires_at:
        try:
            expires = datetime.fromisoformat(body.expires_at.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expires_at format.")

    fc = FreeCode(code=body.code, max_uses=body.max_uses, expires_at=expires)
    db.add(fc)
    await _audit(db, "free_code_created", f"code={body.code} max_uses={body.max_uses}")
    await db.commit()
    await db.refresh(fc)
    logger.info("Admin %s created free code %s", admin_user.id, body.code)
    return {
        "status": "success",
        "message": "Free code created.",
        "data": {"id": str(fc.id), "code": fc.code},
    }


@router.delete("/free-codes/{code_id}", summary="Deactivate a free submission code")
async def delete_free_code(
    code_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    result = await db.execute(select(FreeCode).where(FreeCode.id == code_id))
    fc = result.scalar_one_or_none()
    if fc is None:
        raise HTTPException(status_code=404, detail="Code not found.")

    fc.active = False
    await _audit(db, "free_code_deactivated", f"code={fc.code}")
    await db.commit()
    logger.info("Admin %s deactivated free code %s", admin_user.id, fc.code)
    return {"status": "success", "message": "Code deactivated."}


# ---------------------------------------------------------------------------
# Task 7 — Submission Administration
# ---------------------------------------------------------------------------

_SUB_STATUS_MAP = {
    "processing": [SubmissionStatus.ANALYZING],
    "stuck": [SubmissionStatus.ANALYZING],
    "in_queue": [SubmissionStatus.READY_FOR_REVIEW, SubmissionStatus.IN_REVIEW],
    "completed": [SubmissionStatus.COMPLETED],
}


@router.get("/submissions", summary="List submissions with optional status filter")
async def list_submissions(
    status_filter: Optional[str] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    per_page = 50
    offset = (page - 1) * per_page
    now = get_current_utc()

    q = (
        select(Submission, User.email.label("user_email"), Coach, User.name.label("user_name"))
        .join(User, User.id == Submission.user_id)
        .outerjoin(Coach, Coach.id == Submission.coach_id)
        .order_by(Submission.created_at.desc())
    )

    if status_filter == "stuck":
        cutoff = now - timedelta(minutes=30)
        q = q.where(
            Submission.status == SubmissionStatus.ANALYZING,
            Submission.updated_at <= cutoff,
        )
    elif status_filter and status_filter in _SUB_STATUS_MAP:
        q = q.where(Submission.status.in_(_SUB_STATUS_MAP[status_filter]))
    elif status_filter:
        raise HTTPException(status_code=400, detail=f"Unknown status filter '{status_filter}'.")

    q = q.offset(offset).limit(per_page)
    rows = (await db.execute(q)).all()

    items = []
    for sub, user_email, coach, user_name in rows:
        coach_name = None
        if coach:
            coach_user_row = await db.execute(select(User).where(User.id == coach.user_id))
            coach_user = coach_user_row.scalar_one_or_none()
            coach_name = coach_user.name if coach_user else None

        items.append({
            "id": str(sub.id),
            "user_email": user_email,
            "created_at": sub.created_at.isoformat(),
            "club_type": sub.club_type,
            "status": sub.status.value,
            "coach_name": coach_name,
        })

    return {"status": "success", "data": {"items": items, "page": page}}


class ReassignRequest(BaseModel):
    coach_id: uuid.UUID


@router.put("/submissions/{submission_id}/reassign", summary="Reassign submission to a different coach")
async def reassign_submission(
    submission_id: uuid.UUID,
    body: ReassignRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    sub_result = await db.execute(select(Submission).where(Submission.id == submission_id))
    sub = sub_result.scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    coach_result = await db.execute(select(Coach).where(Coach.id == body.coach_id, Coach.is_active == True))  # noqa: E712
    coach = coach_result.scalar_one_or_none()
    if coach is None:
        raise HTTPException(status_code=404, detail="Active coach not found.")

    old_coach_id = str(sub.coach_id) if sub.coach_id else "none"
    sub.coach_id = body.coach_id
    await _audit(db, "submission_reassigned", f"submission_id={submission_id} from={old_coach_id} to={body.coach_id}")
    await db.commit()
    return {"status": "success", "message": "Submission reassigned."}


@router.post("/submissions/{submission_id}/reprocess", summary="Re-run the AI pipeline on a submission")
async def reprocess_submission(
    submission_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    sub_result = await db.execute(select(Submission).where(Submission.id == submission_id))
    sub = sub_result.scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    sub.status = SubmissionStatus.ANALYZING
    await _audit(db, "submission_reprocessed", f"submission_id={submission_id}")
    await db.commit()

    # Dispatch the Celery task
    try:
        from app.workers.video_tasks import process_golf_swing
        process_golf_swing.delay(str(submission_id))
        logger.info("Admin %s triggered reprocess for submission %s", admin_user.id, submission_id)
    except Exception as exc:
        logger.error("Failed to dispatch reprocess task for %s: %s", submission_id, exc)

    return {"status": "success", "message": "Submission queued for reprocessing."}


@router.delete("/submissions/{submission_id}", summary="Delete a submission (not the user)")
async def delete_submission(
    submission_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    sub_result = await db.execute(select(Submission).where(Submission.id == submission_id))
    sub = sub_result.scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    await _audit(db, "submission_deleted", f"submission_id={submission_id} user_id={sub.user_id}")
    await db.delete(sub)
    await db.commit()
    logger.info("Admin %s deleted submission %s", admin_user.id, submission_id)
    return {"status": "success", "message": "Submission deleted."}


# ---------------------------------------------------------------------------
# Task 8 — Site Announcement Banner (admin side, auth required)
# ---------------------------------------------------------------------------

async def _get_or_create_announcement(db: AsyncSession) -> Announcement:
    result = await db.execute(select(Announcement).limit(1))
    ann = result.scalar_one_or_none()
    if ann is None:
        ann = Announcement(message="", active=False)
        db.add(ann)
        await db.flush()
    return ann


@router.get("/announcement", summary="Get current site announcement (admin)")
async def get_announcement_admin(
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    ann = await _get_or_create_announcement(db)
    return {"status": "success", "data": {"message": ann.message, "active": ann.active}}


class AnnouncementUpdate(BaseModel):
    message: str
    active: bool


@router.put("/announcement", summary="Set site announcement banner")
async def update_announcement(
    body: AnnouncementUpdate,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    ann = await _get_or_create_announcement(db)
    ann.message = body.message
    ann.active = body.active
    await _audit(db, "banner_changed", f"active={body.active} message={body.message[:80]}")
    await db.commit()
    logger.info("Admin %s updated announcement active=%s", admin_user.id, body.active)
    return {"status": "success", "message": "Announcement updated.", "data": {"message": ann.message, "active": ann.active}}


# ---------------------------------------------------------------------------
# Task 9 — Admin Audit Log
# ---------------------------------------------------------------------------

@router.get("/audit-log", summary="Retrieve admin action log, newest first")
async def get_audit_log(
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(_require_admin),
):
    per_page = 50
    offset = (page - 1) * per_page
    rows = (await db.execute(
        select(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .offset(offset)
        .limit(per_page)
    )).scalars().all()

    items = [
        {
            "id": str(entry.id),
            "timestamp": entry.timestamp.isoformat(),
            "action": entry.action,
            "detail": entry.detail,
        }
        for entry in rows
    ]
    return {"status": "success", "data": {"items": items, "page": page}}


# ---------------------------------------------------------------------------
# Legacy GET /admin/settings (kept for backward compat)
# ---------------------------------------------------------------------------

@router.get("/settings", summary="Get current effective pricing settings (legacy)")
async def get_settings(admin_user: User = Depends(_require_admin)):
    return {
        "status": "success",
        "message": "Current effective settings.",
        "data": admin_settings.get_effective_settings(),
    }
