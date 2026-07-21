"""
DB-backed admin settings (persist across restarts).

Keys used:
  SUBMISSION_PRICE_CENTS       — price of one submission (int cents)
  ALL_SUBMISSIONS_FREE         — "1"/"0" testing free-mode toggle
  PAYOUT_REVIEW_RATE_CENTS     — per-review coach payout (int cents)
  PAYOUT_PGA_APPROVAL_CENTS    — PGA Pro per-approval bonus (int cents)
"""
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.app_setting import AppSetting

logger = logging.getLogger(__name__)


async def get_setting(db: AsyncSession, key: str, default: Optional[str] = None) -> Optional[str]:
    row = await db.execute(select(AppSetting.value).where(AppSetting.key == key))
    value = row.scalar_one_or_none()
    return value if value is not None else default


async def set_setting(db: AsyncSession, key: str, value: str) -> None:
    row = await db.execute(select(AppSetting).where(AppSetting.key == key))
    setting = row.scalar_one_or_none()
    if setting is None:
        db.add(AppSetting(key=key, value=str(value)))
    else:
        setting.value = str(value)
    # caller commits


async def get_int_setting(db: AsyncSession, key: str, default: int) -> int:
    raw = await get_setting(db, key)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


async def get_bool_setting(db: AsyncSession, key: str, default: bool = False) -> bool:
    raw = await get_setting(db, key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")
