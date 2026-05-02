"""
Runtime admin settings overrides.

Values set here override app/config.py defaults for the lifetime of the
process. They are lost on restart. To make them persistent, replace the
dict with a DB-backed store (e.g. a key-value admin_settings table).
"""
from typing import Any, Dict

from app.config import settings

_ALLOWED_KEYS = {"SUBMISSION_PRICE_CENTS", "COACH_PAYOUT_CENTS"}

_overrides: Dict[str, Any] = {}


def get_submission_price_cents() -> int:
    return int(_overrides.get("SUBMISSION_PRICE_CENTS", settings.SUBMISSION_PRICE_CENTS))


def get_coach_payout_cents() -> int:
    return int(_overrides.get("COACH_PAYOUT_CENTS", settings.COACH_PAYOUT_CENTS))


def set_override(key: str, value: Any) -> None:
    if key not in _ALLOWED_KEYS:
        raise ValueError(f"'{key}' is not an allowed admin override key.")
    _overrides[key] = value


def get_effective_settings() -> Dict[str, Any]:
    return {
        "SUBMISSION_PRICE_CENTS": get_submission_price_cents(),
        "COACH_PAYOUT_CENTS": get_coach_payout_cents(),
        "STRIPE_CURRENCY": settings.STRIPE_CURRENCY,
        "DISCOUNT_PERCENTAGE": settings.DISCOUNT_PERCENTAGE,
        "note": "SUBMISSION_PRICE_CENTS and COACH_PAYOUT_CENTS are runtime overrides; others are read-only.",
    }
