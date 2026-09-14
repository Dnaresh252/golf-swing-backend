"""
Sentry error tracking, shared by the API, the Celery worker and the purge job.

Entirely optional: with no SENTRY_DSN set, init_error_tracking() does nothing.

Privacy (agreed with Stan, 2026-09-14): error reports carry no customer
personal data. send_default_pii=False alone is not enough here, because
several log lines include email addresses and Sentry turns log records into
breadcrumbs, and because local variables and request bodies can hold
passwords and tokens. So, in addition:
  - request bodies, cookies, headers and query strings are never sent
  - local variables are not captured in stack frames
  - the user block is dropped
  - email addresses, IP addresses, bearer tokens, JWTs and the tokens inside
    Redis keys (password reset, email verification, revoked sessions) are
    redacted from every string in the event and its breadcrumbs before it
    leaves the server. Rate-limit log lines carry client IPs, and the Redis
    integration records keys, so both would otherwise reach Sentry.
"""
import logging
import re
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_REDIS_TOKEN_KEY = re.compile(r"\b(pwd_reset|email_verify_user|email_verify|blacklist):[^\s'\",]+")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6 = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}\b")
_SENSITIVE_KEYS = ("password", "token", "secret", "authorization", "cookie", "api_key", "apikey", "dsn")

_initialised = False


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        value = _JWT.sub("[jwt]", value)
        value = _BEARER.sub("Bearer [redacted]", value)
        value = _REDIS_TOKEN_KEY.sub(r"\1:[redacted]", value)
        value = _EMAIL.sub("[email]", value)
        value = _IPV6.sub("[ip]", value)
        return _IPV4.sub("[ip]", value)
    if isinstance(value, dict):
        return {
            k: ("[redacted]" if isinstance(k, str) and any(s in k.lower() for s in _SENSITIVE_KEYS) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_redact(v) for v in value)
    return value


def scrub_event(event: dict, hint: Any = None) -> dict:
    event.pop("user", None)
    request = event.get("request")
    if isinstance(request, dict):
        url = request.get("url")
        event["request"] = {
            "method": request.get("method"),
            "url": url.split("?", 1)[0] if isinstance(url, str) else url,
        }
    return _redact(event)


def scrub_breadcrumb(crumb: dict, hint: Any = None) -> dict:
    data = crumb.get("data")
    if isinstance(data, dict) and isinstance(data.get("url"), str):
        data["url"] = data["url"].split("?", 1)[0]
    return _redact(crumb)


def init_error_tracking(component: str) -> bool:
    """Initialise Sentry once per process. Never raises; returns True if enabled."""
    global _initialised
    if _initialised or not settings.SENTRY_DSN:
        return _initialised
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.APP_ENV,
            release=settings.APP_VERSION,
            send_default_pii=False,
            max_request_body_size="never",
            include_local_variables=False,
            traces_sample_rate=0.0,
            before_send=scrub_event,
            before_breadcrumb=scrub_breadcrumb,
        )
        sentry_sdk.set_tag("component", component)
        _initialised = True
        logger.info("Sentry error tracking enabled (%s).", component)
    except Exception as exc:  # never let telemetry stop a process starting
        logger.warning("Sentry not enabled: %s", exc)
    return _initialised
