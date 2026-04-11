"""
Shared slowapi rate limiter backed by Redis.

Import `limiter` wherever you need @limiter.limit(...) decorators,
and wire `limiter` + `rate_limit_exceeded_handler` into the FastAPI app.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.REDIS_URL,
    default_limits=[],          # No global default — set per-route
)
