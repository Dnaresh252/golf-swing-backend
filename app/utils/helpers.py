import math
import random
import re
import string
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.constants import DEFAULT_LIMIT, DEFAULT_PAGE, MAX_LIMIT

T = TypeVar("T")


def generate_request_id() -> str:
    """Return a new UUID4 string for request tracing."""
    return str(uuid.uuid4())


def get_current_utc() -> datetime:
    """Return current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def format_file_size(size_bytes: int) -> str:
    """Convert raw bytes to a human-readable string (e.g. '10.5 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.1f} GB"


def safe_filename(filename: str) -> str:
    """
    Sanitise a filename:
    - Strip directory components (prevent path traversal).
    - Replace whitespace with underscores.
    - Remove all characters that are not alphanumeric, dots, dashes, or underscores.
    - Truncate to 255 characters.
    """
    # Strip any path components
    filename = filename.replace("\\", "/").split("/")[-1]
    # Normalise whitespace
    filename = re.sub(r"\s+", "_", filename)
    # Keep only safe characters
    filename = re.sub(r"[^\w.\-]", "", filename)
    # Prevent leading dots (hidden files / double-extension tricks)
    filename = filename.lstrip(".")
    return filename[:255] or "upload"


def _random_suffix(length: int = 5) -> str:
    """Return a random uppercase-alphanumeric string of *length* characters."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


async def generate_discount_code(
    db: AsyncSession,
    prefix: str = "GOLF2024",
    max_attempts: int = 10,
) -> str:
    """
    Generate a unique discount code in the format PREFIX-XXXXX.
    Checks the database to guarantee uniqueness before returning.
    Raises RuntimeError if a unique code cannot be produced within max_attempts.
    """
    from app.models.discount import DiscountCode  # local import avoids circular deps

    for _ in range(max_attempts):
        candidate = f"{prefix}-{_random_suffix(5)}"
        result = await db.execute(
            select(DiscountCode).where(DiscountCode.code == candidate).limit(1)
        )
        if result.scalar_one_or_none() is None:
            return candidate

    raise RuntimeError(
        f"Could not generate a unique discount code after {max_attempts} attempts."
    )


async def paginate_query(
    db: AsyncSession,
    query,
    model,
    page: int = DEFAULT_PAGE,
    limit: int = DEFAULT_LIMIT,
) -> Dict[str, Any]:
    """
    Execute a paginated query and return a dict with items, total, pages, page, limit.

    :param query:  A SQLAlchemy select() statement (without offset/limit applied).
    :param model:  The ORM model class (used for the COUNT sub-query).
    :param page:   1-based page number.
    :param limit:  Items per page (capped at MAX_LIMIT).
    """
    page = max(1, page)
    limit = max(1, min(limit, MAX_LIMIT))
    offset = (page - 1) * limit

    # Total count
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total: int = count_result.scalar_one()

    # Paginated items
    paginated = await db.execute(query.offset(offset).limit(limit))
    items: List[Any] = paginated.scalars().all()

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": math.ceil(total / limit) if total else 0,
    }
