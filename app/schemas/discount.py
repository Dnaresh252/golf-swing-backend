import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class DiscountCodeResponse(BaseModel):
    id: uuid.UUID
    code: str
    discount_percent: int
    valid_until: datetime
    used: bool
    used_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DiscountCodesListResponse(BaseModel):
    items: List[DiscountCodeResponse]
    total: int
    active_count: int
    used_count: int
