import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class JointData(BaseModel):
    id: int
    name: str
    x: float
    y: float
    z: float
    confidence: float


class FrameData(BaseModel):
    frame_num: int
    timestamp: float
    joints: List[JointData]


class SkeletonData(BaseModel):
    submission_id: uuid.UUID
    frames: List[FrameData]


class AvatarAngleResponse(BaseModel):
    angle: str
    image_url: str


class AvatarResponse(BaseModel):
    id: uuid.UUID
    submission_id: uuid.UUID
    status: str
    skeleton_data: Optional[SkeletonData] = None
    avatar_glb_url: Optional[str] = None
    avatar_fbx_url: Optional[str] = None
    angles: List[AvatarAngleResponse] = []
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
