import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

AttireOption = Literal[
    "polo_shorts",
    "casual_shirt_shorts",
    "long_sleeve_pants",
    "athletic_shirt_shorts",
    "business_casual_vest",
]


class JointData(BaseModel):
    id: int
    name: str
    x: float
    y: float
    z: float
    confidence: float
    wx: Optional[float] = None
    wy: Optional[float] = None
    wz: Optional[float] = None


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


class AvatarCustomizationRequest(BaseModel):
    skin_tone_value: Optional[int] = Field(None, ge=0, le=100)
    body_thickness_value: Optional[int] = Field(None, ge=0, le=100)
    attire_selection: Optional[AttireOption] = None
    shoe_color_value: Optional[float] = Field(None, ge=0.0, le=1.0)


class AvatarResponse(BaseModel):
    id: uuid.UUID
    submission_id: uuid.UUID
    status: str
    skeleton_data: Optional[SkeletonData] = None
    avatar_glb_url: Optional[str] = None
    avatar_fbx_url: Optional[str] = None
    angles: List[AvatarAngleResponse] = []
    skin_tone_value: Optional[int] = None
    body_thickness_value: Optional[int] = None
    recommended_avatar_id: Optional[str] = None
    attire_selection: Optional[str] = None
    shoe_color_value: Optional[float] = None
    headgear_enabled: bool = True
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
