import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, field_validator, model_validator


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    full_name: Optional[str] = None  # accepted as alias for name
    terms_accepted: bool = True  # defaults to accepted; explicit False still rejected

    @model_validator(mode="before")
    @classmethod
    def coerce_full_name(cls, data):
        if isinstance(data, dict) and not data.get("name") and data.get("full_name"):
            data = {**data, "name": data["full_name"]}
        return data

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        import re
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number.")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", v):
            raise ValueError("Password must contain at least one special character.")
        return v

    @field_validator("name")
    @classmethod
    def name_length(cls, v: Optional[str]) -> str:
        if v is None:
            raise ValueError("Name is required.")
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters.")
        if len(v) > 100:
            raise ValueError("Name must not exceed 100 characters.")
        return v

    @model_validator(mode="after")
    def terms_must_be_accepted(self) -> "UserRegister":
        if not self.terms_accepted:
            raise ValueError("You must accept the terms and conditions to register.")
        return self


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class CoachLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    profile_picture_url: Optional[str] = None
    is_verified: bool
    created_at: datetime
    role: str = "user"

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        instance = super().model_validate(obj, *args, **kwargs)
        if getattr(obj, "is_admin", False):
            instance.role = "admin"
        elif obj.__dict__.get("coach_profile") is not None:
            # Only check if already eagerly loaded — avoids async lazy-load MissingGreenlet
            instance.role = "coach"
        else:
            instance.role = "user"
        return instance

    model_config = {"from_attributes": True}


class CoachProfileResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    bio: Optional[str] = None
    rating: float
    submissions_completed: int
    is_active: bool

    model_config = {"from_attributes": True}


class CoachLoginResponse(BaseModel):
    tokens: TokenResponse
    user: UserResponse
    coach: CoachProfileResponse


class AuthResponse(BaseModel):
    tokens: TokenResponse
    user: UserResponse
