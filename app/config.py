from functools import lru_cache
from typing import Any, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, DotEnvSettingsSource, SettingsConfigDict
from pydantic.fields import FieldInfo


class _CommaSepDotEnvSource(DotEnvSettingsSource):
    """
    Custom dotenv source that accepts comma-separated strings for List[str] fields.

    Overrides decode_complex_value so that when json.loads() fails on a plain
    comma-separated string, it falls back to splitting by comma instead of
    raising SettingsError.
    """

    def decode_complex_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
    ) -> Any:
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            raise


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            _CommaSepDotEnvSource(
                settings_cls,
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=True,
            ),
            file_secret_settings,
        )

    # Application
    APP_NAME: str = "Golf Swing AI Platform"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False
    SECRET_KEY: str
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis / Celery
    REDIS_URL: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Backblaze B2
    B2_APPLICATION_KEY_ID: str
    B2_APPLICATION_KEY: str
    B2_BUCKET_NAME: str
    B2_BUCKET_ID: str

    # File upload limits
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 100
    MAX_VIDEO_DURATION_SECONDS: int = 15
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png"]
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/quicktime"]

    # SendGrid
    SENDGRID_API_KEY: str
    SENDGRID_FROM_EMAIL: str
    SENDGRID_FROM_NAME: str = "Golf Swing AI"
    SENDGRID_WELCOME_TEMPLATE: str = ""
    SENDGRID_UPLOAD_CONFIRM_TEMPLATE: str = ""
    SENDGRID_ANALYSIS_COMPLETE_TEMPLATE: str = ""
    SENDGRID_COACH_APPROVAL_TEMPLATE: str = ""
    SENDGRID_DISCOUNT_EARNED_TEMPLATE: str = ""

    # YouTube
    YOUTUBE_CLIENT_ID: str = ""
    YOUTUBE_CLIENT_SECRET: str = ""
    YOUTUBE_REDIRECT_URI: str = "http://localhost:8000/auth/youtube/callback"

    # TikTok
    TIKTOK_CLIENT_KEY: str = ""
    TIKTOK_CLIENT_SECRET: str = ""
    TIKTOK_REDIRECT_URI: str = "http://localhost:8000/auth/tiktok/callback"

    # Discounts
    DISCOUNT_CODE_PREFIX: str = "GOLF2024"
    DISCOUNT_PERCENTAGE: int = 15
    DISCOUNT_VALIDITY_DAYS: int = 30

    # Celery workers
    CELERY_WORKER_CONCURRENCY: int = 4
    CELERY_TASK_TIMEOUT: int = 300
    CELERY_MAX_RETRIES: int = 3

    # MediaPipe
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5

    # FFmpeg
    FFMPEG_PATH: str = "ffmpeg"
    VIDEO_OUTPUT_WIDTH: int = 1280
    VIDEO_OUTPUT_HEIGHT: int = 720
    AVATAR_RENDER_SIZE: int = 512

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    @field_validator("ALLOWED_ORIGINS", "ALLOWED_IMAGE_TYPES", "ALLOWED_VIDEO_TYPES", mode="before")
    @classmethod
    def parse_comma_separated(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def max_image_size_bytes(self) -> int:
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024

    @property
    def max_video_size_bytes(self) -> int:
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
