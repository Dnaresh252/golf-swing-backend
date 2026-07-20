import logging
import secrets
from typing import Optional, Tuple

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.coach import Coach
from app.models.user import User
from app.schemas.auth import UserRegister
from app.utils.constants import ErrorMessage
from app.utils.security import (
    check_account_locked,
    create_access_token,
    create_refresh_token,
    hash_password,
    record_failed_attempt,
    reset_failed_attempts,
    verify_password,
    verify_token,
)

logger = logging.getLogger(__name__)

# Redis client (shared singleton)
_redis: Optional[aioredis.Redis] = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
    return _redis


def _blacklist_key(token: str) -> str:
    return f"blacklist:{token}"


class AuthService:

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_user(self, db: AsyncSession, user_data: UserRegister) -> User:
        """Create a new user account. Raises ValueError if email already taken."""
        result = await db.execute(
            select(User).where(User.email == user_data.email.lower())
        )
        if result.scalar_one_or_none():
            raise ValueError(ErrorMessage.EMAIL_ALREADY_REGISTERED)

        user = User(
            email=user_data.email.lower(),
            password_hash=hash_password(user_data.password),
            name=user_data.name.strip(),
            is_admin=False,
        )
        db.add(user)
        await db.flush()  # get id without full commit
        await db.refresh(user)
        logger.info("New user registered: %s", user.email)
        return user

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate_user(
        self, db: AsyncSession, email: str, password: str
    ) -> User:
        """
        Verify credentials with lockout enforcement.
        Raises ValueError on any authentication failure (generic message to caller).
        Raises PermissionError when the account is locked.
        """
        email = email.lower()

        if await check_account_locked(email):
            raise PermissionError(ErrorMessage.ACCOUNT_LOCKED)

        result = await db.execute(
            select(User).where(User.email == email)
        )
        user: Optional[User] = result.scalar_one_or_none()

        if user is None or not verify_password(password, user.password_hash):
            await record_failed_attempt(email)
            # Re-check after incrementing so we return a lock message on the 5th bad attempt
            if await check_account_locked(email):
                raise PermissionError(ErrorMessage.ACCOUNT_LOCKED)
            raise ValueError(ErrorMessage.INVALID_CREDENTIALS)

        if not user.is_active:
            raise PermissionError(ErrorMessage.ACCOUNT_INACTIVE)

        if getattr(user, "suspended", False):
            raise PermissionError("Account is suspended. Contact support.")

        await reset_failed_attempts(email)
        from datetime import datetime, timezone
        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
        logger.info("User authenticated: %s", email)
        return user

    # ------------------------------------------------------------------
    # Coach authentication
    # ------------------------------------------------------------------

    async def authenticate_coach(
        self, db: AsyncSession, email: str, password: str
    ) -> Tuple[User, Coach]:
        """
        Authenticate user and verify an active coach profile exists.
        Raises ValueError / PermissionError on failure.
        """
        user = await self.authenticate_user(db, email, password)

        result = await db.execute(
            select(Coach).where(Coach.user_id == user.id)
        )
        coach: Optional[Coach] = result.scalar_one_or_none()

        if coach is None:
            raise PermissionError("Not authorized as coach.")
        if not coach.is_active:
            raise PermissionError("Coach account is inactive.")

        logger.info("Coach authenticated: %s", email)
        return user, coach

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def create_tokens(self, user_id) -> dict:
        """Return a dict with access_token, refresh_token, token_type, expires_in."""
        subject = str(user_id)
        access_token = create_access_token({"sub": subject})
        refresh_token = create_refresh_token({"sub": subject})
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    async def refresh_access_token(self, refresh_token: str) -> str:
        """
        Validate the refresh token and return a new access token string.
        Raises ValueError if the token is invalid or blacklisted.
        """
        if await self._is_blacklisted(refresh_token):
            raise ValueError("Invalid refresh token.")

        payload = verify_token(refresh_token)
        if payload is None or payload.get("type") != "refresh":
            raise ValueError("Invalid refresh token.")

        user_id: str = payload.get("sub", "")
        if not user_id:
            raise ValueError("Invalid refresh token.")

        new_access = create_access_token({"sub": user_id})
        logger.info("Token refreshed for user: %s", user_id)
        return new_access

    async def logout_user(self, token: str) -> bool:
        """
        Blacklist the given token in Redis until its natural expiry.
        Returns True on success.
        """
        payload = verify_token(token)
        ttl = 0
        if payload and "exp" in payload:
            from datetime import datetime, timezone
            exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            now = datetime.now(timezone.utc)
            ttl = max(0, int((exp - now).total_seconds()))

        r = _get_redis()
        if ttl > 0:
            await r.set(_blacklist_key(token), "1", ex=ttl)
        else:
            await r.set(_blacklist_key(token), "1", ex=3600)

        user_id = payload.get("sub", "unknown") if payload else "unknown"
        logger.info("User logged out: %s", user_id)
        return True

    # ------------------------------------------------------------------
    # Current user dependency
    # ------------------------------------------------------------------

    async def get_current_user(self, db: AsyncSession, token: str) -> User:
        """
        Resolve and return the active user from a JWT access token.
        Raises ValueError if the token is invalid, blacklisted, or the user
        is not found / inactive.
        """
        if await self._is_blacklisted(token):
            raise ValueError(ErrorMessage.TOKEN_INVALID)

        payload = verify_token(token)
        if payload is None or payload.get("type") != "access":
            raise ValueError(ErrorMessage.TOKEN_INVALID)

        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise ValueError(ErrorMessage.TOKEN_INVALID)

        result = await db.execute(
            select(User)
            .where(User.id == user_id)
            .options(selectinload(User.coach_profile))
        )
        user: Optional[User] = result.scalar_one_or_none()

        if user is None:
            raise ValueError(ErrorMessage.USER_NOT_FOUND)
        if not user.is_active:
            raise ValueError(ErrorMessage.ACCOUNT_INACTIVE)
        if getattr(user, "suspended", False):
            raise ValueError("Account is suspended.")

        return user

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    async def send_welcome_email(self, user: User) -> None:
        """
        Send a welcome email via SendGrid.
        Never raises — failures are logged and swallowed.
        """
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            message = Mail(
                from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
                to_emails=user.email,
            )
            message.template_id = settings.SENDGRID_WELCOME_TEMPLATE
            message.dynamic_template_data = {
                "name": user.name,
                "app_name": settings.APP_NAME,
            }

            sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
            sg.send(message)
            logger.info("Welcome email sent to: %s", user.email)
        except Exception as exc:
            logger.warning("Failed to send welcome email to %s: %s", user.email, exc)

    async def send_password_reset_email(self, user: User, reset_token: str) -> None:
        """Send password reset email. Never raises."""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Content, Mail

            reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
            message = Mail(
                from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
                to_emails=user.email,
            )
            if settings.SENDGRID_RESET_PASSWORD_TEMPLATE:
                message.template_id = settings.SENDGRID_RESET_PASSWORD_TEMPLATE
                message.dynamic_template_data = {
                    "name": user.name,
                    "reset_link": reset_link,
                    "app_name": settings.APP_NAME,
                }
            else:
                message.subject = f"Reset your {settings.APP_NAME} password"
                message.add_content(Content(
                    "text/plain",
                    f"Hi {user.name},\n\nReset your password (expires in 1 hour):\n\n{reset_link}\n\n"
                    f"If you didn't request this, ignore this email.\n\n— {settings.APP_NAME}",
                ))
            SendGridAPIClient(settings.SENDGRID_API_KEY).send(message)
            logger.info("Password reset email sent to: %s", user.email)
        except Exception as exc:
            logger.warning("Failed to send password reset email to %s: %s", user.email, exc)

    async def forgot_password(self, db: AsyncSession, email: str) -> None:
        """
        Generate a reset token and email it.
        Silently no-ops if email not found to prevent user enumeration.
        """
        result = await db.execute(select(User).where(User.email == email.lower()))
        user: Optional[User] = result.scalar_one_or_none()
        if user is None or not user.is_active:
            return

        token = secrets.token_urlsafe(32)
        r = _get_redis()
        await r.set(f"pwd_reset:{token}", str(user.id), ex=3600)
        await self.send_password_reset_email(user, token)
        logger.info("Password reset requested for user: %s", user.id)

    async def reset_password(self, db: AsyncSession, token: str, new_password: str) -> None:
        """Validate reset token, update password, delete token. Raises ValueError if invalid."""
        r = _get_redis()
        user_id = await r.get(f"pwd_reset:{token}")
        if not user_id:
            raise ValueError("Invalid or expired password reset token.")

        result = await db.execute(select(User).where(User.id == user_id))
        user: Optional[User] = result.scalar_one_or_none()
        if user is None or not user.is_active:
            raise ValueError("Invalid or expired password reset token.")

        user.password_hash = hash_password(new_password)
        await r.delete(f"pwd_reset:{token}")
        await db.commit()
        logger.info("Password reset completed for user: %s", user.id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _is_blacklisted(self, token: str) -> bool:
        r = _get_redis()
        return bool(await r.exists(_blacklist_key(token)))


auth_service = AuthService()
