import logging
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class SendGridService:
    """
    Centralised SendGrid email service.
    All methods are fire-and-forget — they log failures but never raise.
    """

    # ------------------------------------------------------------------
    # Named email methods
    # ------------------------------------------------------------------

    def send_welcome_email(self, user) -> None:
        self._send_email(
            to_email=user.email,
            template_id=settings.SENDGRID_WELCOME_TEMPLATE,
            dynamic_data={
                "name": user.name,
                "app_name": settings.APP_NAME,
            },
        )

    def send_upload_confirmation(self, user, submission_id: str) -> None:
        self._send_email(
            to_email=user.email,
            template_id=settings.SENDGRID_UPLOAD_CONFIRM_TEMPLATE,
            dynamic_data={
                "name": user.name,
                "submission_id": submission_id,
            },
        )

    def send_analysis_complete(self, user, submission_id: str) -> None:
        self._send_email(
            to_email=user.email,
            template_id=settings.SENDGRID_ANALYSIS_COMPLETE_TEMPLATE,
            dynamic_data={
                "name": user.name,
                "submission_id": submission_id,
                "status_message": "Your avatar is ready for coach review.",
            },
        )

    def send_coach_review_complete(
        self, user, submission_id: str, approved: bool
    ) -> None:
        status_message = (
            "Your swing analysis has been reviewed and corrections have been made."
            if approved
            else "Your submission was rejected by the coach."
        )
        self._send_email(
            to_email=user.email,
            template_id=settings.SENDGRID_COACH_APPROVAL_TEMPLATE,
            dynamic_data={
                "name": user.name,
                "submission_id": submission_id,
                "approved": approved,
                "status_message": status_message,
            },
        )

    def send_discount_earned(
        self,
        user,
        discount_code: str,
        percent: int,
        expiry_date: str,
    ) -> None:
        self._send_email(
            to_email=user.email,
            template_id=settings.SENDGRID_DISCOUNT_EARNED_TEMPLATE,
            dynamic_data={
                "name": user.name,
                "discount_code": discount_code,
                "discount_percent": percent,
                "expiry_date": expiry_date,
            },
        )

    # ------------------------------------------------------------------
    # Core send method
    # ------------------------------------------------------------------

    def _send_email(
        self,
        to_email: str,
        template_id: str,
        dynamic_data: Dict[str, Any],
    ) -> None:
        """
        Send a transactional email via SendGrid dynamic templates.
        Logs success/failure.  Never raises.
        """
        if not template_id:
            logger.warning(
                "SendGrid: skipping email to %s — template_id is not configured.",
                to_email,
            )
            return

        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            message = Mail(
                from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
                to_emails=to_email,
            )
            message.template_id = template_id
            message.dynamic_template_data = dynamic_data

            client = SendGridAPIClient(settings.SENDGRID_API_KEY)
            response = client.send(message)

            logger.info(
                "Email sent to %s via template %s (status=%s)",
                to_email,
                template_id,
                response.status_code,
            )
        except Exception as exc:
            logger.warning(
                "SendGrid failed for %s (template=%s): %s",
                to_email,
                template_id,
                exc,
            )


sendgrid_service = SendGridService()
