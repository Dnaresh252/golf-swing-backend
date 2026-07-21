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

    def send_coach_invitation(
        self, to_email: str, full_name: str, temp_password: str, credential: str
    ) -> None:
        tier = "PGA Pro Golf Coach" if credential == "pga_pro" else "Golf Coach"
        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;color:#1a2b4a">
          <h2 style="color:#1a2b4a">Welcome to GGW Academy, {full_name}!</h2>
          <p>You have been invited to join Golf Game World Academy as a <b>{tier}</b>.</p>
          <p><b>Your login details:</b></p>
          <table style="background:#f6f6f6;border-radius:8px;padding:12px;width:100%">
            <tr><td style="padding:8px 12px"><b>Website:</b></td>
                <td style="padding:8px 12px">https://golfgameworldacademy.com/coach-login</td></tr>
            <tr><td style="padding:8px 12px"><b>Email:</b></td>
                <td style="padding:8px 12px">{to_email}</td></tr>
            <tr><td style="padding:8px 12px"><b>Temporary password:</b></td>
                <td style="padding:8px 12px">{temp_password}</td></tr>
          </table>
          <p>Please sign in and change your password after your first login.</p>
          <p style="color:#888;font-size:12px">Golf Game World LLC</p>
        </div>
        """
        self.send_plain_email(
            to_email=to_email,
            subject="Your GGW Academy Coach Account",
            html_content=html,
        )

    # ------------------------------------------------------------------
    # Plain (non-template) send — needs only the API key
    # ------------------------------------------------------------------

    def send_plain_email(self, to_email: str, subject: str, html_content: str) -> None:
        """Send a plain HTML email (no dynamic template). Never raises."""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            message = Mail(
                from_email=(settings.SENDGRID_FROM_EMAIL, settings.SENDGRID_FROM_NAME),
                to_emails=to_email,
                subject=subject,
                html_content=html_content,
            )
            client = SendGridAPIClient(settings.SENDGRID_API_KEY)
            response = client.send(message)
            logger.info(
                "Plain email sent to %s (subject=%s status=%s)",
                to_email, subject, response.status_code,
            )
        except Exception as exc:
            logger.warning("SendGrid plain email failed for %s: %s", to_email, exc)

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
