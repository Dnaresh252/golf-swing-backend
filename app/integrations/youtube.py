import logging
import time
import urllib.parse
import uuid
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_AUTH_BASE = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"


class YouTubeService:
    """
    YouTube Data API v3 integration.
    Handles OAuth2 flow and video uploads with retry logic.
    """

    # ------------------------------------------------------------------
    # OAuth2 flow
    # ------------------------------------------------------------------

    def get_auth_url(self, user_id: str) -> str:
        """
        Generate the OAuth2 authorisation URL.
        *user_id* is encoded in the state parameter so the callback
        can associate the tokens with the right account.
        """
        state = f"{user_id}:{uuid.uuid4().hex}"
        params = {
            "client_id": settings.YOUTUBE_CLIENT_ID,
            "redirect_uri": settings.YOUTUBE_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{_AUTH_BASE}?{urllib.parse.urlencode(params)}"

    def handle_callback(self, code: str, state: str) -> Dict[str, Any]:
        """
        Exchange an authorisation code for access + refresh tokens.
        Returns {access_token, refresh_token, expires_in, user_id}.
        Raises RuntimeError on failure.
        """
        import requests

        try:
            resp = requests.post(
                _TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.YOUTUBE_CLIENT_ID,
                    "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                    "redirect_uri": settings.YOUTUBE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            user_id = state.split(":")[0] if ":" in state else state
            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "expires_in": data.get("expires_in", 3600),
                "user_id": user_id,
            }
        except Exception as exc:
            logger.error("YouTube OAuth callback failed: %s", exc)
            raise RuntimeError(f"YouTube authorisation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Video upload
    # ------------------------------------------------------------------

    def upload_video(
        self,
        credentials: Dict[str, Any],
        video_path: str,
        title: str,
        description: str,
        max_retries: int = 3,
    ) -> str:
        """
        Upload *video_path* to YouTube and return the public video URL.
        Refreshes the token if expired.  Retries up to *max_retries* times
        on network errors.  Raises RuntimeError if all retries fail.
        """
        import requests

        credentials = self._refresh_token_if_needed(credentials)
        access_token = credentials["access_token"]

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                # Resumable upload — metadata first
                metadata_resp = requests.post(
                    f"{_UPLOAD_URL}?uploadType=resumable&part=snippet,status",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "X-Upload-Content-Type": "video/*",
                    },
                    json={
                        "snippet": {
                            "title": title[:100],
                            "description": description[:5000],
                            "categoryId": "17",  # Sports
                        },
                        "status": {"privacyStatus": "public"},
                    },
                    timeout=30,
                )
                metadata_resp.raise_for_status()
                upload_uri = metadata_resp.headers["Location"]

                # Stream file to resumable URI
                with open(video_path, "rb") as fh:
                    upload_resp = requests.put(
                        upload_uri,
                        data=fh,
                        headers={"Content-Type": "video/*"},
                        timeout=300,
                    )
                upload_resp.raise_for_status()
                video_id = upload_resp.json()["id"]
                url = f"https://www.youtube.com/watch?v={video_id}"
                logger.info(
                    "YouTube upload OK (attempt %d/%d): %s", attempt, max_retries, url
                )
                return url

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "YouTube upload attempt %d/%d failed: %s", attempt, max_retries, exc
                )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError(
            f"YouTube upload failed after {max_retries} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def _refresh_token_if_needed(
        self, credentials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return credentials dict with a fresh access_token if the current
        one is within 60 seconds of expiry.  Original dict is not mutated.
        """
        import requests

        expires_at = credentials.get("expires_at", 0)
        if time.time() < expires_at - 60:
            return credentials

        refresh_token = credentials.get("refresh_token")
        if not refresh_token:
            logger.warning("YouTube: no refresh_token available — using existing access_token.")
            return credentials

        try:
            resp = requests.post(
                _TOKEN_URL,
                data={
                    "client_id": settings.YOUTUBE_CLIENT_ID,
                    "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                **credentials,
                "access_token": data["access_token"],
                "expires_at": time.time() + data.get("expires_in", 3600),
            }
        except Exception as exc:
            logger.warning("YouTube token refresh failed: %s", exc)
            return credentials


youtube_service = YouTubeService()
