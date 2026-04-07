import logging
import time
import urllib.parse
import uuid
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_AUTH_BASE = "https://www.tiktok.com/v2/auth/authorize/"
_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
_UPLOAD_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
_UPLOAD_STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"


class TikTokService:
    """
    TikTok Content Posting API integration.
    Handles OAuth2 flow and video uploads via the direct-post method.
    """

    # ------------------------------------------------------------------
    # OAuth2 flow
    # ------------------------------------------------------------------

    def get_auth_url(self, user_id: str) -> str:
        """
        Generate the TikTok OAuth2 authorisation URL.
        Encodes *user_id* in the state parameter.
        """
        state = f"{user_id}:{uuid.uuid4().hex}"
        params = {
            "client_key": settings.TIKTOK_CLIENT_KEY,
            "redirect_uri": settings.TIKTOK_REDIRECT_URI,
            "response_type": "code",
            "scope": "user.info.basic,video.publish",
            "state": state,
        }
        return f"{_AUTH_BASE}?{urllib.parse.urlencode(params)}"

    def handle_callback(self, code: str) -> Dict[str, Any]:
        """
        Exchange an authorisation code for an access token.
        Returns {access_token, refresh_token, open_id, expires_in}.
        Raises RuntimeError on failure.
        """
        import requests

        try:
            resp = requests.post(
                _TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "client_key": settings.TIKTOK_CLIENT_KEY,
                    "client_secret": settings.TIKTOK_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": settings.TIKTOK_REDIRECT_URI,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "open_id": data.get("open_id"),
                "expires_in": data.get("expires_in", 86400),
            }
        except Exception as exc:
            logger.error("TikTok OAuth callback failed: %s", exc)
            raise RuntimeError(f"TikTok authorisation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Video upload
    # ------------------------------------------------------------------

    def upload_video(
        self,
        access_token: str,
        video_path: str,
        title: str,
        max_retries: int = 3,
    ) -> str:
        """
        Upload *video_path* to TikTok using the direct-post API.
        Returns the TikTok video URL.
        Retries up to *max_retries* times on failure.
        Raises RuntimeError if all retries are exhausted.
        """
        import os
        import requests

        file_size = os.path.getsize(video_path)
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                # Step 1 — Initialise upload
                init_resp = requests.post(
                    _UPLOAD_INIT_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json; charset=UTF-8",
                    },
                    json={
                        "post_info": {
                            "title": title[:150],
                            "privacy_level": "PUBLIC_TO_EVERYONE",
                            "disable_duet": False,
                            "disable_stitch": False,
                            "disable_comment": False,
                            "video_cover_timestamp_ms": 1000,
                        },
                        "source_info": {
                            "source": "FILE_UPLOAD",
                            "video_size": file_size,
                            "chunk_size": file_size,
                            "total_chunk_count": 1,
                        },
                    },
                    timeout=30,
                )
                init_resp.raise_for_status()
                init_data = init_resp.json().get("data", {})
                upload_url = init_data.get("upload_url")
                publish_id = init_data.get("publish_id")

                if not upload_url or not publish_id:
                    raise ValueError(
                        f"TikTok init response missing upload_url or publish_id: {init_data}"
                    )

                # Step 2 — Upload the file
                with open(video_path, "rb") as fh:
                    video_data = fh.read()

                upload_resp = requests.put(
                    upload_url,
                    headers={
                        "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                        "Content-Type": "video/mp4",
                    },
                    data=video_data,
                    timeout=300,
                )
                upload_resp.raise_for_status()

                # Step 3 — Poll for publish completion (max 30 s)
                video_url = self._poll_publish_status(
                    access_token, publish_id, timeout=30
                )
                logger.info(
                    "TikTok upload OK (attempt %d/%d): %s",
                    attempt, max_retries, video_url,
                )
                return video_url

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "TikTok upload attempt %d/%d failed: %s", attempt, max_retries, exc
                )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError(
            f"TikTok upload failed after {max_retries} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poll_publish_status(
        self, access_token: str, publish_id: str, timeout: int = 30
    ) -> str:
        """
        Poll the TikTok publish status endpoint until the video is live
        or *timeout* seconds elapse.  Returns the share URL.
        """
        import requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.post(
                    _UPLOAD_STATUS_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json; charset=UTF-8",
                    },
                    json={"publish_id": publish_id},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})
                pub_status = data.get("status", "")

                if pub_status == "PUBLISH_COMPLETE":
                    share_url = data.get("share_url", "")
                    if share_url:
                        return share_url
                    # share_url not always immediately available
                    return f"https://www.tiktok.com/@user/video/{publish_id}"

                if pub_status in {"FAILED", "SPAM_RISK_CREATOR_BANNED"}:
                    raise RuntimeError(
                        f"TikTok publish failed with status: {pub_status}"
                    )

            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("TikTok status poll error: %s", exc)

            time.sleep(2)

        raise RuntimeError(
            f"TikTok publish timed out after {timeout}s for publish_id={publish_id}"
        )


tiktok_service = TikTokService()
