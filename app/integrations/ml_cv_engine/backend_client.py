"""Backend API client for Golf Swing platform integration.

This client targets:
https://golf-swing-backend-production.up.railway.app
"""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from typing import Dict, Optional
from urllib import parse, request
from urllib.error import HTTPError, URLError


class BackendApiError(RuntimeError):
    """Raised when backend API calls fail."""


class BackendClient:
    """Simple HTTP client for submission upload and polling workflows."""

    def __init__(
        self,
        base_url: str = "https://golf-swing-backend-production.up.railway.app",
        token: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)

    def set_token(self, token: str) -> None:
        """Update bearer token."""
        self.token = token

    def health(self) -> dict:
        """Call backend health endpoint."""
        return self._request_json("GET", "/health")

    def create_submission(self) -> dict:
        """Create a new empty submission."""
        return self._request_json("POST", "/api/v1/submissions/create", auth=True)

    def upload_images(
        self,
        submission_id: str,
        front_image: str,
        left_image: str,
        right_image: str,
        back_image: str,
    ) -> dict:
        """Upload four required images for a submission."""
        fields = {
            "front_image": front_image,
            "left_image": left_image,
            "right_image": right_image,
            "back_image": back_image,
        }
        path = f"/api/v1/submissions/{submission_id}/upload-images"
        return self._request_multipart("POST", path, files=fields, auth=True)

    def upload_video(self, submission_id: str, swing_video: str) -> dict:
        """Upload swing video for a submission."""
        path = f"/api/v1/submissions/{submission_id}/upload-video"
        return self._request_multipart("POST", path, files={"swing_video": swing_video}, auth=True)

    def submit_for_analysis(self, submission_id: str) -> dict:
        """Finalize submission and trigger analysis."""
        path = f"/api/v1/submissions/{submission_id}/submit-for-analysis"
        return self._request_json("POST", path, auth=True)

    def get_submission_status(self, submission_id: str) -> dict:
        """Get submission status for progress polling."""
        path = f"/api/v1/submissions/{submission_id}/status"
        return self._request_json("GET", path, auth=True)

    def get_submission(self, submission_id: str) -> dict:
        """Get full submission details."""
        path = f"/api/v1/submissions/{submission_id}"
        return self._request_json("GET", path, auth=True)

    def get_avatar(self, submission_id: str) -> dict:
        """Get avatar metadata/URLs for a submission."""
        path = f"/api/v1/avatar/{submission_id}/avatar"
        return self._request_json("GET", path, auth=True)

    def get_skeleton(self, submission_id: str) -> dict:
        """Get skeleton JSON stored by backend."""
        path = f"/api/v1/avatar/{submission_id}/avatar/skeleton"
        return self._request_json("GET", path, auth=True)

    def get_angle_view(self, submission_id: str, angle: str) -> dict:
        """Get one angle image URL for avatar viewer."""
        path = f"/api/v1/avatar/{submission_id}/avatar/angles/{angle}"
        return self._request_json("GET", path, auth=True)

    def upload_and_submit(
        self,
        submission_id: str,
        image_paths: Dict[str, str],
        video_path: str,
    ) -> dict:
        """Convenience workflow: upload images + video + submit."""
        self.upload_images(
            submission_id=submission_id,
            front_image=image_paths["front"],
            left_image=image_paths["left"],
            right_image=image_paths["right"],
            back_image=image_paths["back"],
        )
        self.upload_video(submission_id=submission_id, swing_video=video_path)
        return self.submit_for_analysis(submission_id=submission_id)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        auth: bool = False,
        payload: Optional[dict] = None,
        query: Optional[dict] = None,
    ) -> dict:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{parse.urlencode(query)}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if auth:
            headers.update(self._auth_header())
        req = request.Request(url=url, data=body, method=method.upper(), headers=headers)
        return self._execute(req)

    def _request_multipart(self, method: str, path: str, *, files: Dict[str, str], auth: bool) -> dict:
        url = f"{self.base_url}{path}"
        boundary = f"----mlcv-{uuid.uuid4().hex}"
        body = self._encode_multipart(files, boundary)
        headers = {
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }
        if auth:
            headers.update(self._auth_header())
        req = request.Request(url=url, data=body, method=method.upper(), headers=headers)
        return self._execute(req)

    def _execute(self, req: request.Request) -> dict:
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise BackendApiError(f"HTTP {exc.code} for {req.full_url}: {payload}") from exc
        except URLError as exc:
            raise BackendApiError(f"Network error for {req.full_url}: {exc}") from exc
        except json.JSONDecodeError:
            return {}

    def _auth_header(self) -> dict:
        if not self.token:
            raise BackendApiError("Bearer token is required for this endpoint.")
        return {"Authorization": f"Bearer {self.token}"}

    @staticmethod
    def _encode_multipart(files: Dict[str, str], boundary: str) -> bytes:
        chunks: list[bytes] = []
        b = boundary.encode("utf-8")
        for field_name, file_path in files.items():
            if not os.path.exists(file_path):
                raise BackendApiError(f"File not found: {file_path}")
            filename = os.path.basename(file_path)
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            with open(file_path, "rb") as f:
                content = f.read()

            chunks.extend(
                [
                    b"--" + b + b"\r\n",
                    f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode(
                        "utf-8"
                    ),
                    f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                    content,
                    b"\r\n",
                ]
            )
        chunks.append(b"--" + b + b"--\r\n")
        return b"".join(chunks)
