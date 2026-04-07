import logging
import os
import tempfile
import time
import uuid
from typing import Dict, Optional

import b2sdk.v2 as b2

from app.config import settings

logger = logging.getLogger(__name__)


class BackblazeService:
    """
    Wrapper around b2sdk for all Backblaze B2 operations.
    Lazily initialises the SDK connection on first use.
    """

    def __init__(self) -> None:
        self._api: Optional[b2.B2Api] = None
        self._bucket: Optional[b2.Bucket] = None

    # ------------------------------------------------------------------
    # Internal: lazy connection
    # ------------------------------------------------------------------

    def _get_bucket(self) -> b2.Bucket:
        if self._bucket is None:
            info = b2.InMemoryAccountInfo()
            self._api = b2.B2Api(info)
            self._api.authorize_account(
                "production",
                settings.B2_APPLICATION_KEY_ID,
                settings.B2_APPLICATION_KEY,
            )
            self._bucket = self._api.get_bucket_by_name(settings.B2_BUCKET_NAME)
            logger.info("Backblaze B2 connection established (bucket: %s).", settings.B2_BUCKET_NAME)
        return self._bucket

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_file(
        self,
        file_data: bytes,
        file_name: str,
        content_type: str,
        max_retries: int = 3,
    ) -> Dict[str, str]:
        """
        Upload *file_data* to B2 at the given *file_name* path.

        Returns:
            {"file_url": str, "b2_file_id": str, "file_size": int}

        Retries up to *max_retries* times with exponential back-off.
        Raises RuntimeError if all retries are exhausted.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            tmp_path: Optional[str] = None
            try:
                bucket = self._get_bucket()

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name

                uploaded = bucket.upload_local_file(
                    local_file=tmp_path,
                    file_name=file_name,
                    file_infos={"Content-Type": content_type},
                )

                public_url = self._api.get_download_url_for_fileid(uploaded.id_)
                logger.info(
                    "B2 upload OK (attempt %d/%d): %s id=%s",
                    attempt, max_retries, file_name, uploaded.id_,
                )
                return {
                    "file_url": public_url,
                    "b2_file_id": uploaded.id_,
                    "file_size": len(file_data),
                }

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "B2 upload attempt %d/%d failed for %s: %s",
                    attempt, max_retries, file_name, exc,
                )
                # Force re-connect on next attempt
                self._api = None
                self._bucket = None
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))  # 1 s, 2 s, 4 s …

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        raise RuntimeError(
            f"B2 upload failed after {max_retries} attempts for '{file_name}': {last_exc}"
        )

    def delete_file(self, b2_file_id: str, file_name: str) -> bool:
        """
        Delete a specific file version from B2.
        Returns True on success, False on failure — never raises.
        """
        try:
            bucket = self._get_bucket()
            bucket.delete_file_version(b2_file_id, file_name)
            logger.info("B2 delete OK: %s (id=%s)", file_name, b2_file_id)
            return True
        except Exception as exc:
            logger.warning("B2 delete failed for id=%s: %s", b2_file_id, exc)
            return False

    def generate_download_url(self, file_name: str) -> str:
        """
        Return the public download URL for a file already stored in the bucket.
        Does not require the file to exist — constructs the URL from bucket config.
        """
        bucket = self._get_bucket()
        return f"https://f{settings.B2_BUCKET_ID}.backblazeb2.com/file/{settings.B2_BUCKET_NAME}/{file_name}"

    def validate_connection(self) -> bool:
        """
        Test the B2 connection and bucket access.
        Returns True when healthy, False otherwise. Never raises.
        """
        try:
            self._get_bucket()
            logger.info("B2 connection validated successfully.")
            return True
        except Exception as exc:
            logger.error("B2 connection validation failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Path helper (static — no SDK needed)
    # ------------------------------------------------------------------

    @staticmethod
    def build_destination_path(
        user_id: str,
        submission_id: str,
        file_type: str,
        original_filename: str,
    ) -> str:
        """
        Build a collision-safe B2 object path.
        Format: submissions/{user_id}/{submission_id}/{file_type}/{uuid4}.{ext}
        """
        ext = os.path.splitext(original_filename)[1].lower() or ""
        return f"submissions/{user_id}/{submission_id}/{file_type}/{uuid.uuid4().hex}{ext}"


# Singleton used throughout the application
b2_service = BackblazeService()
