"""
GCP Cloud Storage (GCS) Backend — Mentera RAG Pipeline.

Implements the BaseObjectStore interface using google-cloud-storage
to generate signed upload URLs, download, and delete files.
Uses lazy imports to prevent dependency failures on systems without the GCP SDK.
"""

import logging
from datetime import timedelta
from typing import Any

from mentera_rag.storage.base import BaseObjectStore

logger = logging.getLogger(__name__)


class GCSStorageStore(BaseObjectStore):
    """
    Object storage backend backed by Google Cloud Storage (GCS).
    """

    def __init__(self, bucket_name: str = "mentera-uploads", credentials_path: str | None = None):
        """
        Initialize the GCS Storage backend.

        Args:
            bucket_name: GCS bucket name.
            credentials_path: Path to GCP service account JSON key file (optional).
        """
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy-initialize GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage

                if self.credentials_path:
                    self._client = storage.Client.from_service_account_json(self.credentials_path)
                else:
                    self._client = storage.Client()
            except ImportError as e:
                raise ImportError(
                    "google-cloud-storage is required to use GCS. "
                    "Install with: pip install 'mentera-rag[storage]'"
                ) from e
        return self._client

    def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a signed PUT URL for direct file upload to GCS.
        """
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(key)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method="PUT",
                content_type=content_type,
            )
            return str(url)
        except Exception as e:
            logger.error("Failed to generate GCS signed URL: %s", e)
            raise RuntimeError("Failed to generate presigned upload URL.") from e

    def download(self, key: str) -> bytes:
        """
        Download blob bytes from GCS.
        """
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(key)
            return bytes(blob.download_as_bytes())
        except Exception as e:
            # Check for GCS NotFound/404 exception
            if "404" in str(e) or "NotFound" in str(e):
                raise FileNotFoundError(f"GCS key not found: {key}") from e
            logger.error("Failed to download object from GCS: %s", e)
            raise RuntimeError(f"Failed to download key {key} from GCS.") from e

    def delete(self, key: str) -> None:
        """
        Delete a blob from GCS.
        """
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(key)
            blob.delete()
        except Exception as e:
            logger.error("Failed to delete object from GCS: %s", e)
            raise RuntimeError(f"Failed to delete key {key} from GCS.") from e

    def exists(self, key: str) -> bool:
        """
        Check if a blob exists in GCS.
        """
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(key)
            return bool(blob.exists())
        except Exception as e:
            logger.warning("Error checking GCS key existence: %s", e)
            return False
