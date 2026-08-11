"""
Abstract Base Interface for Object Storage Backends — Mentera RAG Pipeline.

Defines the contract for interacting with raw document storage (S3, GCS,
Azure Blob, or local staging). Encapsulates file downloads, deletions,
existence checks, and presigned upload URL generation.
"""

from abc import ABC, abstractmethod


class BaseObjectStore(ABC):
    """
    Abstract Base Class for all object store providers.
    """

    @abstractmethod
    def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned PUT URL allowing clients to upload a file directly.

        Args:
            key: Target unique storage path/filename in the bucket.
            content_type: Expected MIME type of the upload file.
            expires_in: URL validity lifetime in seconds (default 1 hour).

        Returns:
            Presigned PUT upload URL string.
        """
        pass

    @abstractmethod
    def download(self, key: str) -> bytes:
        """
        Download object bytes from the storage backend.

        Args:
            key: Storage object key path.

        Returns:
            Raw file content bytes.

        Raises:
            FileNotFoundError: If the key does not exist.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Permanently delete an object from the storage backend.

        Args:
            key: Storage object key path.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if an object exists in the storage bucket.

        Args:
            key: Storage object key path.

        Returns:
            True if key exists, False otherwise.
        """
        pass
