"""
Local Filesystem Object Storage Backend — Mentera RAG Pipeline.

Simulates cloud object storage by writing files directly to the local filesystem.
Useful for development, testing, and deployment configurations without cloud dependencies.
"""

import os
from pathlib import Path
from mentera_rag.storage.base import BaseObjectStore


class LocalStorageStore(BaseObjectStore):
    """
    Simulates object storage using the local file system.
    """

    def __init__(self, root_dir: str = "/tmp/mentera_storage"):
        """
        Initialize local storage simulator.

        Args:
            root_dir: Base directory where files are stored.
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Resolve a storage key to an absolute local Path, preventing path traversal."""
        # Sanitize key to prevent directory traversal attacks
        safe_key = key.lstrip("/")
        target_path = (self.root_dir / safe_key).resolve()

        if not str(target_path).startswith(str(self.root_dir.resolve())):
            raise ValueError(f"Path traversal detected: {key}")

        return target_path

    def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 3600,
    ) -> str:
        """
        Simulate presigned upload URL by returning a path-based URI.

        In a local environment, the client can use this identifier or a PUT API route.
        For simplicity, we return a mock URL targeting a local upload endpoint.
        """
        # Returns a mock endpoint that FastAPI or test-client can hit
        return f"http://localhost:8000/api/v1/upload/local?key={key}"

    def download(self, key: str) -> bytes:
        """
        Download bytes from local simulated storage.
        """
        file_path = self._get_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Local storage key not found: {key}")

        return file_path.read_bytes()

    def upload_file(self, key: str, content: bytes) -> None:
        """
        Helper method to write bytes directly (used by mock clients or direct tests).
        """
        file_path = self._get_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    def delete(self, key: str) -> None:
        """
        Delete a file from local simulated storage.
        """
        file_path = self._get_path(key)
        if file_path.exists():
            file_path.unlink()

    def exists(self, key: str) -> bool:
        """
        Check if a file exists locally.
        """
        try:
            return self._get_path(key).exists()
        except ValueError:
            return False
