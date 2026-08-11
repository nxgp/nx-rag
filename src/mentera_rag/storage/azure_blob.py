"""
Azure Blob Object Storage Backend — Mentera RAG Pipeline.

Implements the BaseObjectStore interface using azure-storage-blob
to upload, download, delete, and check blobs in Azure Storage containers.
Uses lazy imports to prevent dependency failures on systems without the Azure SDK.
"""

import logging
from datetime import datetime, timedelta, timezone
from mentera_rag.storage.base import BaseObjectStore

logger = logging.getLogger(__name__)


class AzureBlobStorageStore(BaseObjectStore):
    """
    Object storage backend backed by Azure Blob Storage.
    """

    def __init__(self, connection_string: str, container_name: str = "mentera-uploads"):
        """
        Initialize the Azure Blob Storage backend.

        Args:
            connection_string: Connection string containing account details and credentials.
            container_name: Name of the Azure Blob container.
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self._service_client = None

    @property
    def service_client(self):
        """Lazy-initialize BlobServiceClient."""
        if self._service_client is None:
            try:
                from azure.storage.blob import BlobServiceClient

                self._service_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            except ImportError as e:
                raise ImportError(
                    "azure-storage-blob is required to use Azure Blob storage. "
                    "Install with: pip install 'mentera-rag[storage]'"
                ) from e
        return self._service_client

    def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a SAS token URI allowing direct PUT upload to Azure Blob Storage.
        """
        try:
            from azure.storage.blob import BlobSasPermissions, generate_blob_sas

            blob_client = self.service_client.get_blob_client(
                container=self.container_name, blob=key
            )
            account_name = self.service_client.account_name
            # Retrieve account key from connection string credentials
            account_key = self.service_client.credential.account_key

            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=key,
                account_key=account_key,
                permission=BlobSasPermissions(write=True),
                expiry=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            )
            return f"{blob_client.url}?{sas_token}"
        except Exception as e:
            logger.error("Failed to generate Azure Blob SAS URL: %s", e)
            raise RuntimeError("Failed to generate presigned upload URL.") from e

    def download(self, key: str) -> bytes:
        """
        Download blob bytes from Azure.
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, blob=key
            )
            download_stream = blob_client.download_blob()
            return bytes(download_stream.readall())
        except Exception as e:
            # Check for Azure ResourceNotFoundError
            if "ResourceNotFound" in str(e) or "404" in str(e):
                raise FileNotFoundError(f"Azure Blob key not found: {key}") from e
            logger.error("Failed to download blob from Azure: %s", e)
            raise RuntimeError(f"Failed to download blob {key} from Azure.") from e

    def delete(self, key: str) -> None:
        """
        Delete a blob from Azure.
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, blob=key
            )
            blob_client.delete_blob()
        except Exception as e:
            logger.error("Failed to delete blob from Azure: %s", e)
            raise RuntimeError(f"Failed to delete blob {key} from Azure.") from e

    def exists(self, key: str) -> bool:
        """
        Check if a blob exists in Azure container.
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, blob=key
            )
            return bool(blob_client.exists())
        except Exception as e:
            logger.warning("Error checking Azure Blob existence: %s", e)
            return False
