"""
Storage Backend Factory — Mentera RAG Pipeline.

Instantiates and returns the configured object storage backend adapter
based on settings.py settings.
"""

import os

from mentera_rag.config.settings import settings
from mentera_rag.storage.azure_blob import AzureBlobStorageStore
from mentera_rag.storage.base import BaseObjectStore
from mentera_rag.storage.gcs import GCSStorageStore
from mentera_rag.storage.local import LocalStorageStore
from mentera_rag.storage.s3 import S3StorageStore


class StorageFactory:
    """
    Factory for producing BaseObjectStore instances.
    """

    @staticmethod
    def get_store(provider_type: str | None = None) -> BaseObjectStore:
        """
        Return the configured object store instance.

        Args:
            provider_type: 'local', 's3', 'azure_blob', or 'gcs'.
                           Defaults to settings.STORAGE_PROVIDER.

        Returns:
            An instantiated BaseObjectStore subclass.
        """
        p_type = (provider_type or settings.STORAGE_PROVIDER).lower()

        if p_type == "local":
            # Resolves storage root dir from env, settings, or default temp path
            storage_root = os.getenv("LOCAL_STORAGE_DIR", "/tmp/mentera_storage")  # nosec B108
            return LocalStorageStore(root_dir=storage_root)

        elif p_type == "s3":
            bucket = settings.STORAGE_BUCKET
            region = settings.AWS_REGION
            return S3StorageStore(bucket_name=bucket, region_name=region)

        elif p_type == "azure_blob":
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
            if not connection_string:
                raise ValueError(
                    "AZURE_STORAGE_CONNECTION_STRING environment variable must be set "
                    "to use the azure_blob storage provider."
                )
            bucket = settings.STORAGE_BUCKET
            return AzureBlobStorageStore(connection_string=connection_string, container_name=bucket)

        elif p_type == "gcs":
            bucket = settings.STORAGE_BUCKET
            credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            return GCSStorageStore(bucket_name=bucket, credentials_path=credentials)

        else:
            raise ValueError(
                f"Unsupported storage provider type: '{p_type}'. "
                "Choose one of: 'local', 's3', 'azure_blob', 'gcs'."
            )
