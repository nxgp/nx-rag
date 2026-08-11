"""
AWS S3 Object Storage Backend — Mentera RAG Pipeline.

Implements the BaseObjectStore interface using boto3 to interact
with Amazon S3 buckets. Supports presigned upload URLs (PUT method),
downloads, deletions, and existence checks.
"""

import logging
import boto3
from botocore.exceptions import ClientError
from mentera_rag.storage.base import BaseObjectStore

logger = logging.getLogger(__name__)


class S3StorageStore(BaseObjectStore):
    """
    Object storage backend backed by AWS S3.
    """

    def __init__(self, bucket_name: str, region_name: str = "us-east-1"):
        """
        Initialize the S3 Storage backend.

        Args:
            bucket_name: Name of the target S3 bucket.
            region_name: AWS region name.
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.client = boto3.client("s3", region_name=self.region_name)

    def generate_presigned_upload_url(
        self,
        key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned PUT URL for direct file upload to S3.
        """
        try:
            url = self.client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=expires_in,
            )
            return str(url)
        except ClientError as e:
            logger.error("Failed to generate S3 presigned URL: %s", e)
            raise RuntimeError("Failed to generate presigned upload URL.") from e

    def download(self, key: str) -> bytes:
        """
        Download object bytes from S3.
        """
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            return bytes(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"S3 key not found: {key}") from e
            logger.error("Failed to download object from S3: %s", e)
            raise RuntimeError(f"Failed to download key {key} from S3.") from e

    def delete(self, key: str) -> None:
        """
        Delete an object from S3.
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
        except ClientError as e:
            logger.error("Failed to delete object from S3: %s", e)
            raise RuntimeError(f"Failed to delete key {key} from S3.") from e

    def exists(self, key: str) -> bool:
        """
        Verify if key exists in S3 using head_object.
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return False
            logger.warning("Error checking S3 key existence: %s", e)
            return False
