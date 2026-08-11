"""
Object Storage Backend Module — Mentera RAG Pipeline.

Exposes the BaseObjectStore interface and StorageFactory for instantiating
storage adapters (Local, S3, Azure Blob, or GCS) based on settings.
"""

from mentera_rag.storage.base import BaseObjectStore
from mentera_rag.storage.factory import StorageFactory

__all__ = ["BaseObjectStore", "StorageFactory"]
