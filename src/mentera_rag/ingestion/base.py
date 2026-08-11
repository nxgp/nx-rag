"""
Abstract Base Class for Dataset Loaders.
"""

from abc import ABC, abstractmethod
from typing import Any

from mentera_rag.ingestion.schemas import Document, Qrel, Query


class BaseDatasetLoader(ABC):
    """
    Abstract interface for downloading, parsing, and normalizing medical datasets.
    """

    @abstractmethod
    def fetch_raw(self) -> Any:
        """Download or read raw dataset records from HuggingFace or disk."""
        pass

    @abstractmethod
    def process(self) -> tuple[list[Document], list[Query], list[Qrel]]:
        """
        Process raw records into normalized Documents, Queries, and Qrels.

        Returns:
            Tuple of (documents, queries, qrels)
        """
        pass
