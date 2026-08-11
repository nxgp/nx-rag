"""
Abstract Base Interface for Document Parsers — Mentera RAG Pipeline.

All format-specific parsers (PDF, TXT, MD, Image) implement BaseDocumentParser.
The parse() method returns a list of ParsedPage objects — one per logical page
(or a single page for formats like TXT/MD/images that have no pagination).

ParsedPage carries the extracted text and a page_number so the chunker can
embed page provenance into every Chunk it produces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedPage:
    """
    A single logical page of extracted text from a parsed document.

    Attributes:
        content: Extracted text content of the page (may be empty for blank pages).
        page_number: 1-based page number within the document.
                     None for single-page formats (TXT, MD, single images).
        metadata: Page-level metadata (e.g. {'width': 595, 'height': 842} for PDFs).
    """

    content: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDocumentParser(ABC):
    """
    Abstract Base Class for all document format parsers.

    Each concrete parser handles one or more related file formats and is
    responsible for extracting clean text content with page-level granularity.
    """

    @abstractmethod
    def parse(self, file_path: Path, metadata: dict[str, Any] | None = None) -> list[ParsedPage]:
        """
        Extract text content from a file, returning per-page results.

        Args:
            file_path: Absolute path to the file to parse.
            metadata: Optional caller-provided metadata to merge into each page
                      (e.g. {'tenant_id': '...', 'source': '...'}).

        Returns:
            Ordered list of ParsedPage objects. At minimum one page is returned
            for non-empty files. Empty files return an empty list.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If the file format is unsupported or unreadable.
        """
        pass

    @staticmethod
    def _validate_file(file_path: Path) -> None:
        """Raise FileNotFoundError if the file does not exist."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
