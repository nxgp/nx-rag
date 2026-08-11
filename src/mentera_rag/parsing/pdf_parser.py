"""
PDF Document Parser — Mentera RAG Pipeline.

Uses PyMuPDF (fitz) for fast, high-quality text extraction from text-based PDFs.
Each PDF page becomes a separate ParsedPage with its 1-based page_number preserved.

Scanned PDFs (image-only, no embedded text layer) produce empty page content.
For scanned documents, use ImageParser after rendering pages to images — this is
handled automatically in UploadPipeline via fallback detection.
"""

import logging
from pathlib import Path
from typing import Any

from mentera_rag.parsing.base import BaseDocumentParser, ParsedPage

logger = logging.getLogger(__name__)


class PDFParser(BaseDocumentParser):
    """
    PDF text extractor using PyMuPDF (fitz).

    Extracts text page-by-page. Pages with no extractable text (e.g. scanned
    images without a text layer) are skipped with a warning logged.

    Args:
        min_page_chars: Minimum character count for a page to be included.
                        Pages below this threshold are treated as blank/image-only.
    """

    def __init__(self, min_page_chars: int = 10) -> None:
        self.min_page_chars = min_page_chars

    def parse(self, file_path: Path, metadata: dict[str, Any] | None = None) -> list[ParsedPage]:
        """
        Extract text from a PDF, one ParsedPage per PDF page.

        Args:
            file_path: Path to the .pdf file.
            metadata: Optional caller metadata merged into each page's metadata.

        Returns:
            List of ParsedPage objects, one per page with extractable text.
            Pages below min_page_chars are skipped (logged as warnings).
        """
        self._validate_file(file_path)
        extra_meta = metadata or {}
        pages: list[ParsedPage] = []

        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. Install with: pip install pymupdf"
            ) from e

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise ValueError(f"Failed to open PDF: {file_path}") from exc

        total_pages = len(doc)
        skipped = 0

        for page_index in range(total_pages):
            page = doc[page_index]
            text: str = page.get_text("text")  # type: ignore[attr-defined]
            text = text.strip()

            if len(text) < self.min_page_chars:
                logger.warning(
                    "PDF page %d of '%s' has < %d chars — skipping (may be scanned image).",
                    page_index + 1,
                    file_path.name,
                    self.min_page_chars,
                )
                skipped += 1
                continue

            page_meta: dict[str, Any] = {
                "page_number": page_index + 1,
                "total_pages": total_pages,
                "width": page.rect.width,
                "height": page.rect.height,
                **extra_meta,
            }

            pages.append(
                ParsedPage(
                    content=text,
                    page_number=page_index + 1,
                    metadata=page_meta,
                )
            )

        doc.close()

        if skipped > 0:
            logger.info(
                "PDF '%s': extracted %d pages, skipped %d low-content pages.",
                file_path.name,
                len(pages),
                skipped,
            )

        return pages
