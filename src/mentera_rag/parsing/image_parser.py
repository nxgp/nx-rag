"""
Image Document Parser — Mentera RAG Pipeline.

Uses Tesseract OCR (via pytesseract + Pillow) to extract text from image files.
Supports PNG, JPG, JPEG, and TIFF formats.

Tesseract must be installed system-wide:
  Ubuntu/Debian: sudo apt-get install tesseract-ocr
  macOS:         brew install tesseract
  Windows:       https://github.com/UB-Mannheim/tesseract/wiki

Python dependency: pip install 'mentera-rag[ocr]'
  (installs pytesseract + pillow)
"""

import logging
from pathlib import Path
from typing import Any

from mentera_rag.parsing.base import BaseDocumentParser, ParsedPage

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


class ImageParser(BaseDocumentParser):
    """
    OCR-based text extractor for image files using Tesseract.

    Converts image files to text via pytesseract. Returns a single ParsedPage
    since images have no pagination concept. Low-confidence or empty OCR results
    are logged as warnings and return an empty list.

    Args:
        lang: Tesseract language code(s). Defaults to 'eng'.
               Use '+' to combine: 'eng+fra' for English + French.
        config: Additional Tesseract config string (e.g. '--psm 6').
        min_content_chars: Minimum extracted character count to be considered valid.
    """

    def __init__(
        self,
        lang: str = "eng",
        config: str = "",
        min_content_chars: int = 10,
    ) -> None:
        self.lang = lang
        self.config = config
        self.min_content_chars = min_content_chars

    def parse(self, file_path: Path, metadata: dict[str, Any] | None = None) -> list[ParsedPage]:
        """
        Run Tesseract OCR on an image file and return extracted text.

        Args:
            file_path: Path to the image file (.png, .jpg, .jpeg, .tiff).
            metadata: Optional caller metadata merged into the page's metadata.

        Returns:
            A list with a single ParsedPage containing OCR text.
            Returns empty list if OCR produces no usable content.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If the file extension is not a supported image format.
            ImportError: If pytesseract or Pillow are not installed.
            RuntimeError: If Tesseract binary is not found on PATH.
        """
        self._validate_file(file_path)

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{ext}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

        extra_meta = metadata or {}

        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "pytesseract and Pillow are required for image parsing. "
                "Install with: pip install 'mentera-rag[ocr]'"
            ) from e

        try:
            image = Image.open(str(file_path))
        except Exception as exc:
            raise ValueError(f"Failed to open image file: {file_path}") from exc

        try:
            text: str = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract binary not found. Install Tesseract:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "  macOS:         brew install tesseract"
            ) from exc

        text = text.strip()

        if len(text) < self.min_content_chars:
            logger.warning(
                "OCR produced < %d chars from image '%s' — skipping. "
                "Check image quality or Tesseract language settings.",
                self.min_content_chars,
                file_path.name,
            )
            return []

        # Get image dimensions for metadata
        width, height = image.size
        page_meta: dict[str, Any] = {
            "file_extension": ext,
            "image_width": width,
            "image_height": height,
            "ocr_lang": self.lang,
            "char_count": len(text),
            **extra_meta,
        }

        return [
            ParsedPage(
                content=text,
                page_number=None,  # Single-page images have no pagination
                metadata=page_meta,
            )
        ]
