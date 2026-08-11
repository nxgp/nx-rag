"""
Parser Factory — Mentera RAG Pipeline.

Maps file extensions to their concrete BaseDocumentParser implementations.
Central dispatch point — all parsing calls should go through ParserFactory.get_parser().

Supported formats:
  .pdf              → PDFParser (PyMuPDF)
  .txt              → TextParser
  .md, .markdown    → TextParser (with Markdown stripping)
  .png, .jpg, .jpeg,
  .tiff, .tif       → ImageParser (Tesseract OCR)
"""

from mentera_rag.parsing.base import BaseDocumentParser
from mentera_rag.parsing.image_parser import SUPPORTED_IMAGE_EXTENSIONS, ImageParser
from mentera_rag.parsing.pdf_parser import PDFParser
from mentera_rag.parsing.text_parser import TextParser

# Mapping from lowercase file extension → parser class
_EXTENSION_MAP: dict[str, type[BaseDocumentParser]] = {
    ".pdf": PDFParser,
    ".txt": TextParser,
    ".md": TextParser,
    ".markdown": TextParser,
    **dict.fromkeys(SUPPORTED_IMAGE_EXTENSIONS, ImageParser),
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXTENSION_MAP.keys())


class ParserFactory:
    """
    Factory for instantiating document parsers by file extension.
    """

    @staticmethod
    def get_parser(file_extension: str) -> BaseDocumentParser:
        """
        Return the appropriate parser for the given file extension.

        Args:
            file_extension: Lowercase file extension including the dot (e.g. '.pdf', '.md').

        Returns:
            An instantiated BaseDocumentParser ready to call .parse().

        Raises:
            ValueError: If the file extension is not supported.

        Examples:
            >>> parser = ParserFactory.get_parser(".pdf")
            >>> pages = parser.parse(Path("report.pdf"))

            >>> parser = ParserFactory.get_parser(".png")
            >>> pages = parser.parse(Path("scan.png"))
        """
        ext = file_extension.lower().strip()
        parser_cls = _EXTENSION_MAP.get(ext)

        if parser_cls is None:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        return parser_cls()

    @staticmethod
    def is_supported(file_extension: str) -> bool:
        """Return True if the file extension has a registered parser."""
        return file_extension.lower().strip() in _EXTENSION_MAP
