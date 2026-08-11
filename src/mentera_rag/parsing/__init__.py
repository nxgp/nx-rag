"""
Document Parsing Module — Mentera RAG Pipeline.

Provides format-specific parsers for extracting text from uploaded files:
  - PDF (PyMuPDF / fitz)
  - Plain text (.txt)
  - Markdown (.md)
  - Images (.png, .jpg, .jpeg, .tiff) via Tesseract OCR

Usage:
    from mentera_rag.parsing.factory import ParserFactory
    parser = ParserFactory.get_parser(".pdf")
    pages = parser.parse(Path("/path/to/file.pdf"), metadata={})
"""
