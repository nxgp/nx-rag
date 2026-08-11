"""
Plain Text & Markdown Document Parser — Mentera RAG Pipeline.

TextParser handles both plain text (.txt) and Markdown (.md) files.
The entire file content is returned as a single ParsedPage (no pagination).

For Markdown, formatting syntax can optionally be stripped so downstream
embedding models see clean prose rather than raw Markdown syntax.
"""

import logging
import re
from pathlib import Path
from typing import Any

from mentera_rag.parsing.base import BaseDocumentParser, ParsedPage

logger = logging.getLogger(__name__)

# Regex patterns for stripping common Markdown syntax
_MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BOLD_ITALIC = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^\)]+\)")
_MD_HORIZONTAL_RULE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r"^>\s+", re.MULTILINE)
_MD_HTML_TAG = re.compile(r"<[^>]+>")


def _strip_markdown(text: str) -> str:
    """
    Remove Markdown formatting syntax from text, preserving readable prose.

    Strips: fenced code blocks, inline code, headings, bold/italic,
    hyperlinks (keeps label), images (keeps alt text), blockquotes,
    horizontal rules, and HTML tags.
    """
    text = _MD_CODE_BLOCK.sub("", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD_ITALIC.sub(r"\1", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _MD_HORIZONTAL_RULE.sub("", text)
    text = _MD_HTML_TAG.sub("", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TextParser(BaseDocumentParser):
    """
    Parser for plain text (.txt) and Markdown (.md) files.

    Returns the entire file as a single ParsedPage (page_number=None since
    text files have no pagination concept).

    Args:
        strip_markdown: If True, strips Markdown formatting from .md files.
                        Has no effect on .txt files. Defaults to True.
        encoding: File encoding. Defaults to 'utf-8'.
    """

    def __init__(self, strip_markdown: bool = True, encoding: str = "utf-8") -> None:
        self.strip_markdown = strip_markdown
        self.encoding = encoding

    def parse(self, file_path: Path, metadata: dict[str, Any] | None = None) -> list[ParsedPage]:
        """
        Read a text or Markdown file and return its content as a single ParsedPage.

        Args:
            file_path: Path to the .txt or .md file.
            metadata: Optional caller metadata merged into the page's metadata.

        Returns:
            A list containing a single ParsedPage. Returns empty list if the
            file is empty after stripping whitespace.
        """
        self._validate_file(file_path)
        extra_meta = metadata or {}

        try:
            text = file_path.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            logger.warning(
                "Failed to decode '%s' as %s — retrying with latin-1.",
                file_path.name,
                self.encoding,
            )
            text = file_path.read_text(encoding="latin-1")

        is_markdown = file_path.suffix.lower() == ".md"

        if is_markdown and self.strip_markdown:
            text = _strip_markdown(text)

        text = text.strip()
        if not text:
            logger.warning("Text file '%s' is empty after parsing.", file_path.name)
            return []

        page_meta: dict[str, Any] = {
            "file_extension": file_path.suffix.lower(),
            "is_markdown": is_markdown,
            "markdown_stripped": is_markdown and self.strip_markdown,
            "char_count": len(text),
            **extra_meta,
        }

        return [
            ParsedPage(
                content=text,
                page_number=None,  # Text files have no pagination
                metadata=page_meta,
            )
        ]
