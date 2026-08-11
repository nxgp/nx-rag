from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mentera_rag.parsing.factory import ParserFactory
from mentera_rag.parsing.image_parser import ImageParser
from mentera_rag.parsing.pdf_parser import PDFParser
from mentera_rag.parsing.text_parser import TextParser


@pytest.fixture
def temp_file(tmp_path) -> Path:
    file = tmp_path / "test.txt"
    file.write_text("Hello World! This is a test file.")
    return file


@pytest.mark.unit
def test_text_parser_basic(temp_file):
    parser = TextParser(strip_markdown=False)
    pages = parser.parse(temp_file, metadata={"test_key": "val"})

    assert len(pages) == 1
    assert pages[0].content == "Hello World! This is a test file."
    assert pages[0].page_number is None
    assert pages[0].metadata["file_extension"] == ".txt"
    assert pages[0].metadata["test_key"] == "val"
    assert not pages[0].metadata["is_markdown"]


@pytest.mark.unit
def test_text_parser_markdown(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Heading\n\nSome **bold text** and `code` here.")

    parser = TextParser(strip_markdown=True)
    pages = parser.parse(md_file)

    assert len(pages) == 1
    assert "Heading" in pages[0].content
    assert "bold text" in pages[0].content
    assert "code" in pages[0].content
    assert "**" not in pages[0].content
    assert "`" not in pages[0].content


@pytest.mark.unit
def test_pdf_parser_mocked(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("Dummy PDF content")  # fitz won't actually read this, we mock it

    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 content here."
    mock_page1.rect.width = 500
    mock_page1.rect.height = 700

    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Short"  # Will be skipped because < 10 chars by default
    mock_page2.rect.width = 500
    mock_page2.rect.height = 700

    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]

    with patch("fitz.open", return_value=mock_doc) as mock_open:
        parser = PDFParser(min_page_chars=10)
        pages = parser.parse(pdf_file)

        mock_open.assert_called_once_with(str(pdf_file))
        assert len(pages) == 1  # Page 2 skipped because < 10 chars
        assert pages[0].content == "Page 1 content here."
        assert pages[0].page_number == 1
        assert pages[0].metadata["page_number"] == 1
        assert pages[0].metadata["total_pages"] == 2


@pytest.mark.unit
def test_image_parser_mocked(tmp_path):
    img_file = tmp_path / "test.png"
    img_file.write_text("Dummy Image content")

    mock_image = MagicMock()
    mock_image.size = (800, 600)

    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = " Extracted text from OCR. "

    with (
        patch("PIL.Image.open", return_value=mock_image),
        patch.dict("sys.modules", {"pytesseract": mock_pytesseract}),
    ):
        parser = ImageParser(min_content_chars=5)
        pages = parser.parse(img_file)

        mock_pytesseract.image_to_string.assert_called_once()
        assert len(pages) == 1
        assert pages[0].content == "Extracted text from OCR."
        assert pages[0].page_number is None
        assert pages[0].metadata["file_extension"] == ".png"
        assert pages[0].metadata["image_width"] == 800


@pytest.mark.unit
def test_parser_factory():
    assert isinstance(ParserFactory.get_parser(".pdf"), PDFParser)
    assert isinstance(ParserFactory.get_parser(".txt"), TextParser)
    assert isinstance(ParserFactory.get_parser(".md"), TextParser)
    assert isinstance(ParserFactory.get_parser(".png"), ImageParser)
    assert isinstance(ParserFactory.get_parser(".jpg"), ImageParser)

    with pytest.raises(ValueError):
        ParserFactory.get_parser(".docx")  # docx is phase 2
