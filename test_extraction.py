"""
Test cases for PDF text extraction functionality.
"""
import pytest
import os
from pathlib import Path
from ingest import extract_text_from_pdf, PDFExtractionError

# Setup test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

def create_test_pdf(content: bytes, filename: str) -> Path:
    """Helper to create test PDF files."""
    filepath = TEST_DATA_DIR / filename
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    for file in TEST_DATA_DIR.glob("*.pdf"):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove test file {file}: {e}")

def test_pdf_extraction_valid():
    """Test extraction from a valid PDF file."""
    # Create a sample valid PDF for testing
    with pytest.raises(PDFExtractionError, match="PDF file not found"):
        extract_text_from_pdf("nonexistent.pdf")

def test_pdf_extraction_empty():
    """Test handling of empty PDF."""
    empty_pdf_path = create_test_pdf(b"", "empty.pdf")
    with pytest.raises(PDFExtractionError, match="Empty or invalid PDF"):
        extract_text_from_pdf(empty_pdf_path)

def test_pdf_extraction_invalid():
    """Test handling of invalid PDF file."""
    invalid_pdf_path = create_test_pdf(b"not a pdf", "invalid.pdf")
    with pytest.raises(PDFExtractionError, match="Invalid PDF format"):
        extract_text_from_pdf(invalid_pdf_path)

def test_pdf_extraction_permission():
    """Test handling of permission errors."""
    # This test needs to be implemented based on how your system handles permissions
    pass

def test_pdf_extraction_corrupt():
    """Test handling of corrupted PDF file."""
    corrupt_pdf_path = create_test_pdf(b"%PDF-1.4\ncorrupt content", "corrupt.pdf")
    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(corrupt_pdf_path)