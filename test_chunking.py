"""
Test cases for text chunking functionality.
"""
import pytest
from ingest import chunk_text
from utils import estimate_tokens

def test_chunk_size():
    # Test that chunks don't exceed max token size
    text = "This is a test " * 100  # Create long text
    chunks = chunk_text(text)
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 512  # Assuming 512 is max chunk size

def test_chunk_overlap():
    # Test chunk overlap functionality
    pass

def test_chunk_empty():
    # Test handling of empty text
    assert chunk_text("") == []

def test_chunk_single():
    # Test handling of text smaller than chunk size
    short_text = "This is a short text."
    chunks = chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text