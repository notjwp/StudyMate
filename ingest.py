"""
PDF ingestion and preprocessing module with robust error handling.
"""
import fitz  # PyMuPDF
import re
import json
import os
import logging
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass

class PreprocessingError(Exception):
    """Custom exception for text preprocessing errors."""
    pass

class ChunkingError(Exception):
    """Custom exception for text chunking errors."""
    pass

# Configurable defaults
MAX_PAGES = None  # None means all pages
HEADER_FOOTER_THRESHOLD = 0.6  # ngram frequency threshold to treat as header/footer
CHUNK_WORDS = 400             # ~ target chunk size in words (approx 400 words ~ 500 tokens)
OVERLAP_WORDS = 50            # overlap in words between chunks

def extract_pages(file_path: str, max_pages=None) -> List[Dict]:
    """Extract page-level text and metadata using PyMuPDF (fitz). Returns list of dicts {page_num, text}."""
    doc = fitz.open(file_path)
    pages = []
    for i in range(len(doc)):
        if max_pages is not None and i >= max_pages:
            break
        page = doc.load_page(i)
        text = page.get_text("text")  # simple text extraction
        pages.append({"page_num": i+1, "text": text})
    doc.close()
    return pages

def normalize_text(s: str) -> str:
    # Normalize unicode, collapse whitespace, remove weird control chars
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\r\n', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = s.strip()
    return s

def detect_headers_footers(pages: List[Dict], ngram_size=5) -> Tuple[set, set]:
    """Simple heuristic: find common starting ngrams (headers) and ending ngrams (footers) across pages."""
    starts = []
    ends = []
    for p in pages:
        lines = [ln.strip() for ln in p["text"].splitlines() if ln.strip()]
        if not lines:
            continue
        # use first/last line up to ngram_size words
        start_words = " ".join(lines[0].split()[:ngram_size]).lower()
        end_words = " ".join(lines[-1].split()[-ngram_size:]).lower()
        starts.append(start_words)
        ends.append(end_words)
    start_counts = Counter(starts)
    end_counts = Counter(ends)
    n_pages = max(1, len(pages))
    headers = {s for s,c in start_counts.items() if c / n_pages >= HEADER_FOOTER_THRESHOLD}
    footers = {s for s,c in end_counts.items() if c / n_pages >= HEADER_FOOTER_THRESHOLD}
    return headers, footers

def remove_headers_footers(text: str, headers:set, footers:set, ngram_size=5) -> str:
    lines = [ln for ln in text.splitlines()]
    if not lines:
        return text
    # strip headers
    first = " ".join(lines[0].split()[:ngram_size]).lower()
    if first in headers:
        lines = lines[1:]
    # strip footers
    if lines:
        last = " ".join(lines[-1].split()[-ngram_size:]).lower()
        if last in footers:
            lines = lines[:-1]
    return "\n".join(lines)

def chunk_text_by_words(doc_id: str, pages: List[Dict], chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS) -> List[Dict]:
    """
    Convert page-level text to overlapping chunks.
    Each chunk metadata: {chunk_id, doc_id, page_num, start_char, end_char, text}
    We chunk at page granularity and across page breaks as a simple strategy: concatenate pages with page markers.
    """
    # Build a full document text with page markers to preserve page numbers for offsets
    doc_text = []
    char_offset = 0
    page_start_offsets = []
    for p in pages:
        page_marker = f"\n\n[[PAGE {p['page_num']}]]\n\n"
        page_start_offsets.append((p['page_num'], char_offset + len(page_marker)))
        doc_text.append(page_marker)
        doc_text.append(p['text'])
        char_offset += len(page_marker) + len(p['text'])
    full_text = normalize_text("".join(doc_text))
    # split into words
    words = full_text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        start = max(0, i - overlap_words) if chunk_id > 0 else i
        end = min(len(words), i + chunk_words)
        chunk_words_list = words[start:end]
        chunk_text = " ".join(chunk_words_list)
        # estimate char offsets by searching chunk_text in full_text from last seen pos
        # (simple method; for production, preserve offsets during construction)
        start_char = full_text.find(chunk_text)
        end_char = start_char + len(chunk_text) if start_char >= 0 else None
        # find nearest page_num using page_start_offsets
        page_num = None
        for pn, pstart in reversed(page_start_offsets):
            if start_char is not None and start_char >= pstart:
                page_num = pn
                break
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "page_num": page_num or 1,
            "start_char": start_char,
            "end_char": end_char,
            "text": chunk_text
        })
        chunk_id += 1
        i += chunk_words - overlap_words
    return chunks

# Example usage
if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    pages = extract_pages(fp, max_pages=MAX_PAGES)
    headers, footers = detect_headers_footers(pages)
    # Remove headers/footers per page
    for p in pages:
        p["text"] = normalize_text(remove_headers_footers(p["text"], headers, footers))
    chunks = chunk_text_by_words("sample_doc", pages)
    # persist
    os.makedirs("indexes", exist_ok=True)
    with open("indexes/sample_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, indent=2, ensure_ascii=False)
    print(f"Extracted {len(pages)} pages -> {len(chunks)} chunks")