"""
Utility functions for text preprocessing, cleaning, and token estimation.
"""
import re
import logging
from typing import List, Text, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessingError(Exception):
    """Custom exception for text processing errors."""
    pass

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
        
    Raises:
        TextProcessingError: If text is None or not a string
    """
    try:
        if not isinstance(text, str):
            raise TextProcessingError("Input must be a string")
        if not text.strip():
            logger.warning("Empty text provided for cleaning")
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise TextProcessingError(f"Failed to clean text: {str(e)}")

def remove_headers_footers(text: str) -> str:
    """
    Remove common header/footer patterns from text.
    
    Args:
        text (str): Input text to process
        
    Returns:
        str: Text with headers and footers removed
        
    Raises:
        TextProcessingError: If text is None or invalid
    """
    try:
        if not isinstance(text, str):
            raise TextProcessingError("Input must be a string")
        if not text.strip():
            logger.warning("Empty text provided for header/footer removal")
            return ""
            
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            try:
                # Skip page numbers
                if re.match(r'^\d+$', line.strip()):
                    continue
                # Skip common headers/footers
                if any(pattern in line.lower() for pattern in ['confidential', 'all rights reserved', 'page']):
                    continue
                cleaned_lines.append(line)
            except Exception as e:
                logger.warning(f"Error processing line: {str(e)}")
                continue
        
        result = '\n'.join(cleaned_lines)
        if not result.strip():
            logger.warning("All lines were filtered out during header/footer removal")
        return result
    except Exception as e:
        logger.error(f"Error removing headers/footers: {str(e)}")
        raise TextProcessingError(f"Failed to remove headers/footers: {str(e)}")

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text (rough approximation)."""
    # Approximate token count based on words and punctuation
    words = len(text.split())
    # Add 30% overhead for tokenization
    return int(words * 1.3)

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using basic rules."""
    # Basic sentence splitting on punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]