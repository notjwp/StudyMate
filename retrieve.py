# retrieve.py
import logging
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Optional, Union
from embed_index import load_faiss_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class RetrieverError(Exception):
    """Base exception for retriever errors."""
    pass

class ConfigurationError(RetrieverError):
    """Raised when there are configuration issues."""
    pass

class EmbeddingError(RetrieverError):
    """Raised when there are embedding-related errors."""
    pass

class SearchError(RetrieverError):
    """Raised when search operations fail."""
    pass

@dataclass
class RetrieverConfig:
    """Configuration for retriever."""
    index_path: str = "indexes/corpus.index"
    metadata_path: str = "indexes/corpus_metadata.json"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not self.index_path:
            raise ConfigurationError("index_path cannot be empty")
            
        if not self.metadata_path:
            raise ConfigurationError("metadata_path cannot be empty")
            
        if not self.model_name:
            raise ConfigurationError("model_name cannot be empty")
            
        # Check file existence
        if not os.path.exists(self.index_path):
            raise ConfigurationError(f"FAISS index not found: {self.index_path}")
            
        if not os.path.exists(self.metadata_path):
            raise ConfigurationError(f"Metadata file not found: {self.metadata_path}")

@dataclass
class SearchResult:
    """Represents a single search result."""
    score: float
    chunk_id: int
    doc_id: str
    page_num: int
    text: str


# -------------------------------------------------------------------
# Helper to load metadata
# -------------------------------------------------------------------
def load_metadata(path: str) -> List[Dict]:
    """
    Load chunk metadata from JSON file.
    
    Args:
        path (str): Path to metadata file
        
    Returns:
        List[Dict]: List of chunk metadata
        
    Raises:
        ConfigurationError: If metadata cannot be loaded
    """
    try:
        if not os.path.exists(path):
            raise ConfigurationError(f"Metadata file not found: {path}")
            
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid metadata JSON: {str(e)}")
                
        if not isinstance(data, list):
            raise ConfigurationError("Metadata must be a list")
            
        # Validate metadata entries
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ConfigurationError(f"Invalid metadata entry {i}")
                
            required = ['chunk_id', 'doc_id', 'page_num', 'text']
            missing = [f for f in required if f not in entry]
            if missing:
                raise ConfigurationError(
                    f"Metadata entry {i} missing fields: {missing}"
                )
                
        logger.info(f"Loaded {len(data)} metadata entries")
        return data
        
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to load metadata: {str(e)}")


# -------------------------------------------------------------------
# Retriever class
# -------------------------------------------------------------------
class Retriever:
    """
    Lightweight retrieval class using FAISS + SentenceTransformer embeddings.

    Steps:
    1. Embed query using same model used for corpus.
    2. Search FAISS index for top-k similar chunks.
    3. Return list of chunks with metadata (doc_id, page_num, text, score).
    """

    def __init__(self, config: Optional[RetrieverConfig] = None):
        """
        Initialize retriever with configuration.
        
        Args:
            config (Optional[RetrieverConfig]): Configuration object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self.config = config or RetrieverConfig()
            self.config.validate()
            
            # Load FAISS index
            try:
                self.index = load_faiss_index(self.config.index_path)
            except Exception as e:
                raise ConfigurationError(f"Failed to load FAISS index: {str(e)}")
                
            # Load metadata
            self.metadata = load_metadata(self.config.metadata_path)
            
            # Initialize embedding model
            try:
                self.model = SentenceTransformer(self.config.model_name)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load embedding model: {str(e)}"
                )
                
            logger.info(
                f"Initialized retriever with {len(self.metadata)} chunks"
            )
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to initialize retriever: {str(e)}")

    # ---------------------------------------------------------------
    # Query Embedding
    # ---------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode and normalize query for similarity search.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Normalized query embedding
            
        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            if not isinstance(query, str):
                raise EmbeddingError("Query must be a string")
                
            if not query.strip():
                raise EmbeddingError("Query cannot be empty")
                
            try:
                v = self.model.encode([query], convert_to_numpy=True)
            except Exception as e:
                raise EmbeddingError(f"Failed to encode query: {str(e)}")
                
            try:
                # L2 normalize
                norm = np.linalg.norm(v, axis=1, keepdims=True)
                v = v / np.maximum(norm, 1e-10)
                return v.astype("float32")
            except Exception as e:
                raise EmbeddingError(f"Failed to normalize query vector: {str(e)}")
                
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Query embedding failed: {str(e)}")

    # ---------------------------------------------------------------
    # Retrieve top-k chunks
    # ---------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for top-k most relevant chunks.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[SearchResult]: Ranked search results
            
        Raises:
            SearchError: If search fails
        """
        try:
            if top_k < 1:
                raise SearchError("top_k must be positive")
                
            # Get query embedding
            try:
                qv = self.embed_query(query)
            except EmbeddingError as e:
                raise SearchError(f"Query embedding failed: {str(e)}")
                
            # Search FAISS index
            try:
                D, I = self.index.search(qv, top_k)
            except Exception as e:
                raise SearchError(f"FAISS search failed: {str(e)}")
                
            results = []
            for score, idx in zip(D[0], I[0]):
                try:
                    # Skip invalid indices
                    if idx < 0 or idx >= len(self.metadata):
                        logger.warning(f"Invalid index {idx}, skipping")
                        continue
                        
                    meta = self.metadata[idx]
                    
                    # Validate required fields
                    missing = []
                    for field in ['chunk_id', 'doc_id', 'page_num', 'text']:
                        if field not in meta:
                            missing.append(field)
                    if missing:
                        logger.warning(
                            f"Metadata at index {idx} missing fields: {missing}"
                        )
                        continue
                        
                    result = SearchResult(
                        score=float(score),
                        chunk_id=meta["chunk_id"],
                        doc_id=meta["doc_id"],
                        page_num=meta["page_num"],
                        text=meta["text"]
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error processing result {idx}: {str(e)}")
                    continue
                    
            if not results:
                logger.warning("No valid results found")
                
            return results
            
        except Exception as e:
            if isinstance(e, SearchError):
                raise
            raise SearchError(f"Search failed: {str(e)}")


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------
if __name__ == "__main__":
    def main():
        logger.info("🔎 Testing Retriever module...")
        
        try:
            # Initialize with default configuration
            config = RetrieverConfig()
            
            try:
                retriever = Retriever(config)
            except ConfigurationError as e:
                logger.error(f"Failed to initialize retriever: {str(e)}")
                return
                
            # Test search
            query = "What is the definition of mitosis?"
            logger.info(f"Searching for: {query}")
            
            try:
                results = retriever.retrieve(query, top_k=3)
            except SearchError as e:
                logger.error(f"Search failed: {str(e)}")
                return
                
            if not results:
                logger.warning("No results found")
                return
                
            # Display results
            for r in results:
                print(
                    f"[{r.doc_id} | page {r.page_num}] "
                    f"(score={r.score:.3f})"
                )
                # Truncate long text snippets
                preview = r.text[:200]
                if len(r.text) > 200:
                    preview += "..."
                print(f"{preview}\n")
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return

    main()