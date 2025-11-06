# embed_index.py
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from typing import List, Dict, Optional, Union
from pathlib import Path
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass

class IndexError(Exception):
    """Custom exception for FAISS index-related errors."""
    pass

# Constants
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
MAX_TEXT_LENGTH = 2048  # Maximum text length for embedding

# ----------------------------
# Embedding & Model utilities
# ----------------------------
def load_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """
    Load the sentence transformer model.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        SentenceTransformer: Loaded model
        
    Raises:
        EmbeddingError: If model loading fails
    """
    try:
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully")
        return model
    except Exception as e:
        raise EmbeddingError(f"Failed to load model {model_name}: {str(e)}")

def split_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> List[str]:
    """
    Split text into chunks that don't exceed max_length.
    
    Args:
        text (str): Text to split
        max_length (int): Maximum length per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in text.split():
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                # Word itself exceeds max_length, truncate it
                chunks.append(word[:max_length])
        else:
            current_chunk.append(word)
            current_length += word_len
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

def validate_texts(texts: List[str]) -> List[str]:
    """
    Validate and process input texts before embedding.
    
    Args:
        texts (List[str]): List of texts to validate
        
    Returns:
        List[str]: Processed texts, split if necessary
        
    Raises:
        EmbeddingError: If validation fails
    """
    if not isinstance(texts, list):
        raise EmbeddingError("Input must be a list of strings")
    
    if not texts:
        raise EmbeddingError("Empty text list provided")
        
    if not all(isinstance(t, str) for t in texts):
        raise EmbeddingError("All items must be strings")
        
    # Split long texts into smaller chunks
    processed_texts = []
    for text in texts:
        if len(text) > MAX_TEXT_LENGTH:
            processed_texts.extend(split_text(text))
        else:
            processed_texts.append(text)
            
    return processed_texts

def batch_encode_texts(model: SentenceTransformer, 
                      texts: List[str], 
                      batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Batch encode texts to produce embeddings.
    
    Args:
        model (SentenceTransformer): Loaded model
        texts (List[str]): Texts to embed
        batch_size (int): Batch size for processing
        
    Returns:
        np.ndarray: L2-normalized embeddings
        
    Raises:
        EmbeddingError: If embedding fails
    """
    try:
        processed_texts = validate_texts(texts)
        
        logger.info(f"Encoding {len(processed_texts)} texts with batch size {batch_size}")
        embeddings = model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        # L2 normalize embeddings
        try:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        except Exception as e:
            raise EmbeddingError(f"Failed to normalize embeddings: {str(e)}")
            
        logger.info(f"Successfully encoded texts to shape {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        raise EmbeddingError(f"Failed to encode texts: {str(e)}")

def save_embeddings(embeddings: np.ndarray, 
                   metadata: List[Dict], 
                   prefix: str = "sample",
                   output_dir: str = "indexes") -> Dict[str, str]:
    """
    Save embeddings and metadata to disk.
    
    Args:
        embeddings (np.ndarray): Embeddings array
        metadata (List[Dict]): List of metadata dicts
        prefix (str): Prefix for output files
        output_dir (str): Output directory
        
    Returns:
        Dict[str, str]: Paths to saved files
        
    Raises:
        EmbeddingError: If saving fails
    """
    try:
        if not isinstance(embeddings, np.ndarray):
            raise EmbeddingError("embeddings must be numpy array")
            
        if not isinstance(metadata, list):
            raise EmbeddingError("metadata must be list of dicts")
            
        if len(metadata) != embeddings.shape[0]:
            raise EmbeddingError(
                f"Metadata length {len(metadata)} doesn't match "
                f"embeddings shape {embeddings.shape}"
            )
            
        os.makedirs(output_dir, exist_ok=True)
        emb_path = os.path.join(output_dir, f"{prefix}_embeddings.npy")
        meta_path = os.path.join(output_dir, f"{prefix}_metadata.json")
        
        try:
            np.save(emb_path, embeddings)
        except Exception as e:
            raise EmbeddingError(f"Failed to save embeddings: {str(e)}")
            
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            raise EmbeddingError(f"Failed to save metadata: {str(e)}")
            
        logger.info(
            f"Saved embeddings (shape {embeddings.shape}) and "
            f"metadata ({len(metadata)} entries)"
        )
        
        return {
            "embeddings_path": emb_path,
            "metadata_path": meta_path
        }
        
    except Exception as e:
        raise EmbeddingError(f"Failed to save embeddings and metadata: {str(e)}")

# ----------------------------
# FAISS Index utilities
# ----------------------------
def create_faiss_index(embeddings: np.ndarray, 
                      index_path: str = "indexes/sample.index",
                      use_hnsw: bool = False) -> faiss.Index:
    """
    Create and save FAISS index.
    
    Args:
        embeddings (np.ndarray): Embeddings to index
        index_path (str): Path to save index
        use_hnsw (bool): Whether to use HNSW index
        
    Returns:
        faiss.Index: Created index
        
    Raises:
        IndexError: If index creation fails
    """
    try:
        if not isinstance(embeddings, np.ndarray):
            raise IndexError("embeddings must be numpy array")
            
        if embeddings.dtype != np.float32:
            logger.warning("Converting embeddings to float32")
            embeddings = embeddings.astype(np.float32)
            
        d = embeddings.shape[1]
        
        try:
            if use_hnsw:
                index = faiss.IndexHNSWFlat(d, 32)
                index.hnsw.efConstruction = 200
            else:
                index = faiss.IndexFlatIP(d)
                
            index.add(embeddings)
            
        except Exception as e:
            raise IndexError(f"Failed to create FAISS index: {str(e)}")
            
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)
        except Exception as e:
            raise IndexError(f"Failed to save FAISS index: {str(e)}")
            
        logger.info(
            f"Created and saved FAISS index with {index.ntotal} vectors "
            f"to {index_path}"
        )
        return index
        
    except Exception as e:
        raise IndexError(f"Failed to create/save FAISS index: {str(e)}")

def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Load a FAISS index from disk.
    
    Args:
        index_path (str): Path to the index file
        
    Returns:
        faiss.Index: Loaded index
        
    Raises:
        IndexError: If loading fails
    """
    try:
        if not os.path.exists(index_path):
            raise IndexError(f"Index file not found: {index_path}")
            
        if not os.path.isfile(index_path):
            raise IndexError(f"Not a file: {index_path}")
            
        try:
            index = faiss.read_index(index_path)
        except Exception as e:
            raise IndexError(f"Failed to read FAISS index: {str(e)}")
            
        if index.ntotal == 0:
            logger.warning("Loaded index contains no vectors")
            
        logger.info(f"Loaded FAISS index from {index_path} (ntotal={index.ntotal})")
        return index
        
    except Exception as e:
        raise IndexError(f"Failed to load FAISS index: {str(e)}")

# ----------------------------
# CLI / Example usage
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    def main():
        try:
            parser = argparse.ArgumentParser(
                description="Embed text chunks and create FAISS index"
            )
            parser.add_argument(
                "--mode", 
                choices=["embed", "index", "both"], 
                default="both",
                help="Operation mode"
            )
            parser.add_argument(
                "--prefix", 
                default="sample",
                help="Prefix for output files"
            )
            parser.add_argument(
                "--output-dir",
                default="indexes",
                help="Output directory"
            )
            args = parser.parse_args()

            if args.mode in ["embed", "both"]:
                try:
                    model = load_model()
                    
                    meta_path = os.path.join(args.output_dir, f"{args.prefix}_metadata.json")
                    if not os.path.exists(meta_path):
                        raise EmbeddingError(f"Metadata file not found: {meta_path}")
                        
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        try:
                            chunks = json.load(fh)
                        except json.JSONDecodeError as e:
                            raise EmbeddingError(f"Invalid metadata JSON: {str(e)}")
                            
                    texts = [c.get("text", "") for c in chunks]
                    embeddings = batch_encode_texts(model, texts)
                    
                    save_embeddings(
                        embeddings, 
                        chunks, 
                        prefix=args.prefix,
                        output_dir=args.output_dir
                    )
                    
                except Exception as e:
                    logger.error(f"Embedding failed: {str(e)}")
                    raise

            if args.mode in ["index", "both"]:
                try:
                    emb_path = os.path.join(args.output_dir, f"{args.prefix}_embeddings.npy")
                    if not os.path.exists(emb_path):
                        raise IndexError(f"Embeddings file not found: {emb_path}")
                        
                    embs = np.load(emb_path)
                    
                    idx = create_faiss_index(
                        embs,
                        os.path.join(args.output_dir, f"{args.prefix}.index"),
                        use_hnsw=False
                    )
                    
                    # Verify index
                    load_faiss_index(os.path.join(args.output_dir, f"{args.prefix}.index"))
                    
                except Exception as e:
                    logger.error(f"Indexing failed: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Process failed: {str(e)}")
            exit(1)
            
    main()