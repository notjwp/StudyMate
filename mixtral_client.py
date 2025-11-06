"""
Mixtral API client for StudyMate
"""
import logging
import json
import os
import time
from pathlib import Path
import requests
from typing import List, Dict, Optional, Union
from retrieve import SearchResult
from dataclasses import dataclass
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class ConfigurationError(Exception):
    """Raised when there are issues with the configuration."""
    pass

class APIError(Exception):
    """Raised when the API call fails."""
    pass

class PromptError(Exception):
    """Raised when there are issues with prompt assembly."""
    pass

@dataclass
class MixtralConfig:
    """Configuration for Mixtral API."""
    api_url: str
    api_key: str
    model_id: str = "mistral-medium"  # Supported models: mistral-tiny, mistral-small, mistral-medium
    
    @classmethod
    def from_env(cls) -> 'MixtralConfig':
        """Create configuration from environment variables."""
        api_url = os.environ.get("MISTRAL_API_URL", "https://api.mistral.ai/v1")
        api_key = os.environ.get("MISTRAL_API_KEY")
        
        if not api_key:
            raise ConfigurationError("MISTRAL_API_KEY environment variable not set")
            
        return cls(api_url=api_url, api_key=api_key)

# Initialize configuration
try:
    config = MixtralConfig.from_env()
except ConfigurationError as e:
    logger.error(f"Configuration error: {str(e)}")
    raise

# System instruction and prompt template
SYSTEM_INSTRUCTION = """
You are an assistant that answers students' questions using ONLY the provided document excerpts.
Cite each factual claim by appending a source tag like [doc_name | page N].
If the documents do not contain sufficient information to answer, respond exactly:
"Insufficient information in provided documents."
Be concise, pedagogical, and avoid hallucination. Keep focused on factual answers.
"""

PROMPT_TEMPLATE = """
CONTEXT:
{context_chunks}

QUESTION:
{question}

INSTRUCTIONS:
- Use only the CONTEXT above to answer.
- Cite sources inline as [doc_id | page X].
- If answer cannot be derived from context, reply exactly:
  "Insufficient information in provided documents."
- Keep answer concise (<= 300 words) and include a one-line summary at the end.
"""

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    doc_id: str
    page_num: int
    text: str

def validate_chunks(chunks: List[Union[Dict, 'SearchResult']]) -> List[Chunk]:
    """
    Validate and convert chunks to Chunk objects.
    
    Args:
        chunks (List[Union[Dict, SearchResult]]): List of chunks
        
    Returns:
        List[Chunk]: List of validated Chunk objects
        
    Raises:
        PromptError: If chunks are invalid
    """
    if not isinstance(chunks, list):
        raise PromptError("chunks must be a list")
        
    validated = []
    for i, c in enumerate(chunks):
        try:
            # Handle SearchResult objects
            if hasattr(c, 'doc_id') and hasattr(c, 'page_num') and hasattr(c, 'text'):
                chunk = Chunk(
                    doc_id=str(c.doc_id),
                    page_num=int(c.page_num),
                    text=str(c.text)
                )
                validated.append(chunk)
                continue
                
            # Handle dictionaries
            if isinstance(c, dict):
                required = ['doc_id', 'page_num', 'text']
                missing = [f for f in required if f not in c]
                if missing:
                    raise PromptError(f"Chunk {i} missing fields: {missing}")
                    
                chunk = Chunk(
                    doc_id=str(c['doc_id']),
                    page_num=int(c['page_num']),
                    text=str(c['text'])
                )
                validated.append(chunk)
                continue
                
            raise PromptError(f"Chunk {i} is not a valid type")
            
        except (ValueError, TypeError) as e:
            raise PromptError(f"Invalid chunk {i}: {str(e)}")
            
    return validated

def assemble_prompt(chunks: List[Dict], question: str, 
                   max_chars: int = 15000) -> str:
    """
    Build prompt from retrieved chunks.
    
    Args:
        chunks (List[Dict]): List of chunk dictionaries
        question (str): User's question
        max_chars (int): Maximum characters for context
        
    Returns:
        str: Assembled prompt
        
    Raises:
        PromptError: If prompt assembly fails
    """
    try:
        if not isinstance(question, str):
            raise PromptError("question must be a string")
            
        if not question.strip():
            raise PromptError("question cannot be empty")
            
        if max_chars < 1000:
            raise PromptError("max_chars must be at least 1000")
            
        # Validate chunks
        validated_chunks = validate_chunks(chunks)
        
        # Build context
        ctxs = []
        total = 0
        
        for chunk in validated_chunks:
            try:
                snippet = (f"[{chunk.doc_id} | page {chunk.page_num}]\n"
                         f"{chunk.text}\n")
                         
                if total + len(snippet) > max_chars:
                    break
                    
                ctxs.append(snippet)
                total += len(snippet)
                
            except Exception as e:
                logger.warning(f"Error processing chunk: {str(e)}")
                continue
                
        if not ctxs:
            raise PromptError("No valid chunks to include in prompt")
            
        context = "\n\n---\n\n".join(ctxs)
        
        return PROMPT_TEMPLATE.format(
            context_chunks=context,
            question=question
        )
        
    except Exception as e:
        raise PromptError(f"Failed to assemble prompt: {str(e)}")

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, APIError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds..."
    )
)
def call_mixtral(prompt: str,
                system_instruction: str = SYSTEM_INSTRUCTION,
                max_tokens: int = 512,
                temperature: float = 0.1) -> str:
    """
    Call Mixtral API with error handling.
    
    Args:
        prompt (str): The prompt to send
        system_instruction (str): System context
        max_tokens (int): Maximum tokens to generate
        temperature (float): Temperature parameter
        
    Returns:
        str: Generated text response
        
    Raises:
        APIError: If the API call fails
        ConfigurationError: If configuration is invalid
        ValueError: If parameters are invalid
    """
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
            
        if max_tokens < 1 or max_tokens > 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
            
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistral-medium",  # Fixed model name for Mistral API
            "messages": [
                {
                    "role": "system",
                    "content": system_instruction.strip()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            logger.debug(f"Calling Mixtral API")
            resp = requests.post(
                f"{config.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            
        except requests.Timeout:
            raise APIError("API request timed out")
            
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    raise APIError(f"API request failed: {error_detail.get('error', {}).get('message', str(e))}")
                except json.JSONDecodeError:
                    raise APIError(f"API request failed: {str(e)} - {e.response.text}")
            else:
                raise APIError(f"API request failed: {str(e)}")

        try:
            response = resp.json()
            if "choices" not in response:
                raise APIError(f"Unexpected response format: {response}")
                
            answer = response["choices"][0]["message"]["content"]
            return answer.strip()
            
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
            
    except Exception as e:
        if isinstance(e, (APIError, ConfigurationError, ValueError)):
            raise
        raise APIError(f"Unexpected error in API call: {str(e)}")

# Example usage (for testing)
if __name__ == "__main__":
    def main():
        logger.info("🔎 Testing Mixtral API...")
        
        try:
            # Test data
            chunks = [
                {
                    "doc_id": "sample",
                    "page_num": 2,
                    "text": "Mitosis is the process of cell division resulting in two identical daughter cells."
                },
                {
                    "doc_id": "sample",
                    "page_num": 5,
                    "text": "Stages of mitosis include prophase, metaphase, anaphase, and telophase."
                }
            ]

            question = "What are the main stages of mitosis?"
            
            try:
                prompt = assemble_prompt(chunks, question)
                logger.info("Successfully assembled prompt")
                print("🔹 Prompt preview:\n", prompt[:500], "\n---")
                
                answer = call_mixtral(prompt)
                print("🔹 Model answer:\n", answer)
                
            except (PromptError, APIError) as e:
                logger.error(f"Error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    main()