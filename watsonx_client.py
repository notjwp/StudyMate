# watsonx_client.py
import os
import json
import logging
import requests
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

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
class WatsonXConfig:
    """Configuration for WatsonX API."""
    api_url: str
    api_key: str
    model_id: str = "mistralai/mixtral-8x7b-instruct"
    
    @classmethod
    def from_env(cls) -> 'WatsonXConfig':
        """Create configuration from environment variables."""
        api_url = os.environ.get("WATSONX_API_URL")
        api_key = os.environ.get("WATSONX_API_KEY")
        model_id = os.environ.get("WATSONX_MODEL_ID", cls.model_id)
        
        if not api_url:
            raise ConfigurationError("WATSONX_API_URL environment variable not set")
        if not api_key:
            raise ConfigurationError("WATSONX_API_KEY environment variable not set")
            
        return cls(api_url=api_url, api_key=api_key, model_id=model_id)

# Initialize configuration
try:
    config = WatsonXConfig.from_env()
except ConfigurationError as e:
    logger.error(f"Configuration error: {str(e)}")
    raise

# -------------------------------------------------------------------
# Prompt template & system instruction
# -------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are an assistant that answers students' questions using ONLY the provided document excerpts.
Cite each factual claim by appending a source tag like [doc_name | page N].
If the documents do not contain sufficient information to answer, respond exactly:
"Insufficient information in provided documents."
Be concise, pedagogical, and avoid hallucination. Keep temperature low for factual answers.
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

def validate_chunks(chunks: List[Dict]) -> List[Chunk]:
    """
    Validate and convert chunk dictionaries to Chunk objects.
    
    Args:
        chunks (List[Dict]): List of chunk dictionaries
        
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
            if not isinstance(c, dict):
                raise PromptError(f"Chunk {i} is not a dictionary")
                
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

# -------------------------------------------------------------------
# Main function: call IBM watsonx Mixtral
# -------------------------------------------------------------------
def validate_parameters(prompt: str, max_tokens: int, 
                       temperature: float) -> None:
    """
    Validate API call parameters.
    
    Args:
        prompt (str): The prompt to validate
        max_tokens (int): Maximum tokens to generate
        temperature (float): Temperature parameter
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
        
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
        
    if not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")
        
    if max_tokens < 1 or max_tokens > 4096:
        raise ValueError("max_tokens must be between 1 and 4096")
        
    if not isinstance(temperature, float):
        raise ValueError("temperature must be a float")
        
    if temperature < 0.0 or temperature > 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")

def parse_response(response_json: dict) -> str:
    """
    Parse the API response to extract generated text.
    
    Args:
        response_json (dict): API response JSON
        
    Returns:
        str: Generated text
        
    Raises:
        APIError: If response format is invalid
    """
    try:
        # Try different response formats
        if "results" in response_json and isinstance(response_json["results"], list):
            return response_json["results"][0].get("generated_text", "")
            
        if "generated_text" in response_json:
            return response_json["generated_text"]
            
        if "choices" in response_json and isinstance(response_json["choices"], list):
            choice = response_json["choices"][0]
            return (choice.get("text") or 
                   choice.get("message", {}).get("content", ""))
                   
        raise APIError(f"Unexpected response format: {response_json}")
        
    except Exception as e:
        raise APIError(f"Failed to parse response: {str(e)}")

def call_watsonx(prompt: str,
                 system_instruction: str = SYSTEM_INSTRUCTION,
                 max_tokens: int = 512,
                 temperature: float = 0.1) -> str:
    """
    Call IBM watsonx API with error handling.
    
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
        # Validate parameters
        validate_parameters(prompt, max_tokens, temperature)
        
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model_id": config.model_id,
            "input": [
                {
                    "role": "system",
                    "content": system_instruction.strip()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }

        try:
            logger.debug(f"Calling WatsonX API with model {config.model_id}")
            resp = requests.post(
                config.api_url,
                headers=headers,
                json=payload,
                timeout=40
            )
            resp.raise_for_status()
            
        except requests.Timeout:
            raise APIError("API request timed out")
            
        except requests.RequestException as e:
            raise APIError(f"API request failed: {str(e)}")

        try:
            response_json = resp.json()
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")

        return parse_response(response_json)
        
    except Exception as e:
        if isinstance(e, (APIError, ConfigurationError, ValueError)):
            raise
        raise APIError(f"Unexpected error in API call: {str(e)}")

# -------------------------------------------------------------------
# Example usage (for testing standalone)
# -------------------------------------------------------------------
if __name__ == "__main__":
    def main():
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
                
            except PromptError as e:
                logger.error(f"Failed to assemble prompt: {str(e)}")
                return

            try:
                # Only make API call if credentials are properly configured
                answer = call_watsonx(prompt)
                print("🔹 Model answer:\n", answer)
                
            except (ConfigurationError, APIError) as e:
                logger.error(f"API call failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return

    main()