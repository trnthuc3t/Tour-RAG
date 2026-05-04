"""Embedding utilities using OpenAI"""
from __future__ import annotations
from typing import List, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for text using OpenAI API.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector or None if error
    """
    try:
        # Truncate text to avoid token limits
        if len(text) > 8000:
            text = text[:8000]
        
        response = client.embeddings.create(
            input=text,
            model=OPENAI_EMBEDDING_MODEL
        )
        
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding for text of length {len(text)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def get_embeddings_batch(texts: list[str]) -> list[list[float] | None]:
    """
    Get embeddings for multiple texts (batch).
    
    Args:
        texts: List of texts to embed
    
    Returns:
        List of embedding vectors
    """
    try:
        # Truncate texts
        texts = [t[:8000] if len(t) > 8000 else t for t in texts]
        
        response = client.embeddings.create(
            input=texts,
            model=OPENAI_EMBEDDING_MODEL
        )
        
        embeddings = [item.embedding for item in response.data]
        logger.debug(f"Generated {len(embeddings)} embeddings in batch")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return [None] * len(texts)
