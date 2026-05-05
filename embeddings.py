"""Embedding utilities using Gemini API"""
from __future__ import annotations
from typing import List, Optional
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMENSION
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)


def get_embedding(text: str, task_type: str = 'retrieval_document') -> Optional[List[float]]:
    """Generate one embedding vector for text."""
    try:
        if len(text) > 8000:
            text = text[:8000]

        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSION,
        )
        embedding = response['embedding']
        logger.debug(f"Generated embedding for text of length {len(text)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def get_embeddings_batch(texts: List[str], task_type: str = 'retrieval_document') -> List[Optional[List[float]]]:
    """Generate embeddings for many texts by issuing sequential API calls."""
    embeddings: List[Optional[List[float]]] = []
    for text in texts:
        try:
            if len(text) > 8000:
                text = text[:8000]

            response = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=text,
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSION,
            )
            embeddings.append(response['embedding'])
        except Exception as e:
            logger.error(f"Error generating embedding for one text: {e}")
            embeddings.append(None)

    logger.debug(f"Generated {len([x for x in embeddings if x is not None])} embeddings out of {len(texts)}")
    return embeddings
