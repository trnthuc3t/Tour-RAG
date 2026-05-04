"""Text chunking utilities"""
from __future__ import annotations
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
    
    Returns:
        List of chunks
    """
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        
        # Try to break at sentence boundary if not at end
        if end < len(text):
            # Look for last period, comma, or newline
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_comma = chunk.rfind(',')
            
            break_point = max(last_period, last_newline, last_comma)
            if break_point > chunk_size * 0.7:  # At least 70% of chunk size
                end = start + break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = end - overlap
    
    return chunks


def chunk_tour_data(tour_name: str, description: str, detail_info: str) -> List[Dict]:
    """
    Chunk tour data into semantic units.
    
    Returns:
        List of dicts with chunk_index and chunk_text
    """
    chunks = []
    chunk_index = 0
    
    # Chunk 0: Tour name + description
    if description:
        text = f"{tour_name}\n\n{description}"
        for chunk in chunk_text(text):
            chunks.append({
                'chunk_index': chunk_index,
                'chunk_text': chunk
            })
            chunk_index += 1
    
    # Additional chunks: detail information
    if detail_info:
        for chunk in chunk_text(detail_info):
            chunks.append({
                'chunk_index': chunk_index,
                'chunk_text': chunk
            })
            chunk_index += 1
    
    return chunks if chunks else [{'chunk_index': 0, 'chunk_text': tour_name}]
