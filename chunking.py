"""Text chunking utilities"""
from __future__ import annotations
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks with a soft sentence boundary heuristic."""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        
        if end < len(text):
            # Prefer a natural break when possible to reduce semantic cuts.
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_comma = chunk.rfind(',')
            
            break_point = max(last_period, last_newline, last_comma)
            if break_point > chunk_size * 0.7:
                end = start + break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def chunk_tour_data(tour_name: str, description: str, detail_info: str) -> List[Dict]:
    """Build ordered chunks for one tour record."""
    chunks = []
    chunk_index = 0
    
    if description:
        text = f"{tour_name}\n\n{description}"
        for chunk in chunk_text(text):
            chunks.append({
                'chunk_index': chunk_index,
                'chunk_text': chunk
            })
            chunk_index += 1
    
    if detail_info:
        for chunk in chunk_text(detail_info):
            chunks.append({
                'chunk_index': chunk_index,
                'chunk_text': chunk
            })
            chunk_index += 1
    
    return chunks if chunks else [{'chunk_index': 0, 'chunk_text': tour_name}]
