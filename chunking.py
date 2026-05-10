"""Text chunking utilities"""
from __future__ import annotations
from typing import List, Dict
import html
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)


def _normalize_text(raw_text: str) -> str:
    if not raw_text:
        return ''

    text = html.unescape(raw_text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
    normalized_name = _normalize_text(tour_name)
    normalized_description = _normalize_text(description)
    normalized_detail = _normalize_text(detail_info)
    
    if normalized_description:
        text = f"{normalized_name}\n\n{normalized_description}"
        for chunk in chunk_text(text):
            chunks.append({
                'chunk_index': chunk_index,
                'chunk_text': chunk
            })
            chunk_index += 1
    
    if normalized_detail:
        for chunk in chunk_text(normalized_detail):
            chunks.append({
                'chunk_index': chunk_index,
                'chunk_text': chunk
            })
            chunk_index += 1
    
    return chunks if chunks else [{'chunk_index': 0, 'chunk_text': normalized_name or tour_name}]
