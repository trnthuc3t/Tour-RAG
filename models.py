"""Pydantic models for RAG service"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class TourData(BaseModel):
    """Normalized tour payload returned by Odoo."""
    id: int
    name: str
    category: str
    price: float
    currency: str
    description: str
    detail_information: str
    full_text: str
    image_url: str
    created_at: str
    updated_at: str


class EmbeddingChunk(BaseModel):
    """Text chunk metadata before embedding persistence."""
    tour_id: int
    tour_name: str
    chunk_index: int
    chunk_text: str
    metadata: dict


class ChatRequest(BaseModel):
    """Incoming chat request from API clients."""
    question: str
    session_id: Optional[str] = None
    user_id: Optional[int] = None


class RetrievedDocument(BaseModel):
    """Tour snippet returned by retrieval."""
    tour_id: int
    tour_name: str
    chunk_text: str
    score: float
    category: str
    price: float
    currency: str


class ChatResponse(BaseModel):
    """API response for a chat turn."""
    answer: str
    sources: List[RetrievedDocument]
    session_id: str
    tokens_used: Optional[int] = None


class IngestRequest(BaseModel):
    """Request payload for ingestion."""
    skip_existing: bool = True


class IngestResponse(BaseModel):
    """Result summary returned by ingestion endpoint."""
    message: str
    tours_processed: int
    chunks_created: int
    time_taken: float
