"""Pydantic models for RAG service"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class TourData(BaseModel):
    """Tour data from Odoo"""
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
    """Chunk of text with embedding"""
    tour_id: int
    tour_name: str
    chunk_index: int
    chunk_text: str
    metadata: dict  # category, price, etc


class ChatRequest(BaseModel):
    """User chat request"""
    question: str
    session_id: Optional[str] = None
    user_id: Optional[int] = None


class RetrievedDocument(BaseModel):
    """Retrieved document from RAG"""
    tour_id: int
    tour_name: str
    chunk_text: str
    score: float
    category: str
    price: float
    currency: str


class ChatResponse(BaseModel):
    """Chat response"""
    answer: str
    sources: List[RetrievedDocument]
    session_id: str
    tokens_used: Optional[int] = None


class IngestRequest(BaseModel):
    """Request to ingest tours"""
    skip_existing: bool = True


class IngestResponse(BaseModel):
    """Response from ingest"""
    message: str
    tours_processed: int
    chunks_created: int
    time_taken: float
