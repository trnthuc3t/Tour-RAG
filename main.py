"""FastAPI server for RAG chatbot"""
import logging
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from rag import RAGEngine
from ingest import ingest_tours
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and dispose shared resources for the API lifecycle."""
    global rag_engine
    logger.info("Starting RAG service...")
    rag_engine = RAGEngine()
    yield
    logger.info("Shutting down RAG service...")
    if rag_engine:
        rag_engine.db.close()


app = FastAPI(
    title="Tour RAG Chatbot",
    description="RAG-based chatbot for tour recommendations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Return service liveness state."""
    return {"status": "ok", "service": "tour-rag-chatbot"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a user question with retrieved tour context."""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        result = rag_engine.answer_question(request.question.strip())
        
        try:
            source_tour_ids = [source['tour_id'] for source in result.get('sources', [])]
            rag_engine.db.save_chat_history(
                user_id=request.user_id,
                session_id=session_id,
                question=request.question,
                answer=result['answer'],
                source_tour_ids=source_tour_ids,
                tokens_used=None
            )
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")
        
        response = ChatResponse(
            answer=result['answer'],
            sources=[
                {
                    'tour_id': source['tour_id'],
                    'tour_name': source['tour_name'],
                    'chunk_text': source.get('chunk_text', ''),
                    'score': source.get('similarity', 0),
                    'category': source.get('category', ''),
                    'price': source.get('price', 0),
                    'currency': source.get('currency', 'VND')
                }
                for source in result.get('sources', [])
            ],
            session_id=session_id
        )
        
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Trigger on-demand ingestion from Odoo to vector storage."""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    
    try:
        start_time = time.time()
        
        tours_processed, chunks_created = await ingest_tours(
            rag_engine.db,
            skip_existing=request.skip_existing
        )
        
        elapsed_time = time.time() - start_time
        
        return IngestResponse(
            message=f"Ingested {tours_processed} tours successfully",
            tours_processed=tours_processed,
            chunks_created=chunks_created,
            time_taken=elapsed_time
        )
    except Exception as e:
        logger.error(f"Error in ingest endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Return readiness and current indexed tour IDs."""
    if not rag_engine:
        return {"status": "not_initialized"}
    
    try:
        tour_ids = rag_engine.db.get_all_tour_ids()
        return {
            "status": "ready",
            "tours_ingested": len(tour_ids),
            "tour_ids": tour_ids
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    from config import SERVICE_HOST, SERVICE_PORT
    
    uvicorn.run(
        "main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_level="info"
    )
