# Tour RAG Chatbot Service

RAG-based chatbot service for answering questions about tour products using vector similarity search and Gemini API.

## Features

- **Vector Embedding**: Uses Gemini embeddings to convert tour data into vector representations
- **Similarity Search**: PostgreSQL + pgvector for efficient similarity search
- **LLM Integration**: Uses Gemini models for generating contextual answers
- **Chat History**: Tracks user interactions for analytics and improvement
- **FastAPI**: Modern async API server
- **Auto Ingest**: CLI script to ingest tours from Odoo

## Architecture

```
Odoo (/api/tours)
  ↓
Ingest Script (chunk + embed)
  ↓
PostgreSQL + pgvector (storage)
  ↓
RAG Engine (retrieve + rerank + generate)
  ↓
FastAPI Server
  ↓
React Frontend
```

## Setup

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 16+ (with pgvector extension)
- Odoo 18.0
- Gemini API key

### 2. Install Dependencies

```bash
cd rag-service
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update with your values:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
GEMINI_API_KEY=your-gemini-api-key
DB_HOST=localhost
DB_PORT=5433
DB_NAME=rag_service
ODOO_URL=http://localhost:8069
```

### 4. Setup PostgreSQL + pgvector

The service will automatically create the `pgvector` extension and required tables on first run.

Or manually run:
```bash
psql -h localhost -p 5433 -U openpg -d rag_service -f ../config/setup_pgvector.sql
```

### 5. Ingest Tours Data

Before using the chat, ingest tour data from Odoo:

```bash
# Run ingest script
python ingest.py
```

This will:
- Fetch all tours from Odoo API (`/api/tours`)
- Split text into chunks
- Generate embeddings using Gemini
- Save to PostgreSQL + pgvector

### 6. Start the Service

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Service will be available at: `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Chat (Main Endpoint)
```bash
POST /chat

Request:
{
  "question": "Tôi muốn tìm tour đi Đà Nẵng",
  "session_id": "optional-uuid",
  "user_id": null
}

Response:
{
  "answer": "Chúng tôi có các tour đi Đà Nẵng...",
  "sources": [
    {
      "tour_id": 1,
      "tour_name": "Tour Đà Nẵng 3 ngày 2 đêm",
      "category": "Miền Trung",
      "price": 1500000,
      "currency": "VND",
      "similarity": 0.95
    }
  ],
  "session_id": "uuid"
}
```

### Ingest Tours Data
```bash
POST /ingest

Request:
{
  "skip_existing": true
}

Response:
{
  "message": "Ingested 15 tours successfully",
  "tours_processed": 15,
  "chunks_created": 45,
  "time_taken": 120.5
}
```

### Service Status
```bash
GET /status

Response:
{
  "status": "ready",
  "tours_ingested": 15,
  "tour_ids": [1, 2, 3, ...]
}
```

## Configuration

Key settings in `config.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 500 chars)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100 chars)
- `TOP_K_RETRIEVAL`: Documents to retrieve initially (default: 5)
- `RERANK_TOP_K`: Documents to use for generation (default: 3)
- `GEMINI_MODEL`: Gemini model to use (default: gemini-1.5-flash)
- `GEMINI_EMBEDDING_MODEL`: Embedding model (default: models/text-embedding-004)
- `EMBEDDING_DIMENSION`: Vector dimension in pgvector (default: 768)

## File Structure

```
rag-service/
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── models.py            # Pydantic models
├── db.py                # Database connection & queries
├── odoo_client.py       # Odoo API client
├── chunking.py          # Text chunking utilities
├── embeddings.py        # Gemini embedding generation
├── rag.py               # RAG engine (retrieve + generate)
├── ingest.py            # Tour ingest script
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Usage Example

### CLI - Ingest Tours
```bash
python ingest.py
# Output: Ingest completed! Tours processed: 15, Chunks created: 45
```

### Python - Use RAG Engine Directly
```python
from rag import RAGEngine

engine = RAGEngine()
result = engine.answer_question("Có tour nào đi Hà Nội không?")
print(result['answer'])
print(result['sources'])
```

### Frontend Integration
See `fe-tour` for React integration example.

## Performance Tips

1. **Batch Ingest**: Ingest all tours at once, not in real-time
2. **Vector Index**: pgvector uses IVFFlat with 100 lists for good balance
3. **Caching**: Consider caching common questions
4. **Reranking**: Use external reranker for better quality

## Monitoring

Chat history is logged to `tour_chat_history` table with:
- `question`: User's question
- `answer`: Generated answer
- `source_tour_ids`: Tour IDs used in answer
- `tokens_used`: Gemini token count (optional)
- `created_at`: Timestamp

Query to see top questions:
```sql
SELECT question, COUNT(*) as count 
FROM tour_chat_history 
GROUP BY question 
ORDER BY count DESC 
LIMIT 10;
```

## Troubleshooting

### "pgvector extension not found"
```bash
# Connect to PostgreSQL and enable
psql -U openpg -d rag_service
CREATE EXTENSION pgvector;
```

### "No tours found in Odoo"
- Verify Odoo is running at `ODOO_URL`
- Check API endpoint: `curl http://localhost:8069/api/tours`
- Ensure tours exist in Odoo (type=combo, sale_ok=True)

### "Gemini API rate limit"
- Reduce `CHUNK_SIZE` to generate fewer chunks
- Add retry logic in `embeddings.py`
- Check Gemini API quota and usage in Google AI Studio

## Next Steps

1. **Frontend Integration**: Add chat component to React (see `/TourReact`)
2. **Evaluation**: Create test set to measure retrieval & answer quality
3. **Reranking**: Integrate Cohere Rerank for better document relevance
4. **Caching**: Add Redis for common question caching
5. **Monitoring**: Setup logging and error tracking

## License

MIT

## Support

For issues or questions, refer to the main project README.
