"""Database connection and management"""
import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import logging

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info(f"Connected to PostgreSQL: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
            self._setup_pgvector()
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _setup_pgvector(self):
        """Setup pgvector extension and tables"""
        cursor = self.conn.cursor()
        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pgvector;")
            logger.info("pgvector extension enabled")

            # Create tour_embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tour_embeddings (
                    id SERIAL PRIMARY KEY,
                    tour_id INTEGER NOT NULL,
                    tour_name VARCHAR(255),
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tour_id, chunk_index)
                );
            """)
            logger.info("tour_embeddings table created")

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tour_embeddings_vector ON tour_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tour_embeddings_tour_id ON tour_embeddings(tour_id);")
            logger.info("Indexes created")

            # Create chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tour_chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    session_id VARCHAR(255),
                    question TEXT,
                    answer TEXT,
                    source_tour_ids INTEGER[],
                    tokens_used INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON tour_chat_history(session_id);")
            logger.info("tour_chat_history table created")

            self.conn.commit()
        except Exception as e:
            logger.error(f"Error setting up pgvector: {e}")
            self.conn.rollback()
            raise

    def save_embedding(self, tour_id: int, tour_name: str, chunk_index: int, chunk_text: str, embedding: list, metadata: dict):
        """Save embedding to database"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO tour_embeddings (tour_id, tour_name, chunk_index, chunk_text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tour_id, chunk_index) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP;
            """, (tour_id, tour_name, chunk_index, chunk_text, embedding, metadata))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving embedding: {e}")
            self.conn.rollback()
            raise

    def search_embeddings(self, query_embedding: list, top_k: int = 5):
        """Search similar embeddings"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("""
                SELECT id, tour_id, tour_name, chunk_index, chunk_text, metadata,
                       1 - (embedding <=> %s) as similarity
                FROM tour_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (query_embedding, query_embedding, top_k))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            raise

    def get_all_tour_ids(self):
        """Get all tour IDs that have embeddings"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT tour_id FROM tour_embeddings;")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting tour IDs: {e}")
            return []

    def clear_embeddings(self, tour_id: int):
        """Clear embeddings for a tour (for refresh)"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("DELETE FROM tour_embeddings WHERE tour_id = %s;", (tour_id,))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
            self.conn.rollback()

    def save_chat_history(self, user_id: int, session_id: str, question: str, answer: str, source_tour_ids: list, tokens_used: int):
        """Save chat interaction to history"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO tour_chat_history (user_id, session_id, question, answer, source_tour_ids, tokens_used)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (user_id, session_id, question, answer, source_tour_ids, tokens_used))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
            self.conn.rollback()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
