"""Database connection and management"""
from __future__ import annotations
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_BOOTSTRAP_DATABASE,
    EMBEDDING_DIMENSION,
)
import logging

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        """Connect to the configured PostgreSQL database and initialize schema."""
        try:
            self._ensure_database_exists()
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

    def _ensure_database_exists(self):
        """Ensure the configured RAG database exists on the same PostgreSQL server."""
        admin_conn = None
        try:
            admin_conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_BOOTSTRAP_DATABASE,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            admin_conn.autocommit = True
            cursor = admin_conn.cursor()

            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            exists = cursor.fetchone() is not None
            if exists:
                logger.info("Database %s already exists", DB_NAME)
                return

            cursor.execute(
                sql.SQL("CREATE DATABASE {} ENCODING 'UTF8'").format(sql.Identifier(DB_NAME))
            )
            logger.info("Created database %s", DB_NAME)
        except Exception as e:
            logger.error(
                "Failed ensuring database %s exists using bootstrap DB %s: %s",
                DB_NAME,
                DB_BOOTSTRAP_DATABASE,
                e,
            )
            raise
        finally:
            if admin_conn:
                admin_conn.close()

    def _setup_pgvector(self):
        """Ensure extension, tables, and indexes are present."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")

            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS tour_embeddings (
                    id SERIAL PRIMARY KEY,
                    tour_id INTEGER NOT NULL,
                    tour_name VARCHAR(255),
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    embedding vector({EMBEDDING_DIMENSION}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tour_id, chunk_index)
                );
            """)
            logger.info("tour_embeddings table created")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tour_embeddings_vector ON tour_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tour_embeddings_tour_id ON tour_embeddings(tour_id);")
            logger.info("Indexes created")

            # Keep schema aligned with configured embedding size.
            self._migrate_embedding_dimension_if_needed(cursor)

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

    def _migrate_embedding_dimension_if_needed(self, cursor):
        """Migrate embedding column dimension if schema uses a different vector size."""
        cursor.execute("""
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'tour_embeddings'
              AND n.nspname = 'public'
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped;
        """)
        row = cursor.fetchone()
        if not row:
            return

        current_type = row[0] or ''
        expected_type = f'vector({EMBEDDING_DIMENSION})'
        if current_type == expected_type:
            return

        logger.warning(
            "Embedding dimension mismatch detected: current=%s expected=%s. "
            "Truncating tour_embeddings and migrating schema.",
            current_type,
            expected_type,
        )

        cursor.execute("TRUNCATE TABLE tour_embeddings;")
        cursor.execute(f"ALTER TABLE tour_embeddings ALTER COLUMN embedding TYPE vector({EMBEDDING_DIMENSION});")

        # Rebuild ANN index after vector dimension migration.
        cursor.execute("DROP INDEX IF EXISTS idx_tour_embeddings_vector;")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tour_embeddings_vector ON tour_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")

    def save_embedding(self, tour_id: int, tour_name: str, chunk_index: int, chunk_text: str, embedding: list, metadata: dict):
        """Save embedding to database"""
        import json as _json
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO tour_embeddings (tour_id, tour_name, chunk_index, chunk_text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT (tour_id, chunk_index) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP;
            """, (tour_id, tour_name, chunk_index, chunk_text, str(embedding), _json.dumps(metadata, ensure_ascii=False)))
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
                       1 - (embedding <=> %s::vector) as similarity
                FROM tour_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (str(query_embedding), str(query_embedding), top_k))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            raise

    def search_embeddings_by_keyword(self, query: str, top_k: int = 5):
        """Fallback search using lexical matching when semantic retrieval is unavailable."""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            like_pattern = f"%{query.strip()}%"
            cursor.execute(
                """
                SELECT DISTINCT ON (tour_id)
                    id,
                    tour_id,
                    tour_name,
                    chunk_index,
                    chunk_text,
                    metadata,
                    0.0 AS similarity
                FROM tour_embeddings
                WHERE tour_name ILIKE %s OR chunk_text ILIKE %s
                ORDER BY tour_id, chunk_index
                LIMIT %s;
                """,
                (like_pattern, like_pattern, top_k),
            )
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error searching embeddings by keyword: {e}")
            return []

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
