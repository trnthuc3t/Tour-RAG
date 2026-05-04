"""RAG (Retrieval Augmented Generation) logic"""
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, OPENAI_MODEL, SYSTEM_PROMPT, TOP_K_RETRIEVAL, RERANK_TOP_K
from embeddings import get_embedding
from db import Database
import logging
import json

logger = logging.getLogger(__name__)

# Initialize LangChain LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0.7,
)


class RAGEngine:
    def __init__(self):
        self.db = Database()
        self.llm = llm

    def retrieve_documents(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents with scores
        """
        try:
            # Get query embedding
            query_embedding = get_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search similar embeddings
            results = self.db.search_embeddings(query_embedding, top_k=top_k)
            
            # Format results
            documents = []
            for result in results:
                documents.append({
                    'tour_id': result['tour_id'],
                    'tour_name': result['tour_name'],
                    'chunk_text': result['chunk_text'],
                    'similarity': result['similarity'],
                    'metadata': result['metadata'] if result['metadata'] else {}
                })
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def rerank_documents(self, documents: list, query: str, top_k: int = RERANK_TOP_K) -> list:
        """
        Re-rank documents based on relevance (simple version).
        In production, use Cohere reranker or similar.
        
        Args:
            documents: List of documents
            query: User query
            top_k: Top K to return
        
        Returns:
            Re-ranked documents
        """
        # For now, just return top-k by similarity score
        # In future, can use more sophisticated re-ranking
        sorted_docs = sorted(documents, key=lambda x: x['similarity'], reverse=True)
        return sorted_docs[:top_k]

    def generate_answer(self, query: str, documents: list) -> str:
        """
        Generate answer using LLM based on retrieved documents.
        
        Args:
            query: User question
            documents: Retrieved documents
        
        Returns:
            Generated answer
        """
        try:
            # Build context from documents
            context = "\n\n".join([
                f"**{doc['tour_name']}** (Điểm đến: {doc['metadata'].get('category', 'N/A')})\n"
                f"Giá: {doc['metadata'].get('price', 'N/A')} {doc['metadata'].get('currency', 'VND')}\n"
                f"Thông tin: {doc['chunk_text']}"
                for doc in documents
            ])
            
            # Build messages
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""Dựa trên thông tin du lịch dưới đây, hãy trả lời câu hỏi của khách hàng:

Thông tin du lịch:
{context}

Câu hỏi: {query}

Hãy trả lời bằng tiếng Việt, thân thiện và cung cấp thông tin chi tiết từ dữ liệu trên.""")
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Xin lỗi, tôi không thể trả lời câu hỏi này. Vui lòng liên hệ nhân viên bán hàng để được hỗ trợ."

    def answer_question(self, query: str) -> dict:
        """
        Full RAG pipeline: retrieve -> rerank -> generate.
        
        Args:
            query: User question
        
        Returns:
            Dict with answer and sources
        """
        try:
            # Retrieve
            documents = self.retrieve_documents(query, top_k=TOP_K_RETRIEVAL)
            
            if not documents:
                return {
                    'answer': 'Xin lỗi, tôi không tìm thấy thông tin liên quan trong dữ liệu của công ty. Vui lòng liên hệ nhân viên bán hàng để được hỗ trợ tốt hơn.',
                    'sources': []
                }
            
            # Re-rank
            reranked_docs = self.rerank_documents(documents, query)
            
            # Generate
            answer = self.generate_answer(query, reranked_docs)
            
            return {
                'answer': answer,
                'sources': [{
                    'tour_id': doc['tour_id'],
                    'tour_name': doc['tour_name'],
                    'category': doc['metadata'].get('category', 'N/A'),
                    'price': doc['metadata'].get('price', 'N/A'),
                    'currency': doc['metadata'].get('currency', 'VND'),
                    'similarity': doc['similarity']
                } for doc in reranked_docs]
            }
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'answer': 'Có lỗi xảy ra. Vui lòng thử lại sau.',
                'sources': []
            }
