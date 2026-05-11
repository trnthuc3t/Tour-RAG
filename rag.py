"""RAG (Retrieval Augmented Generation) logic"""
from __future__ import annotations
from typing import List, Dict
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT, TOP_K_RETRIEVAL, RERANK_TOP_K
from embeddings import get_embedding
from db import Database
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(GEMINI_MODEL)


class RAGEngine:
    def __init__(self):
        self.db = Database()
        self.llm = llm

    def _detect_price_intent(self, query: str) -> str:
        q = (query or "").lower()

        cheapest_keywords = ["rẻ nhất", "thấp nhất", "giá thấp nhất", "ít tiền nhất", "tiết kiệm nhất"]
        expensive_keywords = ["đắt nhất", "cao nhất", "giá cao nhất"]

        if any(keyword in q for keyword in cheapest_keywords):
            return "cheapest"
        if any(keyword in q for keyword in expensive_keywords):
            return "most_expensive"
        return "none"

    def _answer_price_rank_query(self, query: str, intent: str) -> dict:
        ascending = intent == "cheapest"
        ranked_rows = self.db.get_tours_ranked_by_price(ascending=ascending, limit=5)

        if not ranked_rows:
            return {
                'answer': 'Xin lỗi, tôi chưa có dữ liệu giá tour để so sánh lúc này.',
                'sources': []
            }

        top = ranked_rows[0]
        label = "rẻ nhất" if ascending else "đắt nhất"
        lines = [
            f"Tour {label} hiện tại là {top['tour_name']} với giá {float(top['price']):,.0f} {top['currency']}.",
            "Một số tour theo thứ tự giá để bạn tham khảo:"
        ]
        for idx, row in enumerate(ranked_rows, start=1):
            lines.append(f"{idx}. {row['tour_name']} - {float(row['price']):,.0f} {row['currency']}")
        lines.append("Vui lòng liên hệ để biết thêm thông tin về khuyến mãi hiện tại.")

        sources = []
        for row in ranked_rows[:3]:
            sources.append({
                'tour_id': row['tour_id'],
                'tour_name': row['tour_name'],
                'category': row.get('category', 'N/A'),
                'price': float(row['price']),
                'currency': row.get('currency', 'VND'),
                'similarity': 1.0
            })

        return {
            'answer': "\n".join(lines),
            'sources': sources
        }

    def retrieve_documents(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        try:
            query_embedding = get_embedding(query, task_type='retrieval_query')
            results = []

            if query_embedding:
                results = self.db.search_embeddings(query_embedding, top_k=top_k)
            else:
                logger.warning("Failed to generate query embedding, switching to keyword fallback")

            if not results:
                results = self.db.search_embeddings_by_keyword(query, top_k=top_k)
            
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

    def rerank_documents(self, documents: List[Dict], query: str, top_k: int = RERANK_TOP_K) -> List[Dict]:
        """Return top-k documents ordered by similarity score."""
        sorted_docs = sorted(documents, key=lambda x: x['similarity'], reverse=True)
        return sorted_docs[:top_k]

    def generate_answer(self, query: str, documents: list) -> str:
        """Generate a final answer from the retrieved tour context."""
        try:
            context = "\n\n".join([
                f"**{doc['tour_name']}** (Điểm đến: {doc['metadata'].get('category', 'N/A')})\n"
                f"Giá: {doc['metadata'].get('price', 'N/A')} {doc['metadata'].get('currency', 'VND')}\n"
                f"Thông tin: {doc['chunk_text']}"
                for doc in documents
            ])
            
            prompt = f"""{SYSTEM_PROMPT}

Dựa trên thông tin du lịch dưới đây, hãy trả lời câu hỏi của khách hàng:

Thông tin du lịch:
{context}

Câu hỏi: {query}

Hãy trả lời bằng tiếng Việt, thân thiện và cung cấp thông tin chi tiết từ dữ liệu trên."""

            response = self.llm.generate_content(prompt)
            answer = (response.text or '').strip()
            if not answer:
                answer = 'Xin lỗi, tôi chưa tạo được câu trả lời phù hợp. Vui lòng thử lại.'
            
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Xin lỗi, tôi không thể trả lời câu hỏi này. Vui lòng liên hệ nhân viên bán hàng để được hỗ trợ."

    def answer_question(self, query: str) -> dict:
        """Execute retrieve, rank, and generate for one user question."""
        try:
            price_intent = self._detect_price_intent(query)
            if price_intent != "none":
                return self._answer_price_rank_query(query, price_intent)

            documents = self.retrieve_documents(query, top_k=TOP_K_RETRIEVAL)
            
            if not documents:
                return {
                    'answer': 'Xin lỗi, tôi không tìm thấy thông tin liên quan trong dữ liệu của công ty. Vui lòng liên hệ nhân viên bán hàng để được hỗ trợ tốt hơn.',
                    'sources': []
                }
            
            reranked_docs = self.rerank_documents(documents, query)
            
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
