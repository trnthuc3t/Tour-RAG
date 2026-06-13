"""Configuration for RAG service"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')

# PostgreSQL Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 5433))
DB_NAME = os.getenv('DB_NAME', 'rag_service')
DB_USER = os.getenv('DB_USER', 'openpg')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'openpgpwd')
DB_BOOTSTRAP_DATABASE = os.getenv('DB_BOOTSTRAP_DATABASE', 'postgres')

# Odoo Configuration
ODOO_URL = os.getenv('ODOO_URL', 'http://localhost:8069')
ODOO_DB = os.getenv('ODOO_DB', 'odoo')

# RAG Configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 100))
TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', 5))
RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', 3))

# Service Configuration
SERVICE_HOST = os.getenv('SERVICE_HOST', '0.0.0.0')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', 8001))
SERVICE_WORKERS = int(os.getenv('SERVICE_WORKERS', 2))

# Embedding output size used by both DB schema and embedding requests.
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 768))

# System prompt for answer generation.
SYSTEM_PROMPT = """Bạn là một trợ lý tư vấn du lịch thông minh cho một công ty du lịch tại Việt Nam. 
Bạn giúp khách hàng tìm hiểu về các tour du lịch, giá cả, lịch trình, và các chính sách liên quan.

Quy tắc:
1. Chỉ trả lời dựa trên thông tin du lịch được cung cấp. Không được bịa ra thông tin.
2. Luôn trích dẫn tên tour và các chi tiết cụ thể từ dữ liệu.
3. Nếu không tìm thấy thông tin liên quan, hãy nói rõ "Xin lỗi, tôi không có thông tin chi tiết về điều này".
4. Hãy tư vấn thêm tour liên quan nếu có.
5. Trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp.
6. Nếu khách hỏi về giá, đừng quên thêm "Vui lòng liên hệ để biết thêm thông tin về khuyến mãi hiện tại".
7. Luôn sẵn sàng trợ giúp khách đặt tour hoặc liên hệ với nhân viên bán hàng.
"""

print(f"""
RAG Service Configuration:
- Gemini Model: {GEMINI_MODEL}
- Embedding Model: {GEMINI_EMBEDDING_MODEL}
- Database: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}
- Odoo: {ODOO_URL}
- Embedding Dimension: {EMBEDDING_DIMENSION}
- Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}
- Top-K Retrieval: {TOP_K_RETRIEVAL}, Rerank: {RERANK_TOP_K}
""")
