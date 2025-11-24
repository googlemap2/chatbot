import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import json
import re
# Import cấu hình từ file config.py
import config

def login_huggingface():
    """Đăng nhập vào Hugging Face."""
    if config.HUGGINGFACE_TOKEN:
        login(token=config.HUGGINGFACE_TOKEN)
        print("✅ Đã đăng nhập Hugging Face!")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy HUGGINGFACE_ACCESS_TOKEN.")

# def load_llm_pipeline():
#     """
#     Tải mô hình LLM (4-bit) và tạo ra HuggingFacePipeline của LangChain.
#     """
#     print(f"Bắt đầu tải mô hình: {config.LLM_MODEL_NAME} (chế độ 4-bit)")
    
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         config.LLM_MODEL_NAME,
#         cache_dir=config.MODEL_CACHE_DIR,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         config.LLM_MODEL_NAME,
#         quantization_config=quantization_config,
#         device_map="auto",
#         cache_dir=config.MODEL_CACHE_DIR
#     )
    
#     print("✅ Tải mô hình LLM thành công (chế độ 4-bit).")

#     text_generator = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         do_sample=True,
#         temperature=0.1,
#         return_full_text=False
#     )
    
#     return HuggingFacePipeline(pipeline=text_generator)

def load_llm_pipeline():
    """
    Tải mô hình LLM bằng Transformers Pipeline (không cần vLLM).
    """
    print(f"Bắt đầu tải mô hình: {config.LLM_MODEL_NAME} (chế độ Transformers)")
    
    # Sử dụng quantization nếu có GPU
    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("🚀 Sử dụng 4-bit quantization cho GPU")
    else:
        print("💻 Chạy trên CPU (không quantization)")

    tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL_NAME,
        cache_dir=config.MODEL_CACHE_DIR,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=config.MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    print("✅ Tải mô hình LLM thành công (Transformers Pipeline).")

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Giảm từ 512 xuống 256
        do_sample=True,
        temperature=0.7,  # Tăng để nhanh hơn
        top_p=0.9,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return HuggingFacePipeline(pipeline=text_generator)

def load_embedding_model():
    """Tải mô hình embedding."""
    print(f"Bắt đầu tải embedding: {config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        cache_folder=config.MODEL_CACHE_DIR
    )
    print("✅ Mô hình Embedding đã sẵn sàng.")
    return embeddings

# services.py

# ... (Hàm load_llm_pipeline và load_embedding_model giữ nguyên) ...

def create_database_connection():
    """
    Tạo kết nối PostgreSQL database và SQL Database cho llama-index.
    """
    try:
        # Sử dụng DATABASE_URL từ .env file
        database_url = config.DATABASE_URL
        if not database_url:
            print("❌ Không tìm thấy DATABASE_URL trong file .env")
            return None, None
        
        # Tạo SQLAlchemy engine từ DATABASE_URL với connection pooling
        engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ Kết nối database thành công!")
        return None, engine
        
    except Exception as e:
        print(f"❌ Lỗi kết nối database: {e}")
        return None, None

def query_database_direct(engine, query_text):
    """
    Thực thi truy vấn SQL trực tiếp và trả về kết quả.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query_text))
            rows = result.fetchall()
            
            # Chuyển đổi thành list of dict
            data = []
            for row in rows:
                # Convert Row to dict
                row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
                data.append(row_dict)
            
            return data
    except Exception as e:
        print(f"❌ Lỗi thực thi query: {e}")
        return []

def get_product_info_from_db(engine, search_term):
    """
    Tìm kiếm thông tin sản phẩm từ database dựa trên từ khóa.
    """
    query = """
    SELECT 
        p.id,
        p.name,
        p.description,
        p.price,
        p.sale_price,
        p.stock,
        c.name as category_name,
        pv.sku,
        pv.size,
        pv.color,
        pv.stock as variant_stock
    FROM products p
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN product_variants pv ON p.id = pv.product_id
    WHERE LOWER(p.name) LIKE LOWER(:search1) 
       OR LOWER(p.description) LIKE LOWER(:search2)
       OR LOWER(c.name) LIKE LOWER(:search3)
       OR LOWER(pv.sku) LIKE LOWER(:search4)
    ORDER BY p.id, pv.id
    LIMIT 10
    """
    
    search_pattern = f"%{search_term}%"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), {
                'search1': search_pattern,
                'search2': search_pattern, 
                'search3': search_pattern,
                'search4': search_pattern
            })
            rows = result.fetchall()
            
            # Xử lý kết quả với group theo product_id
            products = {}
            for row in rows:
                # Convert Row to dict for easier access
                row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
                
                product_id = row_dict['id']
                if product_id not in products:
                    products[product_id] = {
                        'id': row_dict['id'],
                        'name': row_dict['name'],
                        'description': row_dict['description'],
                        'price': row_dict['price'],
                        'sale_price': row_dict['sale_price'],
                        'stock': row_dict['stock'],
                        'category': row_dict['category_name'],
                        'variants': []
                    }
                
                # Thêm variant nếu có
                if row_dict['sku']:
                    products[product_id]['variants'].append({
                        'sku': row_dict['sku'],
                        'size': row_dict['size'],
                        'color': row_dict['color'],
                        'stock': row_dict['variant_stock']
                    })
            
            return list(products.values())
            
            return products
            
    except Exception as e:
        print(f"❌ Lỗi tìm kiếm sản phẩm: {e}")
        return []

def get_order_info_from_db(engine, search_term):
    """
    Tìm kiếm thông tin đơn hàng từ database.
    """
    # Map trạng thái từ database sang tiếng Việt
    STATUS_MAP = {
        'pending': 'Chờ xác nhận',
        'confirmed': 'Đã xác nhận',
        'shipping': 'Đang giao hàng',
        'delivered': 'Đã giao hàng',
        'cancelled': 'Đã hủy'
    }
    
    query = """
    SELECT 
        o.id,
        o.order_number,
        o.full_name,
        o.phone,
        o.email,
        o.status,
        o.created_at,
        oi.product_name,
        oi.quantity,
        oi.price,
        oi.subtotal
    FROM orders o
    LEFT JOIN order_items oi ON o.id = oi.order_id
    WHERE LOWER(o.order_number) LIKE LOWER(:search1)
       OR LOWER(o.full_name) LIKE LOWER(:search2)
       OR LOWER(o.phone) LIKE LOWER(:search3)
       OR LOWER(o.email) LIKE LOWER(:search4)
    ORDER BY o.created_at DESC
    LIMIT 5
    """
    
    search_pattern = f"%{search_term}%"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), {
                'search1': search_pattern,
                'search2': search_pattern,
                'search3': search_pattern,
                'search4': search_pattern
            })
            rows = result.fetchall()
            
            # Xử lý kết quả
            orders = {}
            for row in rows:
                # Convert Row to dict for easier access
                row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
                
                order_id = row_dict['id']
                if order_id not in orders:
                    # Map trạng thái sang tiếng Việt
                    status_vi = STATUS_MAP.get(row_dict['status'].lower(), row_dict['status'])
                    
                    orders[order_id] = {
                        'id': row_dict['id'],
                        'order_number': row_dict['order_number'],
                        'customer_name': row_dict['full_name'],
                        'phone': row_dict['phone'],
                        'email': row_dict['email'],
                        'status': status_vi,
                        'created_at': row_dict['created_at'],
                        'items': []
                    }
                
                if row_dict['product_name']:  # product_name exists
                    orders[order_id]['items'].append({
                        'product_name': row_dict['product_name'],
                        'quantity': row_dict['quantity'],
                        'price': row_dict['price'],
                        'subtotal': row_dict['subtotal']
                    })
            
            return list(orders.values())
            
    except Exception as e:
        print(f"❌ Lỗi tìm kiếm đơn hàng: {e}")
        return []

def create_rag_chain(llm, embeddings):
    """
    Tự động QUÉT thư mục DATA_DIR, nạp TẤT CẢ các file (.csv, .pdf, .txt)
    và xây dựng RAG chain với tích hợp database.
    """
    print(f"Bắt đầu quét thư mục kiến thức: {config.DATA_DIR}")
    
    all_documents = [] # List để chứa tất cả tài liệu

    # --- 1. QUÉT THƯ MỤC VÀ LOAD FILE ---
    try:
        # Lấy danh sách file trong thư mục DATA_DIR
        filenames = os.listdir(config.DATA_DIR)
        
        for filename in filenames:
            filepath = os.path.join(config.DATA_DIR, filename)
            
            # --- Xử lý file CSV (Logic cũ của bạn) ---
            if filename.endswith(".csv"):
                print(f"  [CSV] Đang xử lý file: {filename}")
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    content = f"Tên: {row['product_name']}\n"
                    content += f"Loại: {row['category']}\n"
                    if row['price'] > 0:
                        content += f"Giá: {row['price']:,} VNĐ\n"
                    content += f"Mô tả: {row['description']}"
                    doc = Document(page_content=content, metadata={"source": filename})
                    all_documents.append(doc)

            # --- Xử lý file Text ---
            elif filename.endswith(".txt"):
                print(f"  [TXT] Đang xử lý file: {filename}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(page_content=content, metadata={"source": filename})
                all_documents.append(doc)

            # --- Bỏ qua PDF tạm thời để tránh lỗi dependency ---
            elif filename.endswith(".pdf"):
                print(f"  [PDF] Bỏ qua file PDF: {filename} (chưa hỗ trợ)")
            
            else:
                print(f"  [SKIP] Bỏ qua file không hỗ trợ: {filename}")

    except FileNotFoundError:
        print(f"⚠️ LỖI: Không tìm thấy thư mục {config.DATA_DIR}.")
    except Exception as e:
        print(f"⚠️ LỖI khi quét thư mục: {e}")

    # --- 2. KIỂM TRA DỮ LIỆU ---
    if not all_documents:
        print("⚠️ CẢNH BÁO: Không nạp được bất kỳ tài liệu nào. Bot sẽ không có kiến thức.")
        # Tạo một tài liệu rỗng để tránh lỗi
        all_documents = [Document(page_content="Không có kiến thức.")]

    print(all_documents[0:2]) # In 2 tài liệu đầu để kiểm tra

    # --- 3. TẠO VECTOR STORE (Như cũ) ---
    print("Khởi tạo Vector Store FAISS...")
    vector_store = FAISS.from_documents(all_documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Lấy 2 kết quả
    print("✅ Vector Store FAISS và Retriever đã sẵn sàng.")

    # --- 4. TẠO DATABASE CONNECTION ---
    sql_database, engine = create_database_connection()

    # --- 5. TẠO HYBRID RETRIEVER ---
    def hybrid_retriever(question):
        """
        Kết hợp tìm kiếm vector và database query.
        """
        question_lower = question.lower().strip()
        
        # Fast-path: Xử lý câu chào - trả về response cố định
        greetings = ['xin chào', 'hello', 'hi', 'chào', 'hey', 'chào shop', 'alo']
        if any(greeting in question_lower for greeting in greetings) and len(question) < 30:
            return [Document(
                page_content="Khách hàng chào hỏi. Trả lời: 'Dạ, chào anh/chị! Shop em bán quần áo thời trang, anh/Chị cần em tư vấn gì ạ?'",
                metadata={"source": "greeting"}
            )]
        
        # Fast-path: Xử lý câu cảm ơn
        thanks = ['cảm ơn', 'thank', 'thanks', 'cám ơn', 'cam on']
        if any(thank in question_lower for thank in thanks):
            return [Document(
                page_content="Khách hàng cảm ơn. Trả lời: 'Dạ, cảm ơn anh/chị đã ghé thăm cửa hàng, anh/Chị có cần em tư vấn thêm gì không ạ?'",
                metadata={"source": "thanks"}
            )]
        order_only_pattern = r'^ORD\d+$'
        if re.match(order_only_pattern, question.upper().strip()):
            # Chuyển sang tìm kiếm đơn hàng - cập nhật cả question và question_lower
            question = 'đơn hàng ' + question
            question_lower = question.lower()
            
        # 1. Tìm kiếm từ vector store
        vector_results = retriever.invoke(question)
        
        # 2. Tìm kiếm từ database nếu có kết nối
        db_results = []
        if engine:
            # Phát hiện loại câu hỏi và tìm kiếm phù hợp
            question_lower = question.lower()
            
            # Câu hỏi về sản phẩm
            if any(keyword in question_lower for keyword in ['sản phẩm', 'áo', 'quần', 'giá', 'mua', 'bán', 'tìm']):
                # Extract tên sản phẩm hoặc mã SKU
                # Pattern 1: Tìm SKU (có dấu gạch ngang: ATN-PREMIUM-S-BLACK)
                sku_pattern = r'\b[A-Z0-9]+-[A-Z0-9-]+\b'
                sku_match = re.search(sku_pattern, question.upper())
                
                # Pattern 2: Tìm từ khóa sau "sản phẩm", "tìm", "có"
                keyword_pattern = r'(?:sản phẩm|tìm|có|mua|bán)\s+(.+?)(?:\s+không|\s+có|\s*$)'
                keyword_match = re.search(keyword_pattern, question_lower, re.IGNORECASE)
                
                # Ưu tiên SKU, nếu không có thì dùng keyword
                if sku_match:
                    search_term = sku_match.group(0)
                elif keyword_match:
                    search_term = keyword_match.group(1).strip()
                else:
                    search_term = question
                
                print(f"🔍 DEBUG: Tìm kiếm sản phẩm với từ khóa: '{search_term}'")
                products = get_product_info_from_db(engine, search_term)
                print(f"🔍 DEBUG: Tìm thấy {len(products)} sản phẩm")
                for product in products[:3]:  # Chỉ lấy 3 sản phẩm đầu
                    print(f"🔍 DEBUG: Sản phẩm: {product['name']}, Giá: {product['price']}")
                    content = f"Sản phẩm: {product['name']}\n"
                    content += f"Danh mục: {product['category']}\n"
                    content += f"Giá gốc: {product['price']:,.0f} VNĐ\n"
                    if product['sale_price']:
                        content += f"Giá khuyến mãi: {product['sale_price']:,.0f} VNĐ\n"
                    content += f"Tồn kho: {product['stock']}\n"
                    content += f"Mô tả: {product['description']}\n"
                    if product['variants']:
                        content += "Biến thể:\n"
                        for variant in product['variants'][:2]:  # Chỉ hiển thị 2 variant đầu
                            content += f"  - SKU: {variant['sku']}, Size: {variant['size']}, Màu: {variant['color']}, Tồn kho: {variant['stock']}\n"
                    
                    db_results.append(Document(page_content=content, metadata={"source": "database_products"}))
            
            # Câu hỏi về đơn hàng
            elif any(keyword in question_lower for keyword in ['đơn hàng', 'order', 'mua', 'khách hàng']):
                # Extract mã đơn hàng nếu có (ORD...)
                order_code_match = re.search(r'ORD\d+', question.upper())
                search_term = order_code_match.group(0) if order_code_match else question
                
                print(f"🔍 DEBUG: Tìm kiếm đơn hàng với từ khóa: '{search_term}'")
                orders = get_order_info_from_db(engine, search_term)
                print(f"🔍 DEBUG: Tìm thấy {len(orders)} đơn hàng")
                for order in orders[:2]:  # Chỉ lấy 2 đơn hàng đầu
                    print(f"🔍 DEBUG: Đơn hàng {order['order_number']}, trạng thái: {order['status']}")
                    content = f"Đơn hàng: {order['order_number']}\n"
                    content += f"Khách hàng: {order['customer_name']}\n"
                    content += f"Điện thoại: {order['phone']}\n"
                    content += f"Trạng thái: {order['status']}\n"
                    content += f"Ngày tạo: {order['created_at']}\n"
                    if order['items']:
                        content += "Sản phẩm:\n"
                        for item in order['items']:
                            content += f"  - {item['product_name']}: {item['quantity']} x {item['price']:,.0f} = {item['subtotal']:,.0f} VNĐ\n"
                    
                    db_results.append(Document(page_content=content, metadata={"source": "database_orders"}))
        
        # 3. Kết hợp kết quả
        all_results = vector_results + db_results
        return all_results[:3]  # Giới hạn 3 kết quả

    # --- 6. TẠO PROMPT VÀ CHAIN ---
    rag_template = """<s>[INST] Bạn là trợ lý AI của shop thời trang. Trả lời CHÍNH XÁC dựa trên dữ liệu được cung cấp.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin từ "Nội dung" bên dưới
2. KHÔNG được tự bịa hoặc đoán thông tin
3. Nếu không có đủ thông tin: "Em không tìm thấy thông tin về [nội dung] ạ"
4. Xưng hô: tự xưng "em", gọi khách "anh/chị"
5. Kết thúc: "Anh/Chị có cần em tư vấn thêm gì không ạ?"

Nội dung (ĐỌC KỸ và SỬ DỤNG):
{context}

Câu hỏi: {question}

Hãy trả lời DỰA TRÊN Nội dung phía trên, không được tự bịa: [/INST]
"""
    rag_prompt = PromptTemplate.from_template(rag_template)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Tạo chain với cú pháp tương thích
    def enhanced_context_retriever(inputs):
        """Retrieve và format context từ hybrid retriever."""
        question = inputs if isinstance(inputs, str) else inputs.get("question", "")
        docs = hybrid_retriever(question)
        return format_docs(docs)

    # RunnableLambda already imported at top
    
    rag_chain = (
        RunnableLambda(lambda x: {"context": enhanced_context_retriever(x), "question": x})
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("✅ Pipeline RAG với database integration hoàn chỉnh đã sẵn sàng.")
    return rag_chain