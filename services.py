import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, text
import re
import config

def load_llm_pipeline():
    """
    Tải Gemini API - nhanh, miễn phí, không cần GPU.
    """
    print(f"Bắt đầu kết nối Gemini API: {config.LLM_MODEL_NAME}")
    
    if not config.GOOGLE_API_KEY:
        raise ValueError("❌ Chưa set GOOGLE_API_KEY trong file .env")
    
    # Configure Gemini
    genai.configure(api_key=config.GOOGLE_API_KEY)
    
    # Tạo LangChain Gemini LLM với cấu hình đầy đủ
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.8,
        max_output_tokens=8192,  # Tăng lên để đủ chỗ cho reasoning + response
        convert_system_message_to_human=True,
        top_p=0.95,
        top_k=40
    )
    
    print("✅ Kết nối Gemini API thành công!")
    return llm

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

def create_rag_chain(llm):
    """
    Tạo RAG chain chỉ sử dụng database, không đọc file.
    """
    print("Khởi tạo RAG chain với database integration...")

    # TẠO DATABASE CONNECTION
    sql_database, engine = create_database_connection()

    # TẠO DATABASE-ONLY RETRIEVER
    def hybrid_retriever(question):
        """
        Chỉ sử dụng database query, không dùng vector search.
        """
        question_lower = question.lower().strip()
        
        # Fast-path: Xử lý câu chào
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
        
        # Xử lý mã đơn hàng đơn lẻ
        order_only_pattern = r'^ORD\d+$'
        if re.match(order_only_pattern, question.upper().strip()):
            question = 'đơn hàng ' + question
            question_lower = question.lower()
        
        # Tìm kiếm từ database
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
        
        # Trả về kết quả từ database
        return db_results[:3] if db_results else [Document(
            page_content="",
            metadata={"source": "empty"}
        )]

    # TẠO PROMPT VÀ CHAIN
    rag_template = """Bạn là nhân viên tư vấn của shop thời trang. Hãy đọc kỹ thông tin bên dưới và trả lời câu hỏi.

THÔNG TIN SẢN PHẨM/ĐƠN HÀNG:
{context}

CÂUH HỎI: {question}

CÁCH TRẢ LỜI:
1. Đọc kỹ thông tin ở phần "THÔNG TIN SẢN PHẨM/ĐƠN HÀNG"
2. Trả lời dựa trên thông tin đó
3. Xưng "em", gọi khách "anh/chị"
4. Kết thúc: "Anh/Chị có cần em tư vấn thêm gì không ạ?"

HÃY TRẢ LỜI NGAY BÂY GIỜ:"""
    rag_prompt = PromptTemplate.from_template(rag_template)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Tạo chain với cú pháp tương thích
    def enhanced_context_retriever(inputs):
        """Retrieve và format context từ hybrid retriever."""
        question = inputs if isinstance(inputs, str) else inputs.get("question", "")
        docs = hybrid_retriever(question)
        formatted = format_docs(docs)
        print(f"📝 DEBUG Context gửi cho Gemini:\n{formatted[:500]}...")
        return formatted

    # Custom wrapper để log response từ LLM
    def debug_llm_call(prompt_value):
        """Gọi LLM và log response."""
        try:
            print(f"🤖 DEBUG Prompt gửi cho LLM:\n{str(prompt_value)[:300]}...")
            response = llm.invoke(prompt_value)
            print(f"✅ DEBUG Response type: {type(response)}")
            print(f"✅ DEBUG Response obj: {response}")
            if hasattr(response, 'content'):
                print(f"✅ DEBUG Response.content length: {len(response.content)}")
                print(f"✅ DEBUG Response.content: '{response.content}'")
            return response
        except Exception as e:
            print(f"❌ ERROR calling LLM: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # RunnableLambda already imported at top
    rag_chain = (
        RunnableLambda(lambda x: {"context": enhanced_context_retriever(x), "question": x})
        | rag_prompt
        | RunnableLambda(debug_llm_call)
        | StrOutputParser()
    )
    print("✅ Pipeline RAG với database integration hoàn chỉnh đã sẵn sàng.")
    return rag_chain