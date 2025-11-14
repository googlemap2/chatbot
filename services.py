import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login
# from langchain_community.llms import VLLM  # Bỏ vLLM
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader

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

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        return_full_text=False
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

def create_rag_chain(llm, embeddings):
    """
    Tự động QUÉT thư mục DATA_DIR, nạp TẤT CẢ các file (.csv, .pdf, .txt)
    và xây dựng RAG chain.
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

            # --- Xử lý file PDF ---
            elif filename.endswith(".pdf"):
                print(f"  [PDF] Đang xử lý file: {filename}")
                loader = PyPDFLoader(filepath)
                docs = loader.load() # Tải và tách trang
                all_documents.extend(docs) # Thêm các trang vào list

            # --- Xử lý file Text ---
            elif filename.endswith(".txt"):
                print(f"  [TXT] Đang xử lý file: {filename}")
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                all_documents.extend(docs)
            
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Lấy 3 kết quả
    print("✅ Vector Store FAISS và Retriever đã sẵn sàng.")

    # --- 4. TẠO PROMPT VÀ CHAIN (Như cũ) ---
    rag_template = """<s>[INST] Bạn là một trợ lý AI hữu ích, chuyên nghiệp và chỉ trả lời bằng tiếng Việt.
Bạn phải trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào "Nội dung" được cung cấp.
Nếu "Nội dung" không chứa thông tin để trả lời, hãy nói: "Tôi không tìm thấy thông tin trong tài liệu."
Không được bịa đặt thông tin.

Nội dung:
{context}

Câu hỏi: {question} [/INST]
"""
    rag_prompt = PromptTemplate.from_template(rag_template)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("✅ Pipeline RAG (tự động quét) hoàn chỉnh đã sẵn sàng.")
    return rag_chain