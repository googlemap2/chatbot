# services.py
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
from langchain_community.llms import VLLM

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
    Tải mô hình LLM (AWQ 4-bit) bằng vLLM (Siêu tốc).
    """
    QUANTIZED_MODEL_ID = "dangvansam/Vistral-7B-Chat-awq"
    print(f"Bắt đầu tải mô hình: {QUANTIZED_MODEL_ID} (chế độ vLLM + AWQ)")

    # (Bạn có thể thêm cache_dir="/root/chatbot/models" vào đây nếu muốn)
    
    llm = VLLM(
        model=QUANTIZED_MODEL_ID,
        download_dir=config.MODEL_CACHE_DIR,
        quantization="awq",
        dtype="float16", # Dùng float16 cho Tesla T4
        tensor_parallel_size=1,
        max_new_tokens=512,
        temperature=0.1,
        gpu_memory_utilization=0.50
    )
    
    print("✅ Tải mô hình vLLM + AWQ thành công.")
    # vLLM đã là một object LLM của LangChain, không cần pipeline
    return llm

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

def create_rag_chain(llm, embeddings):
    """
    Tải dữ liệu kiến thức và xây dựng RAG chain.
    """
    # 1. Tải tài liệu (Bạn có thể thay bằng code đọc file CSV)
    print("Tải tài liệu kiến thức...")
    documents = [
        Document(page_content="RAG (Retrieval-Augmented Generation) là một kỹ thuật cho phép LLM truy cập kiến thức bên ngoài."),
        Document(page_content="Microsoft phát triển mô hình Phi-3-mini."),
        Document(page_content="FAISS là một thư viện của Facebook AI để tìm kiếm tương đồng hiệu suất cao."),
        Document(page_content="Chính sách đổi trả hàng là 30 ngày kể từ ngày mua hàng.")
    ]
    # (Đây là nơi bạn thêm code đọc file products.csv)

    # 2. Tạo Vector Store
    print("Khởi tạo Vector Store FAISS...")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    print("✅ Vector Store FAISS và Retriever đã sẵn sàng.")

    # 3. Định nghĩa Prompt
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
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. Xây dựng Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("✅ Pipeline RAG hoàn chỉnh đã sẵn sàng.")
    return rag_chain