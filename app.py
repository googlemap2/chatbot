# app.py
from flask import Flask, request, jsonify
import services
import sys

# --- Khởi tạo ứng dụng ---
app = Flask(__name__)
print("Khởi tạo Flask server...")

# --- Load mô hình MỘT LẦN KHI BẮT ĐẦU ---
# Đây là phần quan trọng. Mô hình được load 1 lần khi server chạy,
# không phải load lại mỗi lần call API.
print("Bắt đầu quá trình khởi tạo mô hình AI...")
try:
    services.login_huggingface()
    
    # 1. Load LLM
    llm = services.load_llm_pipeline()
    
    # 2. Load Embeddings
    embeddings = services.load_embedding_model()
    
    # 3. Tạo RAG chain
    rag_chain = services.create_rag_chain(llm, embeddings)
    
    print("🎉🎉🎉 Server đã sẵn sàng nhận request! 🎉🎉🎉")

except Exception as e:
    print(f"FATAL ERROR: Không thể khởi tạo mô hình. Lỗi: {e}")
    sys.exit(1) # Thoát nếu không load được model

# --- Định nghĩa API Endpoint ---
@app.route("/ask", methods=["POST"])
def handle_ask():
    """
    Endpoint này nhận câu hỏi (JSON) và trả về câu trả lời (JSON).
    """
    global rag_chain # Sử dụng chain đã được load toàn cục
    
    try:
        data = request.json
        if not data or "question" not in data:
            print("Lỗi: Request không chứa 'question'")
            return jsonify({"error": "Không tìm thấy 'question' trong JSON body."}), 400

        question = data["question"]
        print(f"\n[API] Đã nhận câu hỏi: {question}")
        
        # Gọi RAG chain
        response = rag_chain.invoke(question)
        
        print(f"[API] Đang trả lời: {response}")
        
        # Trả về kết quả
        return jsonify({"answer": response})

    except Exception as e:
        print(f"[API LỖI] {str(e)}")
        return jsonify({"error": f"Đã xảy ra lỗi server: {str(e)}"}), 500

# --- Chạy Server ---
if __name__ == "__main__":
    print("🚀 Bắt đầu chạy server API tại http://0.0.0.0:5000")
    print("Sử dụng endpoint: POST http://0.0.0.0:5000/ask")
    # debug=False là quan trọng, nếu debug=True, nó sẽ load model 2 lần
    app.run(host='0.0.0.0', port=5000, debug=False)