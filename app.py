from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import services
import sys
import threading
import re


def handle_special_messages(message):
    """Xử lý các câu chào hỏi, cảm ơn đơn giản trước khi gửi xuống AI"""
    message_lower = message.lower().strip()

    # Câu chào
    greetings = ["xin chào", "hello", "hi", "chào", "hey", "chào shop", "alo"]
    if any(greeting in message_lower for greeting in greetings) and len(message) < 30:
        return "Dạ, chào anh/chị! Em là NaHi - nhân viên tư vấn của shop. Shop em bán quần áo thời trang, anh/Chị cần em tư vấn gì ạ?"

    # Câu cảm ơn
    thanks = ["cảm ơn", "thank", "thanks", "cám ơn", "cam on"]
    if any(thank in message_lower for thank in thanks):
        return "Dạ, em cảm ơn anh/chị đã ghé thăm cửa hàng! Nếu cần tư vấn thêm, anh/chị cứ hỏi NaHi bất cứ lúc nào nhé!"

    return None  # Không phải special message


app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(app, cors_allowed_origins="*")
print("Khởi tạo Flask server với Socket.IO...")


def clean_agent_output(text):
    """Loại bỏ thông tin kỹ thuật leak từ SQL Agent"""
    # Loại bỏ các pattern leak thông tin DB
    patterns_to_remove = [
        r"Tôi cần.*?bảng.*?\.",
        r"Để làm điều này.*?\.",
        r"truy vấn bảng.*?\.",
        r"kết hợp với bảng.*?\.",
        r"product_variants?",
        r"products?",
        r"orders?",
        r"order_items?",
        r"categories?",
        r"Thought:.*?(?=\n|$)",
        r"Action:.*?(?=\n|$)",
        r"Observation:.*?(?=\n|$)",
        r"> Entering.*?chain.*",
        r"> Finished.*?chain.*",
    ]

    cleaned = text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Loại bỏ khoảng trắng thừa
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Nếu sau khi clean không còn gì, trả về message mặc định
    if not cleaned or len(cleaned) < 10:
        return "Dạ, em đã tìm kiếm nhưng chưa tìm thấy thông tin phù hợp ạ. Anh/Chị có thể hỏi cụ thể hơn được không ạ?"

    return cleaned


print("Bắt đầu quá trình khởi tạo mô hình AI...")
try:
    llm = services.load_llm_pipeline()

    # Tạo cả RAG chain và SQL Agent
    rag_chain = services.create_rag_chain(llm)
    sql_agent = services.create_text_to_sql_agent(llm)

    print("Server đã sẵn sàng!")

except Exception as e:
    print(f"FATAL ERROR: Không thể khởi tạo mô hình. Lỗi: {e}")
    sys.exit(1)


@app.route("/ask", methods=["POST"])
def handle_ask():
    global rag_chain, sql_agent

    try:
        data = request.json
        if not data or "question" not in data:
            print("Lỗi: Request không chứa 'question'")
            return jsonify({"error": "Không tìm thấy 'question' trong JSON body."}), 400

        question = data["question"]
        use_sql_agent = data.get("use_sql_agent", True)  # Mặc định dùng SQL Agent

        print(f"\n[API] Đã nhận câu hỏi: {question}")
        print(
            f"[API] Mode: {'SQL Agent (Text-to-SQL)' if use_sql_agent else 'RAG Chain (Regex)'}"
        )

        # Kiểm tra ngôn ngữ
        if not services.is_vietnamese(question):
            print(f"[API] Từ chối - Không phải tiếng Việt: {question}")
            return jsonify(
                {
                    "answer": "Xin lỗi, em chỉ hỗ trợ trả lời bằng tiếng Việt ạ. Anh/chị vui lòng nhắn tin bằng tiếng Việt nhé!"
                }
            )

        # Kiểm tra special messages trước
        special_answer = handle_special_messages(question)
        if special_answer:
            return jsonify({"answer": special_answer})

        # Chọn mode xử lý
        if use_sql_agent and sql_agent:
            # Dùng SQL Agent (Text-to-SQL với Function Calling)
            print("[API] Sử dụng SQL Agent...")
            response = sql_agent.invoke({"input": question})
            raw_answer = (
                response.get("output", response)
                if isinstance(response, dict)
                else str(response)
            )
            # Làm sạch output trước khi trả về
            answer = clean_agent_output(raw_answer)
        else:
            # Dùng RAG Chain (regex-based cũ)
            print("[API] Sử dụng RAG Chain...")
            answer = rag_chain.invoke(question)

        print(f"[API] Đang trả lời: {answer}")

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"[API LỖI] {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Đã xảy ra lỗi server: {str(e)}"}), 500


@socketio.on("connect")
def handle_connect():
    print(f"[Socket.IO] Client đã kết nối: {request.sid}")
    emit("connected", {"message": "Đã kết nối thành công với server!"})


@socketio.on("disconnect")
def handle_disconnect():
    print(f"[Socket.IO] Client đã ngắt kết nối: {request.sid}")


@socketio.on("send_message")
def handle_send_message(data):
    global rag_chain, sql_agent

    try:
        message = data.get("message", "")
        session_id = data.get("session_id", None)
        user_id = data.get("user_id", None)

        if not message:
            emit(
                "error", {"message": "Không tìm thấy message", "session_id": session_id}
            )
            return

        print(f"\n[Socket.IO] Session {session_id} - Nhận message: {message}")

        # Kiểm tra ngôn ngữ
        if not services.is_vietnamese(message):
            print(f"[Socket.IO] Từ chối - Không phải tiếng Việt: {message}")
            emit(
                "message_response",
                {
                    "message": message,
                    "answer": "Xin lỗi, em chỉ hỗ trợ trả lời bằng tiếng Việt ạ. Anh/chị vui lòng nhắn tin bằng tiếng Việt nhé!",
                    "session_id": session_id,
                },
            )
            return

        emit(
            "processing",
            {"message": "Đang xử lý câu hỏi của bạn...", "session_id": session_id},
        )

        services.save_chat_message(session_id, "user", message, user_id)

        # Kiểm tra special messages (chào hỏi, cảm ơn) trước
        special_answer = handle_special_messages(message)
        if special_answer:
            answer = special_answer
            print(f"[Socket.IO] Phản hồi special message: {answer[:50]}...")
        # Dùng SQL Agent cho các câu hỏi thực sự
        elif sql_agent:
            print("[Socket.IO] Sử dụng SQL Agent...")
            response = sql_agent.invoke({"input": message})
            raw_answer = (
                response.get("output", response)
                if isinstance(response, dict)
                else str(response)
            )
            # Làm sạch output trước khi trả về
            answer = clean_agent_output(raw_answer)
            print(f"[DEBUG] Raw answer: {raw_answer[:100]}...")
            print(f"[DEBUG] Cleaned answer: {answer[:100]}...")
        else:
            print("[Socket.IO] SQL Agent không khả dụng, dùng RAG Chain...")
            answer = rag_chain.invoke(message)

        services.save_chat_message(session_id, "bot", answer, user_id)
        print(f"[Socket.IO] Session {session_id} - Trả lời: {answer}")

        emit(
            "message_response",
            {"message": message, "answer": answer, "session_id": session_id},
        )

    except Exception as e:
        print(f"[Socket.IO LỖI] {str(e)}")
        emit(
            "error",
            {
                "message": f"Lỗi: {str(e)}",
                "session_id": data.get("session_id", "unknown"),
            },
        )


@socketio.on("ask_question")
def handle_socket_question(data):
    global rag_chain, sql_agent

    try:
        question = data.get("question", "")
        if not question:
            emit("error", {"message": "Không tìm thấy câu hỏi"})
            return

        print(f"\n[Socket.IO] Đã nhận câu hỏi: {question}")

        # Kiểm tra ngôn ngữ
        if not services.is_vietnamese(question):
            print(f"[Socket.IO] Từ chối - Không phải tiếng Việt: {question}")
            emit(
                "answer",
                {
                    "question": question,
                    "answer": "Xin lỗi, em chỉ hỗ trợ trả lời bằng tiếng Việt ạ. Anh/chị vui lòng nhắn tin bằng tiếng Việt nhé!",
                },
            )
            return

        # Kiểm tra special messages trước
        special_answer = handle_special_messages(question)
        if special_answer:
            emit("answer", {"question": question, "answer": special_answer})
            return

        emit("processing", {"message": "Đang xử lý câu hỏi..."})

        # Dùng SQL Agent mặc định
        if sql_agent:
            print("[Socket.IO] Sử dụng SQL Agent...")
            response_data = sql_agent.invoke({"input": question})
            raw_response = (
                response_data.get("output", response_data)
                if isinstance(response_data, dict)
                else str(response_data)
            )
            # Làm sạch output trước khi trả về
            response = clean_agent_output(raw_response)
        else:
            print("[Socket.IO] SQL Agent không khả dụng, dùng RAG Chain...")
            response = rag_chain.invoke(question)

        print(f"[Socket.IO] Đang trả lời: {response}")

        emit("answer", {"question": question, "answer": response})

    except Exception as e:
        print(f"[Socket.IO LỖI] {str(e)}")
        emit("error", {"message": f"Lỗi: {str(e)}"})


if __name__ == "__main__":
    print("Bắt đầu chạy server API tại http://0.0.0.0:5000")
    print("Socket.IO đã được kích hoạt")
    print("REST API endpoint: POST http://0.0.0.0:5000/ask")
    socketio.run(
        app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
    )
