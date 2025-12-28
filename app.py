from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import services
import sys
import time
import re


def handle_greeting(message):
    message_lower = message.lower().strip()

    words = message_lower.split()

    if len(message) < 30:
        if any(greeting in message_lower for greeting in ["xin chào", "chào shop"]):
            return "Dạ, chào anh/chị! Em là NaHi - nhân viên tư vấn của shop. Shop em bán quần áo thời trang, anh/Chị cần em tư vấn gì ạ?"

        if words and words[0] in ["hello", "hi", "chào", "hey", "alo"]:
            return "Dạ, chào anh/chị! Em là NaHi - nhân viên tư vấn của shop. Shop em bán quần áo thời trang, anh/Chị cần em tư vấn gì ạ?"

    return None


def handle_special_messages(message):
    greeting_response = handle_greeting(message)
    if greeting_response:
        return greeting_response
    return None


app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(app, cors_allowed_origins="*")
print("Khởi tạo Flask server với Socket.IO...")


def clean_agent_output(text):
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

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned or len(cleaned) < 10:
        return "Dạ, em đã tìm kiếm nhưng chưa tìm thấy thông tin phù hợp ạ. Anh/Chị có thể hỏi cụ thể hơn được không ạ?"

    return cleaned


print("Bắt đầu quá trình khởi tạo mô hình AI...")
try:
    llm = services.load_llm_pipeline()

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
        use_sql_agent = data.get("use_sql_agent", True)
        print(f"\n[API] Đã nhận câu hỏi: {question}")
        print(
            f"[API] Mode: {'SQL Agent (Text-to-SQL)' if use_sql_agent else 'RAG Chain (Regex)'}"
        )

        is_sensitive, detected_word = services.filter_sensitive_content(question)
        if is_sensitive:
            print(f"[API] Từ chối - Phát hiện từ nhạy cảm: {detected_word}")
            return jsonify(
                {
                    "answer": "Xin lỗi, trong câu của bạn có chứa từ ngữ không phù hợp. Vui lòng sử dụng ngôn từ lịch sự để tôi có thể hỗ trợ bạn tốt hơn ạ."
                }
            )

        special_answer = handle_special_messages(question)
        if special_answer:
            return jsonify({"answer": special_answer})

        if not services.is_vietnamese(question):
            print(f"[API] Từ chối - Không phải tiếng Việt: {question}")
            return jsonify(
                {
                    "answer": "Xin lỗi, em chỉ hỗ trợ trả lời bằng tiếng Việt ạ. Anh/chị vui lòng nhắn tin bằng tiếng Việt nhé!"
                }
            )

        if use_sql_agent and sql_agent:
            print("[API] Sử dụng SQL Agent...")
            response = sql_agent.invoke({"input": question})

            print(f"[DEBUG] Full Agent Response: {response}")
            if isinstance(response, dict):
                if "intermediate_steps" in response:
                    print(
                        f"[DEBUG] Intermediate Steps: {response['intermediate_steps']}"
                    )
                    for i, step in enumerate(response["intermediate_steps"]):
                        print(f"[DEBUG] Step {i+1}: {step}")

            raw_answer = (
                response.get("output", response)
                if isinstance(response, dict)
                else str(response)
            )
            answer = clean_agent_output(raw_answer)
        else:
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

        is_sensitive, detected_word = services.filter_sensitive_content(message)
        services.save_chat_message(session_id, "user", message, user_id)
        if is_sensitive:
            print(f"[Socket.IO] Từ chối - Phát hiện từ nhạy cảm: {detected_word}")
            answer = "Xin lỗi, trong câu của bạn có chứa từ ngữ không phù hợp. Vui lòng sử dụng ngôn từ lịch sự để tôi có thể hỗ trợ bạn tốt hơn ạ."
            emit(
                "message_response",
                {
                    "message": message,
                    "answer": answer,
                    "session_id": session_id,
                },
            )
            services.save_chat_message(session_id, "bot", answer, user_id)
            return

        special_answer = handle_special_messages(message)
        if special_answer:
            answer = special_answer
            emit(
                "message_response",
                {
                    "message": message,
                    "answer": answer,
                    "session_id": session_id,
                },
            )
            services.save_chat_message(session_id, "bot", answer, user_id)
            return

        if not services.is_vietnamese(message):
            print(f"[Socket.IO] Từ chối - Không phải tiếng Việt: {message}")
            answer = "Xin lỗi, em chỉ hỗ trợ trả lời bằng tiếng Việt ạ. Anh/chị vui lòng nhắn tin bằng tiếng Việt nhé!"
            emit(
                "message_response",
                {
                    "message": message,
                    "answer": answer,
                    "session_id": session_id,
                },
            )
            services.save_chat_message(session_id, "bot", answer, user_id)
            return

        emit(
            "processing",
            {"message": "Đang xử lý câu hỏi của bạn...", "session_id": session_id},
        )

        if sql_agent:
            print("[Socket.IO] Sử dụng SQL Agent...")
            try:
                print(f"[DEBUG] Invoking agent with input: {message}")

                time_agent_start = time.time()
                response = sql_agent.invoke({"input": message})
                time_agent_end = time.time()
                time_agent_total = time_agent_end - time_agent_start

                print(f"[TIMING] Agent invoke: {time_agent_total:.3f}s")
                print(f"[DEBUG] Agent invoke completed successfully")
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Agent invoke failed: {error_msg}")

                if "503" in error_msg or "overloaded" in error_msg.lower():
                    print(f"[ERROR] Gemini API quá tải (503)")
                    answer = "Dạ, hệ thống AI đang quá tải. Anh/Chị vui lòng thử lại sau ít phút nhé!"
                elif "timeout" in error_msg.lower():
                    print(f"[ERROR] Request timeout")
                    answer = "Dạ, câu hỏi hơi phức tạp và mất thời gian xử lý. Anh/Chị có thể hỏi đơn giản hơn không ạ?"
                elif "max iterations" in error_msg.lower():
                    print(f"[ERROR] Agent vượt quá số lần lặp")
                    answer = "Dạ, em chưa tìm được câu trả lời phù hợp. Anh/Chị có thể hỏi cụ thể hơn không ạ?"
                else:
                    answer = "Dạ, em gặp lỗi khi xử lý câu hỏi. Anh/Chị thử hỏi lại được không ạ?"

                import traceback

                traceback.print_exc()

                services.save_chat_message(session_id, "bot", answer, user_id)
                emit(
                    "message_response",
                    {"message": message, "answer": answer, "session_id": session_id},
                )
                return

            print(f"[DEBUG] Full Agent Response: {response}")
            print(f"[DEBUG] Response type: {type(response)}")
            print(
                f"[DEBUG] Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}"
            )

            if (
                isinstance(response, dict)
                and "intermediate_steps" in response
                and len(response["intermediate_steps"]) > 0
            ):
                has_steps = True
                print(f"[DEBUG] Số bước xử lý: {len(response['intermediate_steps'])}")
                for i, step in enumerate(response["intermediate_steps"]):
                    action, observation = step
                    tool_name = action.tool if hasattr(action, "tool") else "unknown"

                    print(f"[DEBUG] ===== Step {i+1} =====")
                    print(f"[DEBUG] Action: {tool_name}")
                    print(
                        f"[DEBUG] Tool Input: {action.tool_input if hasattr(action, 'tool_input') else ''}"
                    )
                    print(
                        f"[DEBUG] Observation: {observation[:200]}..."
                        if len(str(observation)) > 200
                        else f"[DEBUG] Observation: {observation}"
                    )

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

        special_answer = handle_special_messages(question)
        if special_answer:
            emit("answer", {"question": question, "answer": special_answer})
            return

        emit("processing", {"message": "Đang xử lý câu hỏi..."})

        if sql_agent:
            print("[Socket.IO] Sử dụng SQL Agent...")
            response_data = sql_agent.invoke({"input": question})
            raw_response = (
                response_data.get("output", response_data)
                if isinstance(response_data, dict)
                else str(response_data)
            )
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
