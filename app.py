from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import services
import sys
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")
print("Khởi tạo Flask server với Socket.IO...")

print("Bắt đầu quá trình khởi tạo mô hình AI...")
try:
    llm = services.load_llm_pipeline()
    
    rag_chain = services.create_rag_chain(llm)
    
    print("Server đã sẵn sàng!")

except Exception as e:
    print(f"FATAL ERROR: Không thể khởi tạo mô hình. Lỗi: {e}")
    sys.exit(1)

@app.route("/ask", methods=["POST"])
def handle_ask():
    global rag_chain
    
    try:
        data = request.json
        if not data or "question" not in data:
            print("Lỗi: Request không chứa 'question'")
            return jsonify({"error": "Không tìm thấy 'question' trong JSON body."}), 400

        question = data["question"]
        print(f"\n[API] Đã nhận câu hỏi: {question}")
        
        response = rag_chain.invoke(question)
        
        print(f"[API] Đang trả lời: {response}")
        
        return jsonify({"answer": response})

    except Exception as e:
        print(f"[API LỖI] {str(e)}")
        return jsonify({"error": f"Đã xảy ra lỗi server: {str(e)}"}), 500

@socketio.on('connect')
def handle_connect():
    print(f"[Socket.IO] Client đã kết nối: {request.sid}")
    emit('connected', {'message': 'Đã kết nối thành công với server!'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[Socket.IO] Client đã ngắt kết nối: {request.sid}")

@socketio.on('send_message')
def handle_send_message(data):
    global rag_chain
    
    try:
        message = data.get('message', '')
        session_id = data.get('session_id', None)
        user_id = data.get('user_id', None)
        
        if not message:
            emit('error', {'message': 'Không tìm thấy message', 'session_id': session_id})
            return
        
        print(f"\n[Socket.IO] Session {session_id} - Nhận message: {message}")
        
        emit('processing', {
            'message': 'Đang xử lý câu hỏi của bạn...',
            'session_id': session_id
        })
        
        services.save_chat_message(session_id, 'user', message, user_id)
        answer = rag_chain.invoke(message)
        services.save_chat_message(session_id, 'bot', answer, user_id)
        print(f"[Socket.IO] Session {session_id} - Trả lời: {answer}")
                
        emit('message_response', {
            'message': message,
            'answer': answer,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"[Socket.IO LỖI] {str(e)}")
        emit('error', {
            'message': f'Lỗi: {str(e)}',
            'session_id': data.get('session_id', 'unknown')
        })

@socketio.on('ask_question')
def handle_socket_question(data):
    global rag_chain
    
    try:
        question = data.get('question', '')
        if not question:
            emit('error', {'message': 'Không tìm thấy câu hỏi'})
            return
        
        print(f"\n[Socket.IO] Đã nhận câu hỏi: {question}")
        
        emit('processing', {'message': 'Đang xử lý câu hỏi...'})
        
        response = rag_chain.invoke(question)
        
        print(f"[Socket.IO] Đang trả lời: {response}")
        
        emit('answer', {'question': question, 'answer': response})
        
    except Exception as e:
        print(f"[Socket.IO LỖI] {str(e)}")
        emit('error', {'message': f'Lỗi: {str(e)}'})

if __name__ == "__main__":
    print("Bắt đầu chạy server API tại http://0.0.0.0:5000")
    print("Socket.IO đã được kích hoạt")
    print("REST API endpoint: POST http://0.0.0.0:5000/ask")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)