# app.py
from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import services
import sys
import threading

# --- Khởi tạo ứng dụng ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")
print("Khởi tạo Flask server với Socket.IO...")

# --- Load mô hình MỘT LẦN KHI BẮT ĐẦU ---
print("Bắt đầu quá trình khởi tạo mô hình AI...")
try:
    # Load LLM
    llm = services.load_llm_pipeline()
    
    # Tạo RAG chain
    rag_chain = services.create_rag_chain(llm)
    
    print("🎉🎉🎉 Server đã sẵn sàng nhận request! 🎉🎉🎉")

except Exception as e:
    print(f"FATAL ERROR: Không thể khởi tạo mô hình. Lỗi: {e}")
    sys.exit(1)

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

# --- Socket.IO Events ---
@socketio.on('connect')
def handle_connect():
    """Xử lý khi client kết nối"""
    print(f"[Socket.IO] Client đã kết nối: {request.sid}")
    emit('connected', {'message': 'Đã kết nối thành công với server!'})

@socketio.on('disconnect')
def handle_disconnect():
    """Xử lý khi client ngắt kết nối"""
    print(f"[Socket.IO] Client đã ngắt kết nối: {request.sid}")

@socketio.on('send_message')
def handle_send_message(data):
    """
    Xử lý event 'send_message' từ client
    Format: {"message": "hello", "session_id": "session_xxx"}
    """
    global rag_chain
    
    try:
        message = data.get('message', '')
        session_id = data.get('session_id', 'unknown')
        
        if not message:
            emit('error', {'message': 'Không tìm thấy message', 'session_id': session_id})
            return
        
        print(f"\n[Socket.IO] Session {session_id} - Nhận message: {message}")
        
        # Gửi trạng thái đang xử lý
        emit('processing', {
            'message': 'Đang xử lý câu hỏi của bạn...',
            'session_id': session_id
        })
        
        # Gọi RAG chain
        response = rag_chain.invoke(message)
        
        print(f"[Socket.IO] Session {session_id} - Trả lời: {response}")
        
        # Push câu trả lời về client
        emit('message_response', {
            'message': message,
            'answer': response,
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
    """
    Xử lý câu hỏi qua Socket.IO và push response theo thời gian thực
    (Giữ lại để tương thích với client cũ)
    """
    global rag_chain
    
    try:
        question = data.get('question', '')
        if not question:
            emit('error', {'message': 'Không tìm thấy câu hỏi'})
            return
        
        print(f"\n[Socket.IO] Đã nhận câu hỏi: {question}")
        
        # Gửi trạng thái đang xử lý
        emit('processing', {'message': 'Đang xử lý câu hỏi của bạn...'})
        
        # Gọi RAG chain
        response = rag_chain.invoke(question)
        
        print(f"[Socket.IO] Đang trả lời: {response}")
        
        # Push câu trả lời về client
        emit('answer', {'question': question, 'answer': response})
        
    except Exception as e:
        print(f"[Socket.IO LỖI] {str(e)}")
        emit('error', {'message': f'Lỗi: {str(e)}'})

@app.route('/')
def index():
    """Trang demo client Socket.IO"""
    return render_template_string(CLIENT_HTML)

# --- HTML Client Demo ---
CLIENT_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chatbot với Socket.IO</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        #messages {
            border: 1px solid #ccc;
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
        }
        .bot-message {
            background-color: #f1f8e9;
        }
        .status-message {
            background-color: #fff3e0;
            font-style: italic;
            color: #666;
        }
        .error-message {
            background-color: #ffebee;
            color: #c62828;
        }
        #input-container {
            display: flex;
            gap: 10px;
        }
        #question-input {
            flex: 1;
            padding: 10px;
            font-size: 16px;
        }
        #send-button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        #send-button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        #status {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .connected {
            background-color: #c8e6c9;
            color: #2e7d32;
        }
        .disconnected {
            background-color: #ffcdd2;
            color: #c62828;
        }
    </style>
</head>
<body>
    <h1>🤖 Chatbot với Socket.IO Push</h1>
    <div id="status" class="disconnected">Chưa kết nối</div>
    <div id="messages"></div>
    <div id="input-container">
        <input type="text" id="question-input" placeholder="Nhập câu hỏi của bạn..." disabled>
        <button id="send-button" disabled>Gửi</button>
    </div>

    <script>
        const socket = io();
        const messagesDiv = document.getElementById('messages');
        const statusDiv = document.getElementById('status');
        const questionInput = document.getElementById('question-input');
        const sendButton = document.getElementById('send-button');

        function addMessage(text, className) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + className;
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        socket.on('connect', () => {
            console.log('Đã kết nối với server');
            statusDiv.textContent = 'Đã kết nối ✓';
            statusDiv.className = 'connected';
            questionInput.disabled = false;
            sendButton.disabled = false;
        });

        socket.on('disconnect', () => {
            console.log('Đã ngắt kết nối với server');
            statusDiv.textContent = 'Mất kết nối ✗';
            statusDiv.className = 'disconnected';
            questionInput.disabled = true;
            sendButton.disabled = true;
            addMessage('Đã mất kết nối với server', 'error-message');
        });

        socket.on('connected', (data) => {
            addMessage(data.message, 'status-message');
        });

        socket.on('processing', (data) => {
            addMessage(data.message, 'status-message');
        });

        socket.on('answer', (data) => {
            addMessage('Bot: ' + data.answer, 'bot-message');
            sendButton.disabled = false;
            questionInput.value = '';
            questionInput.focus();
        });

        socket.on('error', (data) => {
            addMessage('Lỗi: ' + data.message, 'error-message');
            sendButton.disabled = false;
        });

        function sendQuestion() {
            const question = questionInput.value.trim();
            if (question) {
                addMessage('Bạn: ' + question, 'user-message');
                socket.emit('ask_question', { question: question });
                sendButton.disabled = true;
            }
        }

        sendButton.addEventListener('click', sendQuestion);

        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendQuestion();
            }
        });
    </script>
</body>
</html>
'''

# --- Chạy Server ---
if __name__ == "__main__":
    print("🚀 Bắt đầu chạy server API tại http://0.0.0.0:5000")
    print("📡 Socket.IO đã được kích hoạt")
    print("🌐 Mở trình duyệt tại http://localhost:5000 để test client")
    print("📮 REST API endpoint: POST http://0.0.0.0:5000/ask")
    # Sử dụng socketio.run thay vì app.run
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)