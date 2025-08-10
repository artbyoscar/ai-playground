from flask import Flask, render_template_string, request, jsonify
from src.core.edgemind import EdgeMind
import json

app = Flask(__name__)
em = EdgeMind(verbose=False)

# Store chat history
chat_history = []

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>EdgeMind Local AI</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .chat { 
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            height: 400px; 
            overflow-y: auto; 
            padding: 20px;
            margin: 20px 0;
            background: #f9f9f9;
        }
        .message { 
            margin: 15px 0; 
            padding: 12px 15px;
            border-radius: 10px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
            text-align: right;
        }
        .ai { 
            background: #fff;
            border: 1px solid #e0e0e0;
            margin-right: 20%;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        input { 
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        button { 
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        button:hover {
            transform: scale(1.05);
        }
        .status {
            text-align: center;
            color: #666;
            margin-top: 10px;
            font-size: 14px;
        }
        .typing {
            display: none;
            color: #666;
            font-style: italic;
            margin: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 EdgeMind Local AI</h1>
        <div class="status">Running locally on your machine • 100% Private</div>
        <div class="chat" id="chat"></div>
        <div class="typing" id="typing">AI is thinking...</div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Ask anything..." autofocus>
            <button onclick="send()">Send</button>
        </div>
    </div>
    
    <script>
        function send() {
            const input = document.getElementById('input');
            const chat = document.getElementById('chat');
            const typing = document.getElementById('typing');
            const msg = input.value.trim();
            
            if (!msg) return;
            
            // Add user message
            chat.innerHTML += '<div class="message user">' + msg + '</div>';
            input.value = '';
            chat.scrollTop = chat.scrollHeight;
            
            // Show typing indicator
            typing.style.display = 'block';
            
            // Send to backend
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: msg})
            })
            .then(r => r.json())
            .then(data => {
                typing.style.display = 'none';
                chat.innerHTML += '<div class="message ai">' + data.response + '</div>';
                chat.scrollTop = chat.scrollHeight;
            })
            .catch(err => {
                typing.style.display = 'none';
                chat.innerHTML += '<div class="message ai">Error: ' + err + '</div>';
            });
        }
        
        document.getElementById('input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') send();
        });
        
        // Focus input on load
        window.onload = () => document.getElementById('input').focus();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    
    # Generate response
    response = em.generate(message, max_tokens=150)
    
    # Store in history
    chat_history.append({'user': message, 'ai': response})
    
    return jsonify({'response': response})

@app.route('/history')
def history():
    return jsonify(chat_history)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 EdgeMind Web UI")
    print("="*60)
    print("📍 Starting server...")
    print("🔗 Open http://localhost:5000 in your browser")
    print("💡 Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')