from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import requests
import json 

app = Flask(__name__)
CORS(app)

def generate_chat_stream(user_message):
    """
    This function talks to Ollama and streams the response.
    """
    print(f"[Backend] Received message: {user_message}")
    print("[Backend] Connecting to Ollama...")
    
    try:
        ollama_response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'knot',
                'prompt': user_message,
                'stream': True
            },
            stream=True
        )

        ollama_response.raise_for_status()
        print("[Backend] Ollama connection successful. Starting stream...")

        for line in ollama_response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    
                    if 'response' in chunk:
                        yield chunk['response']
                        
                except json.JSONDecodeError as e:
                    print(f"[Backend] Warning: Could not decode JSON line: {line} - {e}")

        print("[Backend] Stream finished.")

    except requests.exceptions.RequestException as e:
        print(f"[Backend] CRITICAL ERROR: {e}")
        yield f"Error: Could not connect to Ollama. Is it running?"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    return Response(generate_chat_stream(user_message), mimetype='text/plain')

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    print("Starting server at http://localhost:5001")
    app.run(port=5001, debug=True)