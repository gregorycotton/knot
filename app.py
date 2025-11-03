from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)

CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        ollama_response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'knot',
                'prompt': user_message,
                'stream': False
            }
        )
        
        ollama_response.raise_for_status()

        ollama_data = ollama_response.json()
        
        return jsonify({'response': ollama_data['response']})

    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    print("Starting server at http://localhost:5001")
    app.run(port=5001, debug=True)