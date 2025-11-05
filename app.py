from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import requests
import json
import sqlite3
import os
import uuid 

app = Flask(__name__)
CORS(app)

# --- Database Setup ---
DB_DIR = "convo"
DB_FILE = os.path.join(DB_DIR, "history.db")

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""
CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);
"""

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    conn = get_db_connection()
    convos = conn.execute(
        'SELECT id, title FROM conversations ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in convos])

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    new_id = str(uuid.uuid4())
    new_title = "New Conversation" 
    
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO conversations (id, title) VALUES (?, ?)',
        (new_id, new_title)
    )
    conn.commit()
    conn.close()
    
    return jsonify({'id': new_id, 'title': new_title}), 201

@app.route('/api/conversations/<convo_id>', methods=['GET'])
def get_conversation_messages(convo_id):
    conn = get_db_connection()
    messages = conn.execute(
        'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC',
        (convo_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in messages])

def generate_chat_stream(conversation_id, messages_history):
    print(f"[Backend] Streaming for convo: {conversation_id}")
    full_response = ""
    
    try:
        ollama_response = requests.post(
            'http://localhost:11434/api/chat', 
            json={
                'model': 'knot',
                'messages': messages_history, 
                'stream': True
            },
            stream=True
        )
        ollama_response.raise_for_status()

        for line in ollama_response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'message' in chunk and 'content' in chunk['message']:
                    chunk_text = chunk['message']['content']
                    full_response += chunk_text
                    yield chunk_text
                if chunk.get('done'):
                    break
                    
    except requests.exceptions.RequestException as e:
        print(f"[Backend] CRITICAL ERROR: {e}")
        yield f"Error: Could not connect to Ollama."
        return 

    print(f"[Backend] Stream finished. Saving to DB.")
    try:
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)',
            (str(uuid.uuid4()), conversation_id, 'assistant', full_response)
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[Backend] DB Save Error: {e}")


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    conversation_id = data.get('conversation_id')

    if not user_message or not conversation_id:
        return jsonify({'error': 'Missing message or conversation_id'}), 400

    try:
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)',
            (str(uuid.uuid4()), conversation_id, 'user', user_message)
        )
        conn.commit()
        
        messages_rows = conn.execute(
            'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC',
            (conversation_id,)
        ).fetchall()
        conn.close()
        
        messages_history = [{'role': row['role'], 'content': row['content']} for row in messages_rows]
        
        return Response(generate_chat_stream(conversation_id, messages_history), mimetype='text/plain')
        
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500

# --- RENAME ENDPOINT ---
@app.route('/api/conversations/<convo_id>', methods=['PUT'])
def rename_conversation(convo_id):
    data = request.json
    new_title = data.get('title')
    
    if not new_title:
        return jsonify({'error': 'New title is required'}), 400
        
    try:
        conn = get_db_connection()
        conn.execute(
            'UPDATE conversations SET title = ? WHERE id = ?',
            (new_title, convo_id)
        )
        conn.commit()
        conn.close()
        return jsonify({'id': convo_id, 'title': new_title}), 200
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500

# --- DELETE ENDPOINT ---
@app.route('/api/conversations/<convo_id>', methods=['DELETE'])
def delete_conversation(convo_id):
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM conversations WHERE id = ?', (convo_id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Conversation deleted'}), 200
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    conn = get_db_connection()
    conn.execute(CREATE_CONVERSATIONS_TABLE) 
    conn.execute(CREATE_MESSAGES_TABLE) 
    conn.commit()
    conn.close()
    
    print("Starting server at http://localhost:5001")
    app.run(port=5001, debug=True)