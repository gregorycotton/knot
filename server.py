import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from llama_cpp import Llama

model_state = {"llm": None, "path": None}

class InferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Load model
        if self.path == '/load_model':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            
            path = data.get('model_path')
            if not os.path.exists(path):
                self.send_error(404, "Model not found")
                return

            print(f"Loading: {path}...")
            # Clean old model
            if model_state["llm"]:
                del model_state["llm"]
            
            # Load new model
            model_state["llm"] = Llama(
                model_path=path,
                n_gpu_layers=-1,
                n_ctx=data.get('n_ctx', 2048),
                verbose=False
            )
            model_state["path"] = path
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        # Chat completions
        elif self.path == '/chat/completions':
            if not model_state["llm"]:
                self.send_error(400, "No model loaded")
                return

            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()

            stream = model_state["llm"].create_chat_completion(
                messages=data['messages'],
                temperature=data.get('temperature', 0.7),
                stream=True
            )

            for chunk in stream:
                if 'content' in chunk['choices'][0]['delta']:
                    text = chunk['choices'][0]['delta']['content']
                    self.wfile.write(text.encode('utf-8'))
                    self.wfile.flush()

if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8000), InferenceHandler)
    print("Minimal Inference Engine running on :8000")
    server.serve_forever()