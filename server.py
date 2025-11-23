import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from llama_cpp import Llama

# Global state
state = {
    "main": {"llm": None, "path": None},
    "utility": {"llm": None, "path": None}
}

class InferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Determine if target model is main or utility
        target = "main"
        if "utility" in self.path:
            target = "utility"

        if self.path in ['/load_model', '/load_utility']:
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            path = data.get('model_path')
            n_ctx = data.get('n_ctx', 2048)

            if not os.path.exists(path):
                self.send_error(404, "Model not found")
                return

            print(f"Loading [{target.upper()}]: {os.path.basename(path)} ...", flush=True)
            
            if state[target]["llm"]:
                del state[target]["llm"]
                state[target]["llm"] = None

            try:
                state[target]["llm"] = Llama(
                    model_path=path,
                    n_gpu_layers=-1,
                    n_ctx=n_ctx,
                    verbose=False
                )
                state[target]["path"] = path
                
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                print(f"Load Error: {e}", flush=True)
                self.send_error(500, str(e))

        elif self.path in ['/chat/completions', '/utility/completions']:
            llm = state[target]["llm"]
            
            if not llm:
                self.send_error(400, f"No {target} model loaded.")
                return

            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            
            should_clean = data.get('clean_response', False)

            self.send_response(200)
            
            if not should_clean:
                # Streaming
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                stream = llm.create_chat_completion(
                    messages=data['messages'],
                    temperature=data.get('temperature', 0.7),
                    max_tokens=data.get('max_tokens', None),
                    stream=True
                )
                for chunk in stream:
                    if 'content' in chunk['choices'][0]['delta']:
                        self.wfile.write(chunk['choices'][0]['delta']['content'].encode('utf-8'))
                        self.wfile.flush()
            else:
                self.send_header('Content-Type', 'application/json')
                self.end_headers()

                response = llm.create_chat_completion(
                    messages=data['messages'],
                    temperature=data.get('temperature', 0.1),
                    max_tokens=data.get('max_tokens', 500),
                    stream=False
                )
                
                raw = response['choices'][0]['message']['content']
                
                # Remove CoT blocks
                #TODO: this needs work...
                clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
                # Delimiter Split
                if "|||" in clean:
                    clean = clean.split("|||")[-1]
                
                self.wfile.write(json.dumps({"content": clean.strip()}).encode('utf-8'))

if __name__ == '__main__':
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(('127.0.0.1', 8000), InferenceHandler)
    print("Dual-Engine Server Running on :8000", flush=True)
    server.serve_forever()