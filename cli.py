import sys
import os
import sqlite3
import json
import uuid
from pathlib import Path

# Embedded engine
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# TUI libs
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# Config
DB_FILE = "convo/history.db"
MODEL_DIR = "./models"

# Current model to test is Phi-3 Mini
REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"

console = Console()

# DB logic
def init_db():
    os.makedirs("convo", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    return conn

def create_conversation(conn, title):
    convo_id = str(uuid.uuid4())
    conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (convo_id, title))
    conn.commit()
    return convo_id

def save_message(conn, convo_id, role, content):
    msg_id = str(uuid.uuid4())
    conn.execute("INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)", (msg_id, convo_id, role, content))
    conn.commit()

def get_history(conn, convo_id):
    rows = conn.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", (convo_id,)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def list_conversations(conn):
    return conn.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC").fetchall()

# App state
class AppState:
    def __init__(self):
        self.conn = init_db()
        self.convo_id = None # None means "Unsaved/New"
        self.context_content = ""
        self.context_filename = ""
        self.llm = None

    def start_new_chat(self):
        # We DO NOT create a DB entry yet. We wait for the first message.
        self.convo_id = None
        console.print(Panel("[bold green]Ready. (Conversation will be saved on first message)[/bold green]", border_style="green"))

    def set_active_convo(self, convo_id, title):
        self.convo_id = convo_id
        console.print(Panel(f"[bold blue]Switched to: {title}[/bold blue]", border_style="blue"))
        msgs = get_history(self.conn, self.convo_id)
        for msg in msgs:
            role_color = "bold blue" if msg['role'] == 'user' else "bold magenta"
            console.print(f"\n[{role_color}]{msg['role'].upper()}:[/{role_color}]")
            console.print(Markdown(msg['content']))

state = AppState()

# Model management
def boot_model():
    """Checks if the model exists locally. If not, downloads it. Then loads it into RAM."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, FILENAME)

    if not os.path.exists(model_path):
        console.print(f"[bold yellow]Model file not found at {model_path}[/bold yellow]")
        console.print(f"Downloading {REPO_ID} from Hugging Face... (This happens once)")
        try:
            model_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=FILENAME, 
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            console.print("[bold green]Download complete![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error downloading model: {e}[/bold red]")
            sys.exit(1)

    console.print(f"[cyan]Loading model into Metal (GPU)...[/cyan]")
    
    # Initialize engine directly
    try:
        state.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Offload all layers to Mac GPU for speed
            n_ctx=4096, # Context window size
            verbose=False
        )
        console.print("[bold green]Engine Ready.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to load model: {e}[/bold red]")
        sys.exit(1)

# Auto-Title Generator
def generate_smart_title(first_message):
    """Uses the LLM to generate a short title based on the user's first message."""
    try:
        prompt_messages = [
            {"role": "system", "content": "You are a summarization tool. Generate a concise title (3-6 words) for the following user query. Do not use quotes. Return ONLY the title."},
            {"role": "user", "content": first_message}
        ]
        
        response = state.llm.create_chat_completion(
            messages=prompt_messages,
            max_tokens=20, 
            temperature=0.5
        )
        
        title = response['choices'][0]['message']['content'].strip().strip('"')
        return title
    except Exception:
        return "New Conversation"

# Embedded chat logic
def stream_llm_response(user_input):
    
    if state.convo_id is None:
        # Generate title, create convo in DB
        with console.status("[bold yellow]Generating title...[/bold yellow]"):
            new_title = generate_smart_title(user_input)
        
        state.convo_id = create_conversation(state.conn, new_title)
        
        console.print(f"[dim]Conversation saved as: {new_title}[/dim]")

    save_message(state.conn, state.convo_id, "user", user_input)
    history = get_history(state.conn, state.convo_id)
    
    if state.context_content:
        system_prompt = f"Context from file '{state.context_filename}':\n{state.context_content}"
        history.insert(0, {"role": "system", "content": system_prompt})

    full_response = ""
    console.print(f"\n[bold magenta]KNOT ({state.context_filename or 'No Context'}):[/bold magenta]")
    
    with Live(Markdown(""), refresh_per_second=10, auto_refresh=False) as live:
        try:
            stream = state.llm.create_chat_completion(
                messages=history,
                stream=True,
                temperature=0.7
            )

            for chunk in stream:
                if 'content' in chunk['choices'][0]['delta']:
                    text_chunk = chunk['choices'][0]['delta']['content']
                    full_response += text_chunk
                    live.update(Markdown(full_response))
                    live.refresh()

        except Exception as e:
            console.print(f"[bold red]Error during inference: {e}[/bold red]")
            return

    save_message(state.conn, state.convo_id, "assistant", full_response)
    console.print("") 

# Commands
def handle_history():
    convos = list_conversations(state.conn)
    if not convos:
        console.print("[yellow]No conversation history found.[/yellow]")
        return
    table = Table(title="Conversation History")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Date", style="magenta")
    table.add_column("Title", style="green")
    for c in convos:
        table.add_row(c["id"][:8], c["created_at"][:16], c["title"])
    console.print(table)
    console.print("[italic]To select, type: :open <first-8-chars-of-id>[/italic]")

def handle_open(args):
    if not args:
        console.print("[red]Usage: :open <id_fragment>[/red]")
        return
    target = args[0]
    convos = list_conversations(state.conn)
    for c in convos:
        if c["id"].startswith(target):
            state.set_active_convo(c["id"], c["title"])
            return
    console.print(f"[red]Conversation starting with {target} not found.[/red]")

def handle_load(args):
    if not args:
        console.print("[red]Usage: :load <path/to/file.md>[/red]")
        return
    filepath = args[0].strip()
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        return
    try:
        state.context_content = path.read_text(encoding='utf-8')
        state.context_filename = path.name
        console.print(f"[green]Loaded context from: {path.name}[/green]")
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")

def handle_summary():
    # Check if convo exists first
    if state.convo_id is None:
        console.print("[red]No active conversation to summarize.[/red]")
        return

    console.print("[yellow]Generating summary...[/yellow]")
    history = get_history(state.conn, state.convo_id)
    history.append({"role": "user", "content": "Summarize this entire conversation into a concise markdown note."})
    
    try:
        response = state.llm.create_chat_completion(
            messages=history,
            stream=False
        )
        full_summary = response["choices"][0]["message"]["content"]
    except Exception as e:
        console.print(f"[red]Error generating summary: {e}[/red]")
        return
        
    filename = f"Summary_{state.convo_id[:8]}.md"
    downloads_path = Path.home() / "Downloads" / filename
    try:
        downloads_path.write_text(full_summary, encoding='utf-8')
        console.print(f"[bold green]Summary saved to: {downloads_path}[/bold green]")
    except Exception as e:
        console.print(f"[red]Error saving file: {e}[/red]")

def handle_help():
    help_text = """
    [bold]Commands:[/bold]
    :new          - Start a new conversation
    :history      - List past conversations
    :open <id>    - Open a conversation by its partial ID
    :load <file>  - Load a text/md file as context
    :summary      - Save a summary of this chat to Downloads
    :quit         - Exit the app
    """
    console.print(Panel(help_text, title="Help", border_style="white"))

# Main
def main():
    boot_model()
    
    session = PromptSession(history=InMemoryHistory())
    state.start_new_chat()
    console.print("[bold yellow]Welcome to Knot CLI (Embedded Mode).[/bold yellow] Type [bold]:help[/bold] for commands.")

    while True:
        try:
            prompt_text = f"[{state.context_filename or 'No Context'}] > "
            user_input = session.prompt(prompt_text).strip()
            if not user_input: continue
            if user_input.startswith(":"):
                parts = user_input[1:].split()
                cmd = parts[0].lower()
                args = parts[1:]
                if cmd in ["quit", "q"]:
                    console.print("Goodbye.")
                    break
                elif cmd == "help": handle_help()
                elif cmd == "history": handle_history()
                elif cmd == "new": state.start_new_chat()
                elif cmd == "open": handle_open(args)
                elif cmd == "load": handle_load(args)
                elif cmd == "summary": handle_summary()
                else: console.print(f"[red]Unknown command: {cmd}[/red]")
            else:
                stream_llm_response(user_input)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()