import sys
import os
import sqlite3
import json
import uuid
import re
import urllib.request
import urllib.error
from pathlib import Path

# Download model file locally
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
API_URL = "http://127.0.0.1:8000"

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
        
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            short_name TEXT UNIQUE NOT NULL,
            repo_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            context_window INTEGER NOT NULL,
            is_active INTEGER DEFAULT 0
        );
        
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content, 
            conversation_id UNINDEXED, 
            content='messages', 
            content_rowid='rowid'
        );
                       
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, conversation_id) VALUES (
                new.rowid, 
                new.content,
                new.conversation_id
            );
        END;
    """);

    conn.commit()
    return conn

# DB helpers
def create_conversation(conn, title):
    convo_id = str(uuid.uuid4())
    conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (convo_id, title))
    conn.commit()
    return convo_id

def delete_conversation_by_id(conn, convo_id):
    """
    Delete a conversation.
    """
    conn.execute("DELETE FROM conversations WHERE id = ?", (convo_id,))
    conn.commit()

def save_message(conn, convo_id, role, content):
    msg_id = str(uuid.uuid4())
    conn.execute("INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)", (msg_id, convo_id, role, content))
    conn.commit()

def get_history(conn, convo_id):
    rows = conn.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", (convo_id,)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def list_conversations(conn):
    return conn.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC").fetchall()

def get_active_model_config(conn):
    return conn.execute("SELECT * FROM models WHERE is_active = 1").fetchone()

def update_active_model(conn, model_id):
    conn.execute("UPDATE models SET is_active = 0")
    conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
    conn.commit()

# App state
class AppState:
    def __init__(self):
        self.conn = init_db()
        self.convo_id = None
        self.context_content = ""
        self.context_filename = ""
        self.current_model_name = ""

    def start_new_chat(self):
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

# API Helpers
def api_request(endpoint, payload, stream=False):
    """
    Send JSON requests to local engine.
    """
    url = f"{API_URL}{endpoint}"
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header('Content-Type', 'application/json')
    
    try:
        return urllib.request.urlopen(req)
    except urllib.error.URLError:
        console.print(f"[bold red]Could not connect to Engine at {API_URL}[/bold red]")
        console.print("[italic]Did you run 'python server.py' in another terminal?[/italic]")
        return None

# Model management
def boot_model(session=None):
    conn = state.conn
    config = get_active_model_config(conn)

    if not config:
        count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        
        if count == 0:
            # No model found
            console.print(Panel("[bold yellow]Welcome to Knot![/bold yellow]\nNo models found. Let's set one up.", border_style="yellow"))
            
            if session:
                console.print("1. Quick Start (Download Phi-3 Mini - 2.4GB)")
                console.print("2. Custom (Enter Repo ID manually)")
                
                choice = session.prompt("Choose (1/2): ").strip()
                
                if choice == '1':
                    repo = "microsoft/Phi-3-mini-4k-instruct-gguf"
                    filename = "Phi-3-mini-4k-instruct-q4.gguf"
                    name = "phi3"
                    ctx = 4096
                    conn.execute("INSERT INTO models (short_name, repo_id, filename, context_window, is_active) VALUES (?, ?, ?, ?, 1)", (name, repo, filename, ctx))
                    conn.commit()
                else:
                    handle_model(['add'], session)
                    if conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 0:
                        console.print("[red]No model added. Exiting.[/red]")
                        sys.exit(1)
                    conn.execute("UPDATE models SET is_active = 1 WHERE id = (SELECT MAX(id) FROM models)")
                    conn.commit()
            else:
                console.print("[red]No models found and no session available. Exiting.[/red]")
                sys.exit(1)
                
            config = get_active_model_config(conn)
            
        else:
            # Models exist but none active
            conn.execute("UPDATE models SET is_active = 1 WHERE id = (SELECT MIN(id) FROM models)")
            conn.commit()
            config = get_active_model_config(conn)
            console.print("[yellow]No active model selected. Auto-selected the first available model.[/yellow]")

    REPO_ID = config['repo_id']
    FILENAME = config['filename']
    CONTEXT_WINDOW = config['context_window']
    MODEL_NAME = config['short_name']

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, FILENAME)

    if not os.path.exists(model_path):
        console.print(f"[bold yellow]Model file not found: {FILENAME}[/bold yellow]")
        console.print(f"Downloading {MODEL_NAME.upper()} from Hugging Face (This happens once)...")
        try:
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODEL_DIR)
            console.print("[bold green]Download complete![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error downloading model: {e}[/bold red]")
            sys.exit(1)

    abs_path = os.path.abspath(model_path)
    console.print(f"[cyan]Requesting engine load {MODEL_NAME.upper()}...[/cyan]")
    
    response = api_request("/load_model", {
        "model_path": abs_path,
        "n_ctx": CONTEXT_WINDOW
    })
    
    if response and response.status == 200:
        state.current_model_name = MODEL_NAME
        console.print(f"[bold green]Engine Loaded: {MODEL_NAME.upper()} (Context: {CONTEXT_WINDOW})[/bold green]")
    else:
        console.print("[bold red]Engine failed to load model.[/bold red]")

def generate_smart_title(first_message):
    prompt = f"Generate a succinct, 3-6 word title for this text: '{first_message}'. Return ONLY the title text."
    
    response = api_request("/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20, 
        "temperature": 0.1
    }, stream=True)
    
    if not response:
        return "New Conversation"

    full_text = response.read().decode('utf-8')
    title = re.sub(r'^(Title:|Word Count:|\S+\s+words|\d+ words|\S+\s+word).*', '', full_text, flags=re.IGNORECASE).strip()
    title = title.strip('"')
    return title if title else "Untitled Conversation"

def sanitize_chat_history(history):
    control_token_pattern = re.compile(r'<\s*\|[^>]*\|?\s*>', re.IGNORECASE) 
    cleaned_history = []
    for message in history:
        cleaned_content = control_token_pattern.sub('', message['content'])
        cleaned_history.append({
            'role': message['role'],
            'content': cleaned_content.strip()
        })
    return cleaned_history

# Chat logic
def stream_llm_response(user_input):
    if state.convo_id is None:
        with console.status("[bold yellow]Generating title...[/bold yellow]"):
            new_title = generate_smart_title(user_input)
        state.convo_id = create_conversation(state.conn, new_title)
        console.print(f"[dim]Conversation saved as: {new_title}[/dim]")

    save_message(state.conn, state.convo_id, "user", user_input)
    history = get_history(state.conn, state.convo_id)
    
    # Context injection
    if state.context_content:
        last_msg = history[-1]
        injected_content = f"Use the following context to answer the question:\n\n---\n{state.context_content}\n---\n\nUser Question: {last_msg['content']}"
        history[-1]['content'] = injected_content

    # Sanitize
    clean_history = sanitize_chat_history(history)

    console.print(f"\n[bold magenta]KNOT ({state.current_model_name.upper()}):[/bold magenta]")
    full_response = ""
    
    # Request stream
    response = api_request("/chat/completions", {
        "messages": clean_history,
        "temperature": 0.7
    }, stream=True)

    if not response:
        return

    with Live(Markdown(""), refresh_per_second=10, auto_refresh=False) as live:
        try:
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                text_chunk = chunk.decode('utf-8', errors='replace')
                full_response += text_chunk
                live.update(Markdown(full_response))
                live.refresh()
        except Exception as e:
            console.print(f"[bold red]Stream error: {e}[/bold red]")

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

def handle_delete(args):
    if not args:
        console.print("[red]Usage: :delete <id_fragment>[/red]")
        return
    
    target = args[0]
    convos = list_conversations(state.conn)
    target_convo = None
    
    for c in convos:
        if c["id"].startswith(target):
            target_convo = c
            break
            
    if not target_convo:
        console.print(f"[red]Conversation starting with '{target}' not found.[/red]")
        return

    full_id = target_convo["id"]
    title = target_convo["title"]

    delete_conversation_by_id(state.conn, full_id)
    console.print(f"[bold red]Deleted conversation:[/bold red] {title} ({full_id[:8]})")

    if state.convo_id == full_id:
        console.print("[yellow]Active conversation deleted. Starting fresh...[/yellow]")
        state.start_new_chat()
        
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
    if state.convo_id is None:
        console.print("[red]No active conversation to summarize.[/red]")
        return

    console.print("[yellow]Generating summary...[/yellow]")
    history = get_history(state.conn, state.convo_id)
    clean_history = sanitize_chat_history(history)
    clean_history.append({"role": "user", "content": "Summarize this entire conversation into a concise markdown note."})
    
    response = api_request("/chat/completions", {
        "messages": clean_history,
        "temperature": 0.5
    }, stream=True)

    if not response:
        return

    full_summary = response.read().decode('utf-8')
        
    filename = f"Summary_{state.convo_id[:8]}.md"
    downloads_path = Path.home() / "Downloads" / filename
    try:
        downloads_path.write_text(full_summary, encoding='utf-8')
        console.print(f"[bold green]Summary saved to: {downloads_path}[/bold green]")
    except Exception as e:
        console.print(f"[red]Error saving file: {e}[/red]")

def handle_search(args):
    if not args:
        console.print("[red]Usage: :search <keyword or phrase>[/red]")
        return
    
    search_query = " ".join(args)
    unique_convo_ids = set()
    
    try:
        conn = state.conn
        search_results = conn.execute(
            """
            SELECT 
                mfts.conversation_id AS convo_id,
                c.title AS title,
                COUNT(mfts.conversation_id) AS occurrences
            FROM messages_fts mfts
            JOIN conversations c ON c.id = mfts.conversation_id
            WHERE mfts.content MATCH ?
            GROUP BY mfts.conversation_id
            ORDER BY occurrences DESC
            """,
            (search_query,)
        ).fetchall()

        if not search_results:
            console.print(f"[yellow]No results found for '{search_query}'.[/yellow]")
            return

        table = Table(title=f"Search Results for '{search_query}'")
        table.add_column("ID", style="cyan", no_wrap=True) 
        table.add_column("Title", style="green")
        table.add_column("Occurrences", style="magenta")

        for row in search_results:
            table.add_row(row['convo_id'][:8], row['title'], str(row['occurrences']))
            unique_convo_ids.add(row['convo_id'])
            
        console.print(table)
        console.print(f"[italic]Found {len(unique_convo_ids)} conversation(s).[/italic]")
    except sqlite3.Error as e:
        console.print(f"[bold red]Database Search Error: {e}[/bold red]")

def handle_help():
    help_text = """
    [bold]Commands:[/bold]
    :new            - Start a new conversation
    :history        - List past conversations
    :open <id>      - Open a conversation by its partial ID
    :delete <id>    - Delete a conversation by its partial ID
    :load <file>    - Load a text/md file as context
    :summary        - Save a summary of this chat to Downloads
    :search <term>  - Search conversations
    :model <cmd>    - Manage active models (list, select, add)
    :quit           - Exit Knot
    """
    console.print(Panel(help_text, title="Help", border_style="white"))

def handle_model(args, session):
    """
    Manage models: list, select, add.
    """
    if not args:
        console.print("[red]Usage: :model <list | select | add>[/red]")
        return
    
    command = args[0].lower()
    conn = state.conn

    # List
    if command == 'list':
        models = conn.execute("SELECT id, short_name, repo_id, filename, context_window, is_active FROM models").fetchall()
        
        table = Table(title="Available Models")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Short Name", style="yellow")
        table.add_column("Context", style="magenta")
        table.add_column("Status", style="green")

        for m in models:
            status = "[bold green]ACTIVE[/bold green]" if m['is_active'] else "[dim]inactive[/dim]"
            table.add_row(str(m['id']), m['short_name'], str(m['context_window']), status)
        
        console.print(table)
    
    # Select
    elif command == 'select':
        if len(args) < 2:
            console.print("[red]Usage: :model select <ID>[/red]")
            return
        
        try:
            model_id = int(args[1])
            if conn.execute("SELECT id FROM models WHERE id = ?", (model_id,)).fetchone() is None:
                console.print(f"[red]Model ID {model_id} not found.[/red]")
                return

            update_active_model(conn, model_id)
            boot_model(session)
            
        except ValueError:
            console.print("[red]ID must be a number.[/red]")

    # Add
    elif command == 'add':
        console.print(Panel("[bold cyan]Add New Model Configuration[/bold cyan]\n[dim]You will need the HuggingFace Repo ID and the specific GGUF filename.[/dim]", border_style="cyan"))

        try:
            # Short Name
            while True:
                name = session.prompt("1. Short Name (e.g. 'mistral'): ").strip()
                if not name: continue
                if conn.execute("SELECT 1 FROM models WHERE short_name = ?", (name,)).fetchone():
                    console.print(f"[red]Name '{name}' already exists. Pick another.[/red]")
                    continue
                break

            # Repo ID
            repo = session.prompt("2. HuggingFace Repo (e.g. 'TheBloke/Mistral-7B-GGUF'): ").strip()
            if not repo: 
                console.print("[red]Cancelled.[/red]")
                return

            # Filename
            console.print("[italic dim]Tip: Go to the 'Files' tab on HuggingFace and copy the filename of a Q4_K_M.gguf file.[/italic dim]")
            filename = session.prompt("3. GGUF Filename (e.g. 'mistral-7b.Q4_K_M.gguf'): ").strip()
            if not filename.endswith('.gguf'):
                console.print("[yellow]Warning: Filename usually ends in .gguf[/yellow]")
            
            # Context window
            ctx_input = session.prompt("4. Context Window (default 4096): ").strip()
            ctx = int(ctx_input) if ctx_input.isdigit() else 4096

            # Save
            console.print(f"\n[bold]Summary:[/bold]\nName: {name}\nRepo: {repo}\nFile: {filename}\nCtx:  {ctx}")
            confirm = session.prompt("Save this model? (y/n): ").lower()
            
            if confirm == 'y':
                conn.execute("""
                    INSERT INTO models (short_name, repo_id, filename, context_window, is_active) 
                    VALUES (?, ?, ?, ?, 0)
                """, (name, repo, filename, ctx))
                conn.commit()
                console.print(f"[bold green]Model '{name}' added![/bold green]")
                console.print(f"[italic]Type ':model select <id>' to switch to it. (Download will start automatically on switch)[/italic]")
            else:
                console.print("[yellow]Operation cancelled.[/yellow]")

        except KeyboardInterrupt:
            console.print("\n[red]Cancelled.[/red]")
            return

    else:
        console.print(f"[red]Unknown model command: {command}[/red]")

def main():
    console.print("[bold yellow]Connecting to Knot Engine...[/bold yellow]")
    
    session = PromptSession(history=InMemoryHistory())
    
    boot_model(session)
    
    state.start_new_chat()
    console.print("[bold yellow]Welcome to Knot CLI (Client Mode).[/bold yellow] Type [bold]:help[/bold] for commands.")

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
                elif cmd == "delete": handle_delete(args)
                elif cmd == "summary": handle_summary()
                elif cmd == "search": handle_search(args)
                elif cmd == "model": handle_model(args, session)
                else: console.print(f"[red]Unknown command: {cmd}[/red]")
            else:
                stream_llm_response(user_input)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()