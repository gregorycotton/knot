# DEPRECATED – THIS FILE IS NO LONGER IN USE
# MANY FEATURES NOT AVAILABLE HERE BUT I GUESS IT KINDA WORKS
# TO DELETE

import sys
import os
import sqlite3
import json
import uuid
from pathlib import Path
import re

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

DB_FILE = "convo/history.db"
MODEL_DIR = "./models"

console = Console()

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
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO models (short_name, repo_id, filename, context_window, is_active) VALUES 
            (?, ?, ?, ?, 1)
            """, 
            ('phi3', 'microsoft/Phi-3-mini-4k-instruct-gguf', 'Phi-3-mini-4k-instruct-q4.gguf', 4096)
        )
        cursor.execute("""
            INSERT INTO models (short_name, repo_id, filename, context_window, is_active) VALUES 
            (?, ?, ?, ?, 0)
            """, 
            ('gpt-oss', 'bartowski/openai_gpt-oss-20b-GGUF', 'openai_gpt-oss-20b-Q4_K_M.gguf', 8192)
        )

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

def get_active_model_config(conn):
    """Retrieves the model marked as active (is_active = 1)."""
    return conn.execute("SELECT * FROM models WHERE is_active = 1").fetchone()

def update_active_model(conn, model_id):
    """
    Set a new model as active by ID and deactivates the others.
    """
    conn.execute("UPDATE models SET is_active = 0")
    conn.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
    conn.commit()

class AppState:
    def __init__(self):
        self.conn = init_db()
        self.convo_id = None
        self.context_content = ""
        self.context_filename = ""
        self.llm = None

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

def boot_model():
    """
    Check DB for the active model, ensure the file is local, load it.
    """
    
    conn = state.conn
    config = get_active_model_config(conn)

    if not config:
        console.print("[bold red]FATAL: No active model found. Run :model list and :model select.[/bold red]")
        sys.exit(1)
        
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
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
            console.print("[bold green]Download complete![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error downloading model: {e}[/bold red]")
            sys.exit(1)

    console.print(f"[cyan]Loading {MODEL_NAME.upper()} into Metal (CTX={CONTEXT_WINDOW})...[/cyan]")
    
    try:
        state.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=CONTEXT_WINDOW,
            verbose=False
        )
        console.print(f"[bold green]Engine Ready. Active Model: {MODEL_NAME.upper()} (Context: {CONTEXT_WINDOW})[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to load model: {e}[/bold red]")
        sys.exit(1)


def generate_smart_title(first_message):
    """
    Use the LLM to generate short title based on your first message.
    """
    try:
        prompt = f"Generate a succinct, 3-6 word title for this text: '{first_message}'. Return ONLY the title text."
        
        response = state.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20, 
            temperature=0.1 
        )
        
        title = response['choices'][0]['message']['content'].strip().strip('"')
        
        title = re.sub(r'^(Title:|Word Count:|\S+\s+words|\d+ words|\S+\s+word).*', '', title, flags=re.IGNORECASE).strip()
        
        if not title:
            return "Untitled Conversation"
            
        return title
    except Exception:
        return "New Conversation"

def stream_llm_response(user_input):
    if state.convo_id is None:
        with console.status("[bold yellow]Generating title...[/bold yellow]"):
            new_title = generate_smart_title(user_input)
        
        state.convo_id = create_conversation(state.conn, new_title)
        
        console.print(f"[dim]Conversation saved as: {new_title}[/dim]")

    save_message(state.conn, state.convo_id, "user", user_input)
    history = get_history(state.conn, state.convo_id)
    
    if state.context_content:
        last_msg = history[-1]
        
        injected_content = f"Use the following context to answer the question:\n\n---\n{state.context_content}\n---\n\nUser Question: {last_msg['content']}"
        
        history[-1]['content'] = injected_content

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

def handle_history():
    """
    Display conversation history.
    """
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
    """
    Open a conversation based on ID or partial ID.
    """
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
    """
    Load a local .md file into the conversation context.
    """
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

def sanitize_chat_history(history):
    """Aggressively removes known problematic internal model control tokens."""
    control_token_pattern = re.compile(r'<\s*\|[^>]*\|?\s*>', re.IGNORECASE) 
    
    cleaned_history = []
    for message in history:
        cleaned_content = control_token_pattern.sub('', message['content'])
        cleaned_history.append({
            'role': message['role'],
            'content': cleaned_content.strip()
        })
    return cleaned_history

def handle_summary():
    if state.convo_id is None:
        console.print("[red]No active conversation to summarize.[/red]")
        return

    console.print("[yellow]Generating summary...[/yellow]")
    history = get_history(state.conn, state.convo_id)
    
    cleaned_history = sanitize_chat_history(history)
    
    cleaned_history.append({"role": "user", "content": "Summarize this entire conversation into a concise markdown note."})
    
    try:
        response = state.llm.create_chat_completion(
            messages=cleaned_history,
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

def handle_search(args):
    """
    Search history, group results by conversation, displays a count of occurances.
    """
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
            table.add_row(
                row['convo_id'][:8], 
                row['title'],
                str(row['occurrences']),
            )
            unique_convo_ids.add(row['convo_id'])
            
        console.print(table)
        console.print(f"[italic]Found {len(unique_convo_ids)} conversation(s).[/italic]")

    except sqlite3.Error as e:
        console.print(f"[bold red]Database Search Error: {e}[/bold red]")

def handle_help():
    """
    Help menu with descriptions of possible actions.
    """
    help_text = """
    [bold]Commands:[/bold]
    :new            - Start a new conversation
    :history        - List past conversations
    :open <id>      - Open a conversation by its partial ID
    :load <file>    - Load a text/md file as context
    :summary        - Save a summary of this chat to Downloads
    :search <term>  - Search conversations (* at end for partial matches)
    :model <cmd>    - Manage active and downloaded models (add, select, list)
    :quit           - Exit Knot
    """
    console.print(Panel(help_text, title="Help", border_style="white"))

def handle_model(args):
    """
    Manages models: list, select, add (WIP).
    """
    if not args:
        console.print("[red]Usage: :model <list | select <ID>>[/red]")
        return
    
    command = args[0].lower()
    conn = state.conn

    if command == 'list':
        models = conn.execute("SELECT id, short_name, repo_id, context_window, is_active FROM models").fetchall()
        
        table = Table(title="Available Models")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="yellow")
        table.add_column("Context (n_ctx)", style="magenta")
        table.add_column("Status", style="green")

        for m in models:
            status = "[bold green]ACTIVE[/bold green]" if m['is_active'] else "[dim]inactive[/dim]"
            table.add_row(str(m['id']), m['short_name'], str(m['context_window']), status)
        
        console.print(table)
        console.print("[italic]Use :model select <ID> to switch models (requires restart).[/italic]")
        
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
            console.print(Panel("[bold green]Model switched successfully. Please restart the app to load the new engine.[/bold green]", border_style="green"))
        except ValueError:
            console.print("[red]ID must be a number.[/red]")
        
    elif command == 'add':
        console.print("[red]I need to add this still lol you'll need to do this manually[/red]")
        
    else:
        console.print(f"[red]Unknown model command: {command}[/red]")

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
                elif cmd == "search": handle_search(args)
                elif cmd == "model": handle_model(args)
                else: console.print(f"[red]Unknown command: {cmd}[/red]")
            else:
                stream_llm_response(user_input)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()