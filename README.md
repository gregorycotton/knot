![Knot banner image](https://github.com/gregorycotton/knot/blob/main/img/knot.webp)

# 🪢 Knot
Knot is my client-server TUI for running LLMs locally.

* Convos automatically saved to SQLite DB.
* Install models to run locally.
* Markdown rendering, tables, syntax highlighting, etc.
* Load .md files into chat context.
* Generate and download summaries of a given conversation.

See the TODO section at the bottom of the README for known errors and future improvments.

<br>

## Details
Knot consists of two component that run simultaneously, `server.py`, the inference server which uses `llama-cpp-python`, and `knot.py`, the TUI client that renders the streaming response.

<br>

* Engine: [Llama-cpp-python][llama-cpp-python] (python bindings for llama.cpp),
* UI: [Rich][rich] and [Prompt Toolkit][prompt-toolkit],
* SQLite DB

[llama-cpp-python]: https://github.com/abetlen/llama-cpp-python
[rich]: https://github.com/Textualize/rich
[prompt-toolkit]: https://github.com/prompt-toolkit/python-prompt-toolkit

<br>

## Installation
To get started follow the below.

**Note**: Currently optimized for Apple Silicon (M1/M2/M3) with Metal GPU acceleration.

<br>

#### 1. Clone & prepare
Clone the repository and create your virtual environment:
```bash
# Create the project folder
mkdir knot
cd knot

# Create your venv
python3 -m venv knot
source knot/bin/activate
```
**Note**: If you don't have it already, this will create a folder titled `convo` in your project root as well as a SQLite DB in the folder for your conversation history.

<br>

#### 2. Install engine with Metal support
Compile `llama-cpp-python` with Metal support to use Mac GPU:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```
**Note**: If you are on Linux/Windows, you should be able to just omit the CMAKE_ARGS part.

<br>

#### 3. Install dependancies
UI and utility libs need installation still:
```bash
pip install rich prompt_toolkit huggingface_hub
```

<br>

#### 4. Use the thing
WIP still so lots for me to fix but can be played around with now.

**Note**: By default Phi 3 mini will be downloaded upon running. ~sub-2.5GB in size. This can be changed by altering the code yourself though (crtl+f in `knot.py` for "No model found" (should be around line 160)).

Run the server in the background and the client in the foreground:
```bash
# In one terminal tab
python3 server.py

# In your second terminal tab
python3 knot.py
```
**Note**: It's cumbersome to cd into your directory, activate your venv and run two python files whenever you want to use knot. I've created a custom command by adding an alias to my shell config file so that I can run `knot` from anywhere in my terminal to automatically activate the venv and launch the application. Example:
```bash
#!/bin/bash

cd /{YOUR_FILE_PATH}/knot || exit

source {YOUR_VENV_NAME}/bin/activate || exit

cleanup() {
    kill $SERVER_PID 2>/dev/null
}

trap cleanup EXIT INT TERM

python3 server.py > server.log 2>&1 &

SERVER_PID=$!

sleep 1

python3 knot.py
```

Make this executable and alias it to `knot` in your shell config.

<br>

## Comand reference
Type normally to chat or start a line with `:` to enter a command. Quick overview:

| Command           | Action                                                                             |
|-------------------|------------------------------------------------------------------------------------|
| `:new`            | Start a new conversation and clear the current context                             |
| `:history`        | List past conversations                                                            |
| `:open <id>`      | Open a conversation by its partial ID                                              |
| `:delete <id>`    | Delete a conversation permenantly                                                  |
| `:load <file>`    | Load a text/md file as context                                                     |
| `:summary`        | Save a summary of this chat to Downloads                                           |
| `:search <term>`  | Search conversations (* at end for partial matches)                                |
| `:job <cmd>`      | Assign tasks to models (list, set summary, set title)                              |
| `:model <cmd>`    | Manage active / downloaded models (add, select, list)                              |
| `:quit`           | Exit Knot                                                                          |
| `:help`           | View possible commands                                                             |

To set a model's job using the `:job` command, use `:job set <task> <model_ID>`. Currently, the two tasks available for designating models to are `summary` (ie. the `:summary` command) and `title` (ie. generating a title for the conversation). For example:
* `job set title 1` ensures all titles are generated using the model with the ID of `1`.
* `job set summary 2` ensures all conversations are summarized using then model with the ID of `2`.

**Note**: I would currently reccomend using a non-CoT model for these jobs (see known errors).

<br>

## TODO:

#### Known errors
* CoT models title gen includes CoT tokens/generally is no good (low-priority as ability to set a model job to title gen exists)
* `:summary` command sometimes doesn't work well for GPT OSS converations due to CoT.

<br>

#### Future improvements
* Add ability to "branch" a new conversation from any previous message.
* Need to explore most expedient way to display maths/proofs, etc.
* Explore possibility of web search and/or search over local documents.
* Hide 'thinking' from models by default with optional command to expose/toggle CoT tokens.
* CoT tokens being saved to DB means they eat up context windows.
