# 🪢 Knot
Knot is my client-server TUI for running LLMs locally.

<br>

## Features
* Convos automatically saved to SQLite DB,
* Install models to run locally,
* Markdown rendering, tables, syntax highlighting, etc.,
* Load .md files into chat context,
* Generate and download summaries of a given conversation;

<br>

## Details
Knot consists of two component that run simultaneously, `server.py`, the lightweight HTTP inference server which uses `llama-cpp-python`, and `knot.py`, the TUI client that renders the streaming response.

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
Compile `llama-cpp-python` with Metal support to use Mac GPU. __(If you are on Linux/Windows, you should be able to just omit the CMAKE_ARGS part)__:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

<br>

#### 3. Install dependancies
UI and utility libs need installation still:
```bash
pip install rich prompt_toolkit huggingface_hub
```

<br>

#### 4. Use the thing
WIP still so lots for me to fix but can be played around with now.
Run the server in the background and the client in the foreground:
```bash
# In one terminal tab
python3 server.py

# In your second terminal tab
python3 knot.py
```
**Note**: It's cumbersome to cd into your directory, activate your venv and run two python files whenever you want to use knot. I've created a custom command by adding an alias to my shell config file so that I can run `knot` from anywhere in my terminal to automatically activate the venv and launch the application. Example:
```bash
# #!/bin/bash

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
Type normally to chat or start a line with `:` to enter a command.

| Command           | Action                                                                             |
|-------------------|------------------------------------------------------------------------------------|
| :new              | Start a new conversation and clear the current context                             |
| :history          | List past conversations                                                            |
| :open <id>        | Open a conversation by its partial ID                                              |
| :delete <id>      | Delete a conversation permenantly                                                  |
| :load <file>      | Load a text/md file as context                                                     |
| :summary          | Save a summary of this chat to Downloads                                           |
| :help             | Self explanatory                                                                   |
| :search <term>    | Search conversations (* at end for partial matches)                                |
| :model <cmd>      | Manage active / downloaded models (add, select, list)                            |
| :quit             | Exit Knot                                                                          |

<br>

## TODO:
* Add ability to "branch" a new conversation from any previous message,
* Explore possibility of web search and/or search over local documents,
* For now need to handle how GPT OSS "thinks" visually, especially with regard to generating titles for new convos (possibly have ability to dedicate models to tasks and dedicate Phi3 to title gen)

**Note**: `cli.py` is kicking around in this repo but is dead weight as of right now. It was the original embedded version of Knot, but it's architecture became a bottleneck. Will delete later; it still works but is less feature rich and is a bit buggy.