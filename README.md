# 🪢 Knot
Standalone LLM chat for the terminal. Knot uses `llama.cpp` and runs the LLM directly inside the Python process and handles model management, persistent history, and context injection automatically.

## Features
* Persistent memory: All convos automatically saved to a SQLite DB,
* Currently optimized for Apple Silicon (M1/M2/M3) with Metal GPU acceleration,
* Markdown rendering, tables, syntax highlighting, etc.,
* Load local .md files into chat context **(WIP)**,
* Generate and download summaries of a given conversation,

## Details
* Engine: [Llama-cpp-python][https://github.com/abetlen/llama-cpp-python] (python bindings for llama.cpp),
* UI: [Rich][https://github.com/Textualize/rich] and [Prompt Toolkit][https://github.com/prompt-toolkit/python-prompt-toolkit],
* SQLite DB

## Installation
Note: currently is optimized for Apple Silicon.

1. Clone & prepare
```bash
# Create the project folder
mkdir knot
cd knot

# Create your venv
python3 -m venv knot
source knot/bin/activate
```

2. Install engine with Metal support
Note: You **cannot** run `pip install llama-cpp-python`: this must be compiled with Metal support in order to use Mac GPU.
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

3. Install dependancies
UI and utility libs need installation still.
```bash
pip install rich prompt_toolkit huggingface_hub
```

4. Use the thing
```bash
python3 cli_embedded.py
```

## Comand reference
Type normally to chat or start a line with `:` to enter a command (vim style).

| Command      | Action                                                                             |
|--------------|------------------------------------------------------------------------------------|
| :new         | Start a new conversation and clear the current context                             |
| :history     | Display past conversations                                                         |
| :open {id}   | Switch to old conversation (only need the first few unique  characters of the UUID |
| :load {file} | Load a local .md file into the LLM's short term memory (WIP)                       |
| :summary     | Download a summary of the current convo to your downloads folder as a .md          |
| :help        | Self explanatory                                                                   |
| :quit        | Even more self explanatory                                                         |