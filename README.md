# 🪢 Knot
Standalone LLM chat for the terminal. Knot uses `llama.cpp`, runs the LLM directly inside the Python process and handles model management, persistent history, and context injection automatically.

<br>

## Features
* Convos automatically saved to a SQLite DB,
* Currently optimized for Apple Silicon (M1/M2/M3) with Metal GPU acceleration,
* Markdown rendering, tables, syntax highlighting, etc.,
* Load local .md files into chat context,
* Generate and download summaries of a given conversation,

<br>

## Details
* Engine: [Llama-cpp-python][llama-cpp-python] (python bindings for llama.cpp),
* UI: [Rich][rich] and [Prompt Toolkit][prompt-toolkit],
* SQLite DB

[llama-cpp-python]: https://github.com/abetlen/llama-cpp-python
[rich]: https://github.com/Textualize/rich
[prompt-toolkit]: https://github.com/prompt-toolkit/python-prompt-toolkit

<br>

## Installation
To get started follow the below.
**Note**: currently is optimized for Apple Silicon.

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
**Note**: This won't work if you just run `pip install llama-cpp-python`: it must be compiled with Metal support in order to use Mac GPU:
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
WIP still so lots for me to fix but can be played around with now:
```bash
python3 cli.py
```
**Note**: I've created a custom command by adding an alias to my terminal config file so that I can run `knot` from anywhere in my terminal which automatically activates the virtual environment and launches the application. Something like this can be done with a shell file like the below:
```bash
#!/bin/bash

cd /Users/{YOUR PATH TO THE PROJECT FOLDER}/App/knot || exit

source {YOUR VENV NAME}/bin/activate || exit

python3 cli.py
```

<br>

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
