# 🪢 Knot
Standalone LLM chat for the terminal. Knot uses `llama.cpp`, runs the LLM directly inside the Python process and handles model management, persistent history, and context injection automatically.

<br>

## Features
* Convos automatically saved to a SQLite DB,
* Currently optimized for Apple Silicon (M1/M2/M3) with Metal GPU acceleration,
* Markdown rendering, tables, syntax highlighting, etc.,
* Load local .md files into chat context,
* Generate and download summaries of a given conversation,

**Note** Currently I have Phi 3 Mini and GPT OSS 20b set up: you can add more as you see fit but will need to do it manually (see **TODO** section at bottom of this README).
**Note 2**: GPT OSS doesn't always jive well with how this is set up, reccomend Phi 3 or another model that is __not__ chain-of-thought (again, see **TODO** section).

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

| Command           | Action                                                                             |
|-------------------|------------------------------------------------------------------------------------|
| :new              | Start a new conversation and clear the current context                             |
| :history          | List past conversations                                                            |
| :open <id>        | Open a conversation by its partial ID                                              |
| :load <file>      | Load a text/md file as context                                                     |
| :summary          | Save a summary of this chat to Downloads                                           |
| :help             | Self explanatory                                                                   |
| :search <term>    | Search conversations (* at end for partial matches)                                |
| :model <cmd>      | Manage active and downloaded models (add, select, list)                            |
| :quit             | Exit Knot                                                                          |

<br>

## TODO:
* Load models from TUI with `:model add`,
* Add ability to "branch" a new conversation from any previous message,
* Explore possibility of web search and/or search over local documents,
* For now need to handle how GPT OSS "thinks" visually, especially with regard to generating titles for new convos.

<br>

**Longer-term**: In the future will make a change, architecturally. When you change the active model today you need to restart to apply the change. This is a consequence of running the model directly inside the python process (lazy and simple way to get started with this). In the future I'd like to run `llama-cpp-python` as a separate persistent HTTP service rather than as an embedded library to make this smoother. This will also allow for models dedicated to specific tasks (eg. for generating conversation titles or summaries, etc.).
