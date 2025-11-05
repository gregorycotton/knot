# Knot
Web app for chatting with local language models.

**Features:**
* Chat history stored in SQLite DB
* More... eventually (surely, right)

**Stack:**
* Python 3, Flask
* Ollama for LLM service
* SQLite DB for convo history

**Quick set-up & pre-reqs:**
    1.    Install Ollama & pull your favourite model (I find ```gpt-oss: 20b``` runs well on my Macbook M1 32GB)/
    2.    Clone this repository/
    3.    Optional: create a Modelfile with any instructions or system prompt/
    4.    Create the knot model in Ollama (```ollama create knot -f Modelfile```)/
    5.    Install dependencies in your virtual environment (```pip install flask requests flask-cors```)/
    6.    Run Ollama to serve your model (```ollama run knot```) then run the server (```python3 app.py```)/
