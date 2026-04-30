# langgraph-agent001

Minimal LangGraph/LangChain agent using OpenRouter by default, with Gemini fallback.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt


# 3. Configure the API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
# Optional fallback key: GOOGLE_API_KEY
```

## Run

```bash
python agent.py
```

Type a message and the agent will try OpenRouter first. If OpenRouter fails, it falls back to Gemini.

## Project structure

```
agent.py          # LangGraph graph + entry point
requirements.txt  # Python dependencies
.env.example      # Environment variables template
```

## How it works

`agent.py` defines a single-node LangGraph graph:

```
START → llm → END
```

The `llm` node sends the user input to OpenRouter first and writes the reply back into the shared state. If the call fails, it retries with Gemini.
