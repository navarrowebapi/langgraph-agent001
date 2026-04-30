# langgraph-agent001

Minimal LangGraph/LangChain agent using Google Gemini Flash — starting point for incremental development.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Run

```bash
python agent.py
```

Type a message and the agent will reply using Gemini Flash.

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

The `llm` node sends the user input to `gemini-1.5-flash` and writes the reply back into the shared state.
