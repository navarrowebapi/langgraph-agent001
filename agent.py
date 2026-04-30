"""Minimal LangGraph agent with OpenRouter default and Gemini fallback."""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()


# --- State ---

class AgentState(TypedDict):
    user_input: str
    response: str


# --- Model ---

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def build_openrouter_llm() -> ChatOpenAI | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )


def build_gemini_llm() -> ChatGoogleGenerativeAI | None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
    )


openrouter_llm = build_openrouter_llm()
gemini_llm = build_gemini_llm()


# --- Nodes ---

def call_llm(state: AgentState) -> AgentState:
    """Try OpenRouter first; if it fails, fallback to Gemini."""
    user_input = state["user_input"]
    provider_errors: list[str] = []

    if openrouter_llm is not None:
        try:
            message = openrouter_llm.invoke(user_input)
            return {"response": message.content}
        except Exception as exc:  # noqa: BLE001
            provider_errors.append(f"OpenRouter failed: {exc}")

    if gemini_llm is not None:
        try:
            message = gemini_llm.invoke(user_input)
            return {"response": message.content}
        except Exception as exc:  # noqa: BLE001
            provider_errors.append(f"Gemini failed: {exc}")

    missing_keys: list[str] = []
    if openrouter_llm is None:
        missing_keys.append("OPENROUTER_API_KEY")
    if gemini_llm is None:
        missing_keys.append("GOOGLE_API_KEY")

    if missing_keys:
        return {
            "response": (
                "No model provider is configured. "
                f"Set at least one API key: {', '.join(missing_keys)}"
            )
        }

    return {
        "response": (
            "All configured providers failed. "
            + " | ".join(provider_errors)
        )
    }


# --- Graph ---

builder = StateGraph(AgentState)
builder.add_node("llm", call_llm)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

graph = builder.compile()


# --- Entry point ---

if __name__ == "__main__":
    user_input = input("You: ")
    result = graph.invoke({"user_input": user_input})
    print(f"Agent: {result['response']}")
