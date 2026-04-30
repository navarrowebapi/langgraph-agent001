"""Minimal LangGraph agent using Google Gemini Flash."""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()


# --- State ---

class AgentState(TypedDict):
    user_input: str
    response: str


# --- Model ---

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# --- Nodes ---

def call_llm(state: AgentState) -> AgentState:
    """Send the user input to Gemini Flash and store the reply."""
    message = llm.invoke(state["user_input"])
    return {"response": message.content}


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
