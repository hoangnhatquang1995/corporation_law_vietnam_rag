from langgraph.graph import START,END, StateGraph
from settings.types import StateNode
from .nodes import llm_node

def create_agent_graph():
    builder = StateGraph(StateNode)
    builder.add_node("llm",llm_node)
    builder.add_edge(START,"llm")
    builder.add_edge("llm",END)
    return builder.compile()

