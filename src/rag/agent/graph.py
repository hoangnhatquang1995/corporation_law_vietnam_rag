from langgraph.graph import START,END, StateGraph
from settings.types import StateNode
from .nodes import llm_node, rag_node
from langgraph.checkpoint.memory import MemorySaver 

def create_agent_graph():

    builder = StateGraph(StateNode)
    builder.add_node("rag",rag_node)
    builder.add_edge(START,"rag")
    builder.add_edge("rag",END)
    
    checkpoint = MemorySaver()

    return builder.compile(checkpointer = checkpoint)

