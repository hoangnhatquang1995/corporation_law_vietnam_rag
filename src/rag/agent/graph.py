from langgraph.graph import START,END, StateGraph
from settings.types import StateNode
from . import nodes
from langgraph.checkpoint.memory import MemorySaver 

def create_agent_graph():
    builder = StateGraph(StateNode)
    builder.add_node("rag",nodes.rag_node)
    builder.add_node("retrieving",nodes.retriving_node)
    
    builder.add_edge(START,"retrieving")
    builder.add_edge("retrieving","rag")
    builder.add_edge("rag",END)
    
    checkpoint = MemorySaver()

    return builder.compile(checkpointer = checkpoint)

