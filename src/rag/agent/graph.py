from langgraph.graph import START,END, StateGraph
from settings.types import StateNode
from . import nodes
from langgraph.checkpoint.memory import MemorySaver 

def create_agent_graph():
    builder = StateGraph(StateNode)
    builder.add_node("rag",nodes.rag_node)
    builder.add_node("rerank",nodes.rerank_node)
    builder.add_node("retrieving",nodes.retriving_node)
    builder.add_node("llm",nodes.llm_node)
    
    builder.add_edge(START,"llm")
    builder.add_conditional_edges("llm", llm_router)
    builder.add_edge("retrieving","rerank")
    builder.add_edge("rerank","rag")
    builder.add_edge("rag",END)
    
    checkpoint = MemorySaver()

    return builder.compile(checkpointer = checkpoint)

def llm_router(state: StateNode):
    type = state.get("type")
    if type == "legal_question":
        return "retrieving"
    elif type == "general_knowledge": 
        return END
    elif type == "chitchat":
        return END