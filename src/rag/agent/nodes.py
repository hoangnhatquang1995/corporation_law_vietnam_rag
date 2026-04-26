from settings.types import StateNode
from rag.llm.models import get_llm_model, LLMProvider
from rag.llm import llm
from .system_prompts import llm_system_prompt,rag_system_prompt
from dataset import documentDB 



def llm_node(state: StateNode) -> StateNode:
    messages = [
        llm_system_prompt,
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {
        "messages": [response]
    }

def rag_node(state: StateNode) -> StateNode:
    question: str = str(state["messages"][-1].content) or ""
    if not question:
        raise ValueError("Question is required for RAG node")
    context = documentDB.query(question, top_k=5)
    context_str = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(context)])
    print(f"RAG Node - Retrieved Context:\n{context_str}\n")
    messages = [
        rag_system_prompt.format(context=context_str, question=question),
    ]

    response = llm.invoke(messages)
    return {
        "messages": [response]
    }   


    
