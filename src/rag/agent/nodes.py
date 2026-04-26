from settings.types import StateNode
from rag.llm.models import get_llm_model, LLMProvider
from rag.llm import llm
from .system_prompts import llm_system_prompt,rag_system_prompt
from dataset import documentDB 



def llm_node(state: StateNode):
    messages = [
        llm_system_prompt,
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {
        "messages": [response]
    }

def retriving_node(state: StateNode):
    question: str = str(state["messages"][-1].content) or ""
    if not question:
        raise ValueError("Question is required for RAG node")
    context = documentDB.query(question, top_k=5)
    context_str = "\n\n".join([f"""Document {doc.metadata['title']} - Number {doc.metadata['document_number']}:\n{doc.page_content}""" for i, doc in enumerate(context)])
    print(f"RAG Node - Retrieved Context:\n{context_str}\n")
    return {
        "context": context_str,
        "args": {
            "retrieved_docs": [doc.metadata for doc in context]
        }
    }

def rag_node(state: StateNode) :
    question: str = str(state["messages"][-1].content) or ""
    context: str = state.get("context") or ""
    if not question:
        raise ValueError("Question is required for RAG node")
    msgs = rag_system_prompt.format_messages(context=context, question=question)
    response = llm.invoke(msgs)
    if response is not None:
        if state["args"] is not None and "retrieved_docs" in state["args"]:
            retrieved_docs = state["args"]["retrieved_docs"]
            print(f"RAG Node - Retrieved Documents Metadata:\n{retrieved_docs}\n")
            refrence_docs = [f"[{doc.get('title')}]({doc.get('url')})\n" for doc in retrieved_docs if doc.get("url")]
            if refrence_docs:
                response.content += "\n\nTham khảo:\n" + "\n".join(refrence_docs)
            else:
                response.content += "\n\nKhông có tài liệu tham khảo nào được cung cấp."
    return {
        "messages": [response]
    }   


    
