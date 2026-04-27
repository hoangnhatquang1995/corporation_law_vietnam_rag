from settings.types import StateNode
from rag.llm.models import get_llm_model, LLMProvider
from rag.llm import llm
from .system_prompts import llm_system_prompt,rag_system_prompt
from rag.grading.rerank import rerank_documents

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
    docs = documentDB.query(question, top_k=5)
    return {
        "args": {
            "retrieved_docs": docs
        }
    }

def rerank_node(state: StateNode):
    question: str = str(state["messages"][-1].content) or ""
    if not question:
        raise ValueError("Question is required for RAG node")
    retrieved_docs = state.get("args", {}).get("retrieved_docs", [])
    if not retrieved_docs:
        raise ValueError("No documents to rerank")
    reranked_docs = rerank_documents(llm, question, retrieved_docs, n_top=2)
    print(f"Rerank Node - Reranked Documents Metadata:\n{[doc for doc in reranked_docs]}\n")
    state["args"]["retrieved_docs"] = reranked_docs
    return state

def rag_node(state: StateNode) :
    question: str = str(state["messages"][-1].content) or ""
    retrieved_docs = state.get("args", {}).get("retrieved_docs", [])
    context = "\n\n".join([f"""Document {doc.metadata['title']} - Number {doc.metadata['document_number']}:\n{doc.page_content}""" for i, doc in enumerate(retrieved_docs)])
    if not question:
        raise ValueError("Question is required for RAG node")
    msgs = rag_system_prompt.format_messages(context=context, question=question)
    response = llm.invoke(msgs)
    if response is not None:
        if state["args"] is not None and "retrieved_docs" in state["args"]:
            retrieved_docs = state["args"]["retrieved_docs"]
            print(f"RAG Node - Retrieved Documents Metadata:\n{retrieved_docs}\n")
            refrence_docs = [f"- [{doc.metadata.get('title')}]({doc.metadata.get('url')})\n" for doc in retrieved_docs if doc.metadata.get("url")]
            if refrence_docs:
                response.content += "\n\nTham khảo:\n" + "\n".join(refrence_docs)
            else:
                response.content += "\n\nKhông có tài liệu tham khảo nào được cung cấp."
    return {
        "messages": [response]
    }   


    
