from settings.types import StateNode
from rag.llm.models import get_llm_model, LLMProvider
from rag.llm import get_llm
from langchain_core.messages import AIMessage
from .system_prompts import rag_system_prompt,assistant_system_prompt,llm_answer_system_prompt
from rag.grading.rerank import rerank_documents
from rag.agent.state import LLMAnswer,MessageType

from dataset import documentDB 

def llm_node(state: StateNode):
    messages = [
        llm_answer_system_prompt,
        *state["messages"]
    ]
    llm_answer = get_llm().with_structured_output(LLMAnswer, method="function_calling")
    raw_response = llm_answer.invoke(messages)
    response = LLMAnswer.model_validate(raw_response) if raw_response is not None else None
    message_type = response.answer if response else MessageType.GENERAL_KNOWLEDGE
    if message_type==MessageType.LEGAL_QUESTION:
        messages = []
    else:
        messages = [AIMessage(content=response.message) if response else AIMessage(content="Xin lỗi, tôi không thể phân tích câu hỏi của bạn.")]
    return {
        "type": message_type,
        "messages": messages
    }

def retriving_node(state: StateNode):
    question: str = str(state["messages"][-1].content) or ""
    if not question:
        raise ValueError("Question is required for RAG node")
    docs = documentDB.query(question, top_k=15)
    return {
        "args": {
            "retrieved_docs": docs
        }
    }

def rerank_node(state: StateNode):
    question: str = str(state["messages"][-1].content) or ""
    if not question:
        raise ValueError("Question is required for RAG node")
    args = state.get("args") or {}
    retrieved_docs = args.get("retrieved_docs", [])
    if not retrieved_docs:
        raise ValueError("No documents to rerank")
    reranked_docs = rerank_documents(question, retrieved_docs, n_top=2)
    return {
        "args": {
            **args,
            "retrieved_docs": reranked_docs,
        }
    }

def rag_node(state: StateNode) :
    chat_history = state["messages"]
    question: str = str(chat_history[-1].content) or ""
    args = state.get("args") or {}
    retrieved_docs = args.get("retrieved_docs", [])
    context = "\n\n".join([f"""Document {doc.metadata['title']} - Number {doc.metadata['document_number']}:\n{doc.page_content}""" for i, doc in enumerate(retrieved_docs)])
    if not question:
        raise ValueError("Question is required for RAG node")
    msgs = [assistant_system_prompt] + rag_system_prompt.format_messages(context=context, question=question)
    response = get_llm().invoke(msgs)
    if response is not None:
        if "retrieved_docs" in args:
            retrieved_docs = args["retrieved_docs"]
            print(f"RAG Node - Retrieved Documents Metadata:\n{retrieved_docs}\n")
            refrence_docs = [f"- [{doc.metadata.get('title')}]({doc.metadata.get('url')})\n" for doc in retrieved_docs if doc.metadata.get("url")]
            if refrence_docs:
                response.content += "\n\nTham khảo:\n" + "\n".join(refrence_docs)
            else:
                response.content += "\n\nKhông có tài liệu tham khảo nào được cung cấp."
    return {
        "messages": [response]
    }   


    
