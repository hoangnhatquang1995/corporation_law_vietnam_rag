from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from rag.llm import llm
from .system_prompts import prompt

def create_rag_chain(retriever):
    qa_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=prompt
    )
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_documents_chain= qa_chain
    )
    return retrieval_chain

