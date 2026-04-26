from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

from rag.llm import llm
from .system_prompts import prompt

def create_qa_chain():
    return create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

def create_rag_chain(retriever):
    qa_chain = create_qa_chain()
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain= qa_chain
    )
    return RunnablePassthrough.assign(
        question=lambda values: values["input"]
    ) | retrieval_chain

