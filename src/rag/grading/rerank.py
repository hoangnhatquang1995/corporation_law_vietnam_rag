from pydantic import BaseModel, Field
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import BaseChatModel
from rag.llm import get_llm
from rag.llm.sentence import get_cross_encoder

rerank_prompt_message = """
    Bạn là một hệ thống đánh giá lại độ liên quan của tài liệu retrival được so với truy vấn.
    Thông qua thang điểm từ 0 đến 10, hãy đánh giá lại mức độ liên quan của tài liệu so với truy vấn, với 0 là không liên quan và 10 là rất liên quan. 
    Nên tham chiếu cả ngữ cảnh và query, chứ không chỉ dựa vào keywords.
    Hãy chỉ trả về điểm số mà không cần giải thích.
    Query: {query_embedding}
    Document: {document_embedding}
    Điểm số đánh giá lại:
"""

class RerankScore(BaseModel):
    score : float = Field(..., description="The relevance score of the document to the query.") 

def rerank_documents(query_embedding : str, document_embedding : list[Document], n_top : int = 5) -> list[Document]:
    # return rerank_using_llm(query_embedding, document_embedding, n_top)
    return rerank_using_embedding(query_embedding, document_embedding, n_top)

def rerank_using_embedding(query : str, documents : list[Document], n_top : int = 5) -> list[Document]:
    print(f"Reranking using embedding for query: '{query}' with {len(documents)} documents.")
    rerank_encoder = get_cross_encoder()
    setences = [[query, doc.page_content] for doc in documents]
    scores = rerank_encoder.predict(setences)
    result = list(zip(documents, scores))
    print(f"Reranking results (before sorting):\n{result}\n")
    reranked = sorted(
        zip(documents, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    return [doc for doc, score in reranked[:n_top]]


def rerank_using_llm(query : str, documents : list[Document], n_top : int = 5) -> list[Document]:
    llm = get_llm()
    prompt = SystemMessagePromptTemplate.from_template(rerank_prompt_message)
    
    scored_documents = []
    prompts_batch =[
        prompt.format_messages(query_embedding=query, document_embedding=doc.page_content)
        for doc in documents
    ]

    llm_score = llm.with_structured_output(RerankScore, method="function_calling")
    try:
        # Gọi batch song song
        responses = llm_score.batch(prompts_batch)
    except Exception as e:
        print(f"[LLM Rerank Error]: {e}")
        responses =[RerankScore(score=0.0)] * len(documents)

    scored_documents =[]
    for doc, response in zip(documents, responses):
        # Đã fix lỗi response["score"] -> response.score
        score = float(response.score) if response else 0.0
        doc.metadata["rerank_score"] = score
        scored_documents.append((doc, score))
    scored_documents.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_documents[:n_top]]
