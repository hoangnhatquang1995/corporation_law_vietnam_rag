from pydantic import BaseModel, Field
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import BaseChatModel

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

def rerank_documents(llm : BaseChatModel, query_embedding : str, document_embedding : list[Document], n_top : int = 5) -> list[Document]:
    prompt = SystemMessagePromptTemplate.from_template(rerank_prompt_message)
    llm_score = llm.with_structured_output(RerankScore)
    scored_documents = []
    for doc in document_embedding:
        doc_prompt = prompt.format_messages(
            query_embedding=query_embedding, 
            document_embedding=doc.page_content
        )
        try:
            response = llm_score.invoke(doc_prompt)
            score = float(response["score"])
        except Exception as e:
            score = 0.0
        scored_documents.append((doc, score))
    scored_documents.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_documents[:n_top]]