from typing import Any
from pydantic import BaseModel, Field
from enum import Enum

class MessageType(str, Enum):
    LEGAL_QUESTION = "legal_question"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CHITCHAT = "chitchat"

class LLMAnswer(BaseModel):
    answer: MessageType = Field(..., description="Xác định loại câu trả lời dựa trên lịch sử hội thoại và câu hỏi hiện tại. Các loại có thể là 'legal_question', 'general_knowledge', hoặc 'chitchat'.")
    message: str = Field(..., description="Câu trả lời được tạo ra bởi LLM dựa trên lịch sử hội thoại và câu hỏi hiện tại.")

    