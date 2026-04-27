from . import embeddings
from . import models
from langchain.chat_models import BaseChatModel
from .sentence import get_cross_encoder

_llm : BaseChatModel | None = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = models.get_llm_model(models.LLMProvider.Cloud.DEEPSEEK, "deepseek-chat")
    return _llm