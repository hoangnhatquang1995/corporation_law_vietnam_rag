import os

from . import embeddings
from . import models
from settings import settings 
from sentence_transformers import CrossEncoder
from settings.settings import RERANKER_MODEL
 
llm  = models.get_llm_model(models.LLMProvider.Cloud.DEEPSEEK,"deepseek-chat")

_rerank_encoder : CrossEncoder | None = None

def get_cross_encoder():
    print(f" SENTENCE_TRANSFORMERS_HOME : {os.getenv('SENTENCE_TRANSFORMERS_HOME')}")
    global _rerank_encoder
    if _rerank_encoder is None:
        _rerank_encoder = CrossEncoder(RERANKER_MODEL)
    return _rerank_encoder