from . import embeddings
from . import models

llm  = models.get_llm_model(models.LLMProvider.Cloud.DEEPSEEK,"deepseek-chat")
