from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Optional

from enum import Enum
import os

class EmbeddingProvider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"

def embedding_factory(provider: EmbeddingProvider, model_name: Optional[str] = None, **kwargs) -> Embeddings:
    if provider == EmbeddingProvider.HUGGINGFACE:
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    
def get_embedding_dim(embedder: Embeddings) -> int:
    if embedder is None:
        raise ValueError("Embedder must be provided to get embedding dimension.")
    return len(embedder.embed_query("test query"))
