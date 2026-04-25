from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys
from typing import Optional

from enum import Enum

from huggingface_hub import snapshot_download
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from settings.settings import HUGGINGFACE_HUB_CACHE_DIR

MODEL_ALLOW_PATTERNS = ["*.json", "*.txt", "*.model", "*.py", "*.safetensors"]

class EmbeddingProvider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


def _resolve_cached_model_path(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return model_name

    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return str(local_path)

    snapshot_root = (
        HUGGINGFACE_HUB_CACHE_DIR
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
    )
    if not snapshot_root.exists():
        downloaded_snapshot = snapshot_download(
            repo_id=model_name,
            cache_dir=str(HUGGINGFACE_HUB_CACHE_DIR),
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
        return downloaded_snapshot

    snapshots = sorted(
        (path for path in snapshot_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        downloaded_snapshot = snapshot_download(
            repo_id=model_name,
            cache_dir=str(HUGGINGFACE_HUB_CACHE_DIR),
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
        return downloaded_snapshot

    return str(snapshots[0])


def embedding_factory(provider: EmbeddingProvider, model_name: Optional[str] = None, **kwargs) -> Embeddings:
    if provider == EmbeddingProvider.HUGGINGFACE:
        resolved_model_name = _resolve_cached_model_path(model_name)
        return HuggingFaceEmbeddings(model_name=resolved_model_name, **kwargs)
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    
def get_embedding_dim(embedder: Embeddings) -> int:
    if embedder is None:
        raise ValueError("Embedder must be provided to get embedding dimension.")
    return len(embedder.embed_query("test query"))
