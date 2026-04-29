import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_DIR = PROJECT_ROOT / "datasets" / "dataset_cache"
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "model_cache"
SENTENCE_CACHE_DIR = PROJECT_ROOT / "models" / "sentence_cache"
HUGGINGFACE_HUB_CACHE_DIR = MODEL_CACHE_DIR / "hub"
TRANSFORMERS_CACHE_DIR = MODEL_CACHE_DIR / "transformers"
SENTENCE_TRANSFORMERS_CACHE_DIR = MODEL_CACHE_DIR / "sentence_transformers"
REQUIRED_CONSULT_CLASSIFICATION_MODEL_DIR = PROJECT_ROOT / "models" / "fine-tune_model" / "legal_consult_phobert"

for cache_dir in (
	DATASET_CACHE_DIR,
	MODEL_CACHE_DIR,
	HUGGINGFACE_HUB_CACHE_DIR,
	TRANSFORMERS_CACHE_DIR,
	SENTENCE_TRANSFORMERS_CACHE_DIR,
):
	cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HUGGINGFACE_HUB_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HUGGINGFACE_HUB_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_TRANSFORMERS_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL  = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# RERANKER_MODEL =  "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SENTENCE_TRANFORMER_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-r1"
PERSIST_DIR = "vectorstore"
CHUNK_SIZE = 1200
CHUNK_OVERLAPPED = 150

