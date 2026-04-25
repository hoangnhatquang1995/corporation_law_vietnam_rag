
from typing import Optional, cast

import pandas as pd

from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from settings.settings import DATASET_CACHE_DIR
from datasets import load_dataset

DATASET_NAME = "vohuutridung/vietnamese-legal-documents"


def _build_split(limit: Optional[int] = None) -> str:
    return "data" if limit is None else f"data[:{limit}]"


def _load_dataset_frame(config_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    dataset = load_dataset(
        DATASET_NAME,
        config_name,
        cache_dir=str(DATASET_CACHE_DIR),
        split=_build_split(limit),
    )
    return cast(pd.DataFrame, dataset.to_pandas())


def load_vietnamese_legal_datasets(limit: Optional[int] = None) -> pd.DataFrame:
    df_content = _load_dataset_frame("content", limit)
    df_metadata = _load_dataset_frame("metadata", limit)
    return df_content.merge(df_metadata, on="id")


def load_vietnamese_legal_documents(limit: Optional[int] = None) -> list[Document]:
    merged_rows = load_vietnamese_legal_datasets(limit).to_dict(orient="records")
    docs = []
    for row in merged_rows:
        page_content = row.pop("content")
        docs.append(Document(page_content=page_content, metadata=row))
    return docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.create_documents([doc.page_content for doc in documents], metadatas=[doc.metadata for doc in documents])
    return chunked_docs

