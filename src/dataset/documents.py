
import re
from typing import Optional, cast

import pandas as pd

from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from settings.settings import DATASET_CACHE_DIR
from datasets import load_dataset

DATASET_NAME = "vohuutridung/vietnamese-legal-documents"
LEGAL_STRUCTURE_PATTERN = re.compile(
    r"(?m)^\s*(?:PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục|TIỂU MỤC|Tiểu mục|Điều)\s+[^\n]*"
)
LEGAL_SUBSECTION_SEPARATORS = [
    r"\nKhoản\s+\d+[.:]?\s*",
    r"\nĐiểm\s+[a-zđ][).]?\s*",
    r"\n\d+\.\s+",
    r"\n[a-zđ]\)\s+",
    r"\n\n",
    r"\n",
    r"\.\s+",
    r";\s+",
    r",\s+",
    r"\s+",
]


def _build_split(start: Optional[int] = None , limit: Optional[int] = None) -> str:
    if start is None and limit is None:
        return "data"
    if start is None:
        return f"data[:{limit}]"
    if limit is None:
        return f"data[{start}:]"
    return f"data[{start}:{start + limit}]"


def _load_dataset_frame(config_name: str, start: Optional[int]= None, limit: Optional[int] = None) -> pd.DataFrame:
    dataset = load_dataset(
        DATASET_NAME,
        config_name,
        cache_dir=str(DATASET_CACHE_DIR),
        split=_build_split(start, limit),
    )
    return cast(pd.DataFrame, dataset.to_pandas())


def load_vietnamese_legal_datasets(start: Optional[int]= None, limit: Optional[int] = None) -> pd.DataFrame:
    df_content = _load_dataset_frame("content",start, limit)
    df_metadata = _load_dataset_frame("metadata", start,limit)
    return df_content.merge(df_metadata, on="id")


def load_vietnamese_legal_documents(start: Optional[int]= None, limit: Optional[int] = None) -> list[Document]:
    merged_rows = load_vietnamese_legal_datasets(start,limit).to_dict(orient="records")
    docs = []
    for row in merged_rows:
        page_content = row.pop("content")
        docs.append(Document(page_content=page_content, metadata=row))
    return docs

def _normalize_legal_text(text: str) -> str:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = re.sub(r"[ \t]+\n", "\n", normalized_text)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)
    return normalized_text.strip()


def _split_legal_sections(text: str) -> list[str]:
    normalized_text = _normalize_legal_text(text)
    if not normalized_text:
        return []

    matches = list(LEGAL_STRUCTURE_PATTERN.finditer(normalized_text))
    if not matches:
        return [normalized_text]

    sections = []
    preamble = normalized_text[:matches[0].start()].strip()
    if preamble:
        sections.append(preamble)

    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        section = normalized_text[match.start():section_end].strip()
        if section:
            sections.append(section)

    return sections


def _merge_small_chunks(split_docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    if not split_docs:
        return []

    min_chunk_size = max(220, chunk_size // 5)
    max_merged_size = chunk_size + chunk_overlap
    merged_docs = [split_docs[0]]

    for chunk in split_docs[1:]:
        previous_chunk = merged_docs[-1]
        can_merge = len(previous_chunk.page_content) + len(chunk.page_content) <= max_merged_size
        should_merge = len(previous_chunk.page_content) < min_chunk_size or len(chunk.page_content) < min_chunk_size

        if can_merge and should_merge:
            merged_docs[-1] = Document(
                page_content=f"{previous_chunk.page_content}\n{chunk.page_content}".strip(),
                metadata=previous_chunk.metadata.copy(),
            )
            continue

        merged_docs.append(chunk)

    return merged_docs


def chunk_documents(documents, chunk_size=1200, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SUBSECTION_SEPARATORS,
        keep_separator="start",
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_docs = []
    for doc in documents:
        legal_sections = _split_legal_sections(doc.page_content)
        if not legal_sections:
            continue

        split_docs = text_splitter.create_documents(
            legal_sections,
            metadatas=[doc.metadata.copy() for _ in legal_sections],
        )
        split_docs = _merge_small_chunks(split_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunk_count = len(split_docs)
        for chunk_index, chunk in enumerate(split_docs):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_count"] = chunk_count
            chunk.metadata["chunk_method"] = "legal_recursive"

        chunked_docs.extend(split_docs)

    return chunked_docs

