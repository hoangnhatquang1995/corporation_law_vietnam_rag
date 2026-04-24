
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from datasets import load_dataset 

DATASET_NAME = "vohuutridung/vietnamese-legal-documents"

def load_vietnamese_legal_datasets():
    df_content = load_dataset(DATASET_NAME, "content", cache_dir= "../../datasets/dataset_cache")["data"].to_pandas()
    df_metadata = load_dataset(DATASET_NAME, "metadata", cache_dir= "../../datasets/dataset_cache")["data"].to_pandas()
    df = df_content.merge(df_metadata, on="id")
    return df

def load_vietnamese_legal_documents():
    df_content = load_dataset(DATASET_NAME, "content", cache_dir= "../../datasets/dataset_cache")["data"].to_pandas().head(10)
    df_metadata = load_dataset(DATASET_NAME, "metadata", cache_dir= "../../datasets/dataset_cache")["data"].to_pandas().head(10)
    docs = [
       Document(
            page_content=df_content[df_content["id"] == id]["content"].values[0],
            metadata = df_metadata[df_metadata["id"] == id].to_dict(orient="records")[0]
        )
        for id in df_content["id"].tolist()
    ]
    return docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.create_documents([doc.page_content for doc in documents], metadatas=[doc.metadata for doc in documents])
    return chunked_docs

