import os
import json
from typing import Annotated,TypedDict, List, Optional, Union

from langchain_qdrant import QdrantVectorStore
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest 
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore 
import enum

from rag.llm.embeddings import get_embedding_dim
from settings.settings import QDRANT_URL


class VectorStoreType(enum.Enum):
    QDRANT = "qdrant"
    CHROMA = "chroma"

class VectorStoreDB:
    db : Optional[VectorStore]
    name : Optional[str]
    embedder : Optional[Embeddings]
    type : VectorStoreType
    path : Optional[str]
    client : Optional[QdrantClient]
    timeout : Optional[int]
    qdrant_batch_size : int
    qdrant_wait : bool


    def __init__(self, type: VectorStoreType, name: Optional[str] = None, embedder: Optional[Embeddings] = None, path: Optional[str] = None):
        self.type = type 
        self.name = name
        self.embedder = embedder
        self.path = path
        self.db = None
        self.client = None
        self.timeout = int(os.getenv("QDRANT_TIMEOUT", "120"))
        self.qdrant_batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "16"))
        self.qdrant_wait = os.getenv("QDRANT_UPSERT_WAIT", "false").lower() == "true"
        if self.type == VectorStoreType.QDRANT:
            self.client = QdrantClient(url=QDRANT_URL, timeout=self.timeout)
    
    def init(self, embedder: Optional[Embeddings] = None, path: Optional[str] = None):
        if embedder is not None:
            self.embedder = embedder
        if path is not None:
            self.path = path


    def set_embedder(self, embedder: Embeddings):
        self.embedder = embedder

    def _ensure_ready(self):
        if self.db is None:
            self.build()

    def build(self):
        if self.embedder is None:
            raise ValueError("[1][ERROR] Embedder phải được cung cấp để xây dựng vector store.")
        if self.name is None:
            raise ValueError("[2][ERROR] Tên bộ sưu tập phải được cung cấp để xây dựng vector store.")
        if self.type == VectorStoreType.QDRANT:
            if self.client is None:
                self.client = QdrantClient(url=QDRANT_URL, timeout=self.timeout)
            if not self.client.collection_exists(self.name):
                self.client.recreate_collection(
                    collection_name=self.name,
                    vectors_config=rest.VectorParams(
                        size= get_embedding_dim(self.embedder), 
                        distance=rest.Distance.COSINE
                    )
                )
            self.db = QdrantVectorStore(
                client=self.client,
                collection_name=self.name,
                embedding=self.embedder
            )
            pass
        elif self.type == VectorStoreType.CHROMA:
            self.db = Chroma(
                collection_name=self.name, 
                embedding_function= self.embedder, 
                persist_directory=self.path
            )
            pass
        else:
            raise ValueError(f"Unsupported vector store type: {self.type}")
                
    def add(self, documents: List[Document], batch_size: Optional[int] = None, wait: Optional[bool] = None, timeout: Optional[int] = None):
        self._ensure_ready()
        if self.db is None:
            raise ValueError("[1][ERROR] DB Không được khởi tạo.")
        if self.type == VectorStoreType.QDRANT:
            resolved_batch_size = self.qdrant_batch_size if batch_size is None else batch_size
            resolved_wait = self.qdrant_wait if wait is None else wait
            resolved_timeout = self.timeout if timeout is None else timeout

            # Upsert theo batch nhỏ và không chờ indexing xong để tránh notebook bị ReadTimeout.
            self.db.add_documents(
                documents,
                batch_size=resolved_batch_size,
                wait=resolved_wait,
                timeout=resolved_timeout,
            )
            return
        self.db.add_documents(documents)
        
    def query(self, query: str, top_k: int = 5) -> List[Document]:
        self._ensure_ready()
        if self.db is None:
            raise ValueError("[1][ERROR] DB Không được khởi tạo.")
        results = self.db.similarity_search( query, k=top_k )
        return results

    def delete(self, document_ids: List[str]):
        self._ensure_ready()
        if self.db is None:
            raise ValueError("[1][ERROR] DB Không được khởi tạo.")
        if self.name is None:
            raise ValueError("[2][ERROR] Tên bộ sưu tập phải được cung cấp để xóa tài liệu.")
        if self.type == VectorStoreType.QDRANT:
            #TODO: delete documents from qdrant vector store
            if self.client is None:
                raise ValueError("Qdrant client must be initialized to delete documents.")
            else:
                for doc_id in document_ids:
                    self.client.delete(
                        collection_name=self.name,
                        points_selector=rest.PointIdsList(points=[doc_id])
                    )
            pass
        elif self.type == VectorStoreType.CHROMA:
            #TODO: delete documents from chroma vector store
            self.db.delete(ids=document_ids)
            pass
        else:
            raise ValueError(f"Unsupported vector store type: {self.type}")

    def update(self, documents: List[Document]):
        self._ensure_ready()
        if self.db is None:
            raise ValueError("[1][ERROR] DB Không được khởi tạo.")
        if self.name is None:
            raise ValueError("[2][ERROR] Tên bộ sưu tập phải được cung cấp để cập nhật tài liệu.")
        if self.embedder is None:
            raise ValueError("[3][ERROR] Embedder phải được cung cấp để cập nhật tài liệu.")
        if self.type == VectorStoreType.QDRANT:
            #TODO: update documents in qdrant vector store
            if self.client is None:
                raise ValueError("[1][ERROR] Qdrant client phải được khởi tạo để cập nhật tài liệu.")
            else:
                for doc in documents:
                    metadata_id = doc.metadata.get("id")
                    if metadata_id is None:
                        raise ValueError("[2][ERROR] Mỗi tài liệu phải có 'id' trong metadata để cập nhật.")
                    self.client.upsert(
                        collection_name=self.name,
                        points=[
                            rest.PointStruct(
                                id= metadata_id, 
                                vector= self.embedder.embed_query(doc.page_content), 
                                payload=doc.metadata
                            )
                        ]
                    )
            pass
        elif self.type == VectorStoreType.CHROMA:
            
            pass
        else:
            raise ValueError(f"Unsupported vector store type: {self.type}")

    def as_retriver(self):
        self._ensure_ready()
        if self.db is None:
            raise ValueError("[1][ERROR] DB Không được khởi tạo.")
        if self.type == VectorStoreType.QDRANT: 
            return self.db.as_retriever()
        elif self.type == VectorStoreType.CHROMA:
            return self.db.as_retriever()  
        else:
            raise ValueError(f"Unsupported vector store type: {self.type}")   
    