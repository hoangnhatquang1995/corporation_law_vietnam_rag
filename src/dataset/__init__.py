from . import documents
from . import vectorstore
from . import sql
from rag.llm.embeddings import embedding_factory, EmbeddingProvider

from settings.settings import EMBEDDING_MODEL, PERSIST_DIR, PROJECT_ROOT

SQL_DIR = PROJECT_ROOT / "db" / "sql"
SQL_DIR.mkdir(parents=True, exist_ok=True)

documentDB = vectorstore.VectorStoreDB(
    type=vectorstore.VectorStoreType.QDRANT,
    name="vietnamese_legal_docs",
    embedder=embedding_factory(
        EmbeddingProvider.HUGGINGFACE,
        EMBEDDING_MODEL,
    ),
    path=str(PROJECT_ROOT / "db" / PERSIST_DIR),
)

vietnamLawSql = sql.SQLiteDatabase(name="vietnam_laws.db", path=str(SQL_DIR), model=sql.VietnamLawModel)
logEntrySql   = sql.SQLiteDatabase(name="log_entry.db", path=str(SQL_DIR), model=sql.LogEntryModel)
chatroomSQL   = sql.SQLiteDatabase(name="chatroom", path=str(SQL_DIR), model=sql.ChatroomModel)

