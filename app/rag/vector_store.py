import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import settings
from app.rag.embeddings import EmbeddingModel, cosine_similarity


class DocumentChunk:

    def __init__(
        self,
        doc_id: str,
        content: str,
        corpus_type: str,  # 'static' or 'news'
        symbol: Optional[str] = None,
        source: str = "unknown",
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.doc_id = doc_id
        self.content = content
        self.corpus_type = corpus_type
        self.symbol = symbol
        self.source = source
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}


class VectorStore:

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.db_path = db_path or (settings.DATA_DIR / "vector_store.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or EmbeddingModel()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    corpus_type TEXT NOT NULL,
                    symbol TEXT,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding_json TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_type_ts 
                ON vector_chunks(corpus_type, timestamp);
            """)
            conn.commit()

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for chunk, emb in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO vector_chunks
                    (doc_id, content, corpus_type, symbol, source, timestamp, metadata_json, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.doc_id,
                        chunk.content,
                        chunk.corpus_type,
                        chunk.symbol,
                        chunk.source,
                        chunk.timestamp,
                        json.dumps(chunk.metadata),
                        json.dumps(emb),
                    ),
                )
            conn.commit()

    def retrieve(
        self,
        query: str,
        symbol: Optional[str] = None,
        k: int = 8,
        recency_hours: float = settings.NEWS_RECENCY_HOURS,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks. Applies 72h recency decay filter to news chunks."""
        query_emb = self.embedding_model.embed_query(query)
        now = datetime.now(timezone.utc)

        results = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, content, corpus_type, symbol, source, timestamp, metadata_json, embedding_json FROM vector_chunks"
            )
            rows = cursor.fetchall()

            for row in rows:
                corpus_type = row["corpus_type"]
                doc_symbol = row["symbol"]
                ts_str = row["timestamp"]

                # Symbol filtering (if symbol specified and chunk has a specific symbol tag)
                if symbol and doc_symbol and doc_symbol.upper() != symbol.upper():
                    continue

                # Recency decay filter for news chunks (>72 hours dropped)
                if corpus_type == "news":
                    try:
                        chunk_time = datetime.fromisoformat(ts_str)
                        if chunk_time.tzinfo is None:
                            chunk_time = chunk_time.replace(tzinfo=timezone.utc)
                        age_hours = (now - chunk_time).total_seconds() / 3600.0
                        if age_hours > recency_hours:
                            continue
                    except ValueError:
                        pass

                emb = json.loads(row["embedding_json"])
                sim = cosine_similarity(query_emb, emb)

                # Bonus for symbol match
                if symbol and doc_symbol and doc_symbol.upper() == symbol.upper():
                    sim += 0.05

                results.append({
                    "doc_id": row["doc_id"],
                    "content": row["content"],
                    "corpus_type": corpus_type,
                    "symbol": doc_symbol,
                    "source": row["source"],
                    "timestamp": ts_str,
                    "score": round(sim, 4),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                })

        # Sort by similarity score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


vector_store = VectorStore()
