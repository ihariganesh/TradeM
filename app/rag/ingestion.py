import hashlib
import re
from datetime import datetime, timezone
from typing import List, Optional
from app.rag.vector_store import DocumentChunk, VectorStore, vector_store


def chunk_text(
    text: str, chunk_size: int = 600, overlap: int = 100
) -> List[str]:
    """Chunk text into ~600 character blocks with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
        if i >= len(words):
            break
    return chunks or [text]


def ingest_static_document(
    title: str,
    content: str,
    source: str = "Trading Literature / Strategy Doc",
    vstore: Optional[VectorStore] = None,
) -> int:
    """Ingest static book/strategy text into RAG vector store."""
    target_store = vstore or vector_store
    chunks = chunk_text(content, chunk_size=300, overlap=50)
    doc_chunks = []
    for idx, c in enumerate(chunks):
        doc_id = f"static_{hashlib.md5((title + str(idx)).encode()).hexdigest()[:12]}"
        doc_chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                content=f"[{title}] {c}",
                corpus_type="static",
                source=source,
                timestamp="2026-01-01T00:00:00+00:00",  # static
                metadata={"title": title, "chunk_index": idx},
            )
        )
    target_store.add_chunks(doc_chunks)
    return len(doc_chunks)


def ingest_news_article(
    headline: str,
    body: str,
    symbol: Optional[str] = None,
    source: str = "ET Markets / Reuters",
    published_at: Optional[str] = None,
    vstore: Optional[VectorStore] = None,
) -> int:
    """Ingest live news article into RAG vector store."""
    target_store = vstore or vector_store
    full_text = f"HEADLINE: {headline}\n\n{body}"
    chunks = chunk_text(full_text, chunk_size=200, overlap=30)
    ts = published_at or datetime.now(timezone.utc).isoformat()

    doc_chunks = []
    for idx, c in enumerate(chunks):
        doc_id = f"news_{hashlib.md5((headline + str(idx) + ts).encode()).hexdigest()[:12]}"
        doc_chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                content=c,
                corpus_type="news",
                symbol=symbol,
                source=source,
                timestamp=ts,
                metadata={"headline": headline, "chunk_index": idx},
            )
        )
    target_store.add_chunks(doc_chunks)
    return len(doc_chunks)


def seed_initial_knowledge_base(vstore: Optional[VectorStore] = None):
    """Seed initial trading books and strategy rules into static RAG vector store."""
    books = [
        (
            "Options Volatility & Pricing Principles",
            "Option Greeks measure sensitivity to market variables: Delta measures directional risk, Gamma measures rate of change of Delta, Vega measures sensitivity to Implied Volatility (IV), and Theta measures daily time decay. High IV relative to historical volatility indicates expensive options suited for credit strategies (iron condor, sell strangles), while low IV favors debit spreads or long options. PCR (Put-Call Ratio) above 1.3 indicates strong bullish sentiment or put writing support, while PCR below 0.7 reflects extreme bearishness or call writing resistance.",
        ),
        (
            "Market Dynamics & Price Action Theory",
            "Support and resistance levels represent supply and demand imbalances. Breakouts accompanied by volume at least 2x the 20-day average confirm institutional participation. Open Interest (OI) buildup alongside rising price signals fresh long position buildup. Rising price with falling OI indicates short covering. Falling price with rising OI signals aggressive short buildup.",
        ),
        (
            "NSE F&O Trading & Risk Management",
            "Never rely on a single technical or news catalyst. Always analyze contradicting factors before entering F&O positions. Invalidation conditions must be specified before entering any trade (e.g. key technical support break or PCR flip below threshold). Short backtest windows (< 1 year or < 100 sample trades) yield highly unstable Sharpe ratios.",
        ),
    ]

    for title, text in books:
        ingest_static_document(
            title, text, source="Option & Trading Literature", vstore=vstore
        )
