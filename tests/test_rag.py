from datetime import datetime, timedelta, timezone
from app.rag.ingestion import ingest_news_article
from app.rag.vector_store import VectorStore


def test_rag_recency_decay_filter(tmp_path):
    db_file = tmp_path / "test_rag.db"
    store = VectorStore(db_path=db_file)

    now = datetime.now(timezone.utc)
    recent_ts = now.isoformat()
    old_ts = (now - timedelta(hours=96)).isoformat()  # 96h old (>72h)

    # Ingest recent news
    ingest_news_article(
        headline="Reliance quarterly results strong",
        body="Reliance announced robust earnings driven by retail and digital services growth.",
        symbol="RELIANCE",
        source="ET Markets",
        published_at=recent_ts,
        vstore=store,
    )

    # Ingest 96h old news
    ingest_news_article(
        headline="Old Reliance market rumor",
        body="Previous week rumors regarding Reliance retail stake sale.",
        symbol="RELIANCE",
        source="ET Markets",
        published_at=old_ts,
        vstore=store,
    )

    # Retrieve with 72h window (should include recent, exclude 96h old news)
    chunks_72h = store.retrieve(
        query="Reliance earnings", symbol="RELIANCE", recency_hours=72.0
    )
    assert len(chunks_72h) > 0
    headlines = [c["metadata"].get("headline") for c in chunks_72h]
    assert "Reliance quarterly results strong" in headlines
    assert "Old Reliance market rumor" not in headlines
