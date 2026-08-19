import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.market_data.scheduler import refresh_watchlist_snapshots
from app.orchestrator.service import orchestrator_service
from app.rag.ingestion import ingest_news_article, ingest_static_document, seed_initial_knowledge_base
from app.scanner.service import scanner_service
from app.schemas.analysis import SymbolAnalysisRequest, SymbolAnalysisResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TradeM.Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing TradeM Trading Research Assistant...")
    # Seed initial trading books and theory into RAG vector store
    seed_initial_knowledge_base()
    # Refresh market snapshots
    refresh_watchlist_snapshots()
    logger.info("TradeM ready.")
    yield
    logger.info("Shutting down TradeM...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Trading Research Assistant — System Architecture implementation with FastAPI, RAG, Ollama/Plutus, and Pydantic Section 5 Schema enforcement.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {
        "service": settings.APP_NAME,
        "status": "online",
        "docs": "/docs",
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.OLLAMA_MODEL,
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/analyze", response_model=SymbolAnalysisResponse)
def analyze_symbol(request: SymbolAnalysisRequest):
    """POST /analyze endpoint for request/response decision support analysis."""
    try:
        response = orchestrator_service.analyze_symbol(
            query=request.query,
            symbol=request.symbol,
            instrument_type=request.instrument_type,
        )
        return response
    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed processing analysis: {str(e)}"
        )


class IngestNewsRequest(BaseModel):
    headline: str
    body: str
    symbol: Optional[str] = None
    source: str = "ET Markets / Reuters"


@app.post("/rag/ingest/news")
def ingest_news(req: IngestNewsRequest):
    num_chunks = ingest_news_article(
        headline=req.headline,
        body=req.body,
        symbol=req.symbol,
        source=req.source,
    )
    return {
        "message": f"Successfully ingested news article into RAG vector store.",
        "chunks_created": num_chunks,
    }


class IngestBookRequest(BaseModel):
    title: str
    content: str
    source: str = "Trading Literature / Strategy Doc"


@app.post("/rag/ingest/book")
def ingest_book(req: IngestBookRequest):
    num_chunks = ingest_static_document(
        title=req.title, content=req.content, source=req.source
    )
    return {
        "message": f"Successfully ingested static document into RAG vector store.",
        "chunks_created": num_chunks,
    }


@app.post("/scanner/trigger")
def trigger_scanner(background_tasks: BackgroundTasks):
    """Trigger background scan across watchlist."""
    alerts = scanner_service.run_scan(
        orchestrator_service=orchestrator_service
    )
    return {
        "message": "Market scan complete.",
        "alerts_triggered_count": len(alerts),
        "alerts": alerts,
    }


@app.get("/scanner/watchlist")
def get_watchlist():
    return {
        "watchlist": settings.WATCHLIST,
        "interval_minutes": settings.SCANNER_INTERVAL_MINUTES,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
