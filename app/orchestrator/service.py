import logging
from typing import Any, Dict
from app.backtest.engine import backtest_engine
from app.llm.ollama_client import ollama_client
from app.llm.prompts import build_analysis_prompt
from app.market_data.provider import market_provider
from app.rag.vector_store import vector_store
from app.schemas.analysis import SymbolAnalysisResponse

logger = logging.getLogger(__name__)


class OrchestratorService:
    """Core Orchestration Layer orchestrating Market Data, RAG, Backtest, LLM & Schema Validation."""

    def analyze_symbol(
        self, query: str, symbol: str, instrument_type: str = "options"
    ) -> SymbolAnalysisResponse:
        logger.info(f"Orchestrating analysis for {symbol} (query: '{query}')")

        # 1. Fetch Market Snapshot (Python source of truth for numbers)
        snapshot = market_provider.get_snapshot(symbol)

        # 2. RAG Retrieval (news chunks filtered by 72h recency decay, static books unfiltered)
        rag_chunks = vector_store.retrieve(
            query=query, symbol=symbol, k=8, recency_hours=72.0
        )

        # 3. Backtest Metrics (Sharpe, win rate, max drawdown, sample size, mandatory caveats)
        backtest_data = backtest_engine.run_backtest(
            symbol=symbol, strategy="breakout_straddle"
        )

        # 4. Build prompt carrying pre-computed numerical evidence
        prompt = build_analysis_prompt(
            query=query,
            symbol=symbol,
            market_snapshot=snapshot,
            rag_chunks=rag_chunks,
            backtest_data=backtest_data,
        )

        # 5. Call Ollama / Plutus LLM (or offline reasoning fallback)
        raw_response = ollama_client.generate_analysis(
            prompt=prompt,
            symbol=symbol,
            market_snapshot=snapshot,
            backtest_data=backtest_data,
            rag_chunks=rag_chunks,
        )

        # 6. Validate & Enforce Section 5 Pydantic Schema (reject/format bare verdicts or invalid fields)
        validated_response = SymbolAnalysisResponse(**raw_response)

        return validated_response


orchestrator_service = OrchestratorService()
