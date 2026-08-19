import json
from typing import Any, Dict, List


def build_analysis_prompt(
    query: str,
    symbol: str,
    market_snapshot: Dict[str, Any],
    rag_chunks: List[Dict[str, Any]],
    backtest_data: Dict[str, Any],
) -> str:
    """Build structured prompt injecting pre-computed numbers, RAG evidence chunks, and backtest results."""

    rag_text = ""
    if rag_chunks:
        rag_lines = []
        for c in rag_chunks:
            source = c.get("source", "RAG")
            corpus = c.get("corpus_type", "doc")
            ts = c.get("timestamp", "")
            rag_lines.append(
                f"- [{corpus.upper()} | Source: {source} | Time: {ts}] {c.get('content')}"
            )
        rag_text = "\n".join(rag_lines)
    else:
        rag_text = "No retrieved external news or theoretical chunks found."

    prompt = f"""You are Plutus, an advanced AI Trading Research Assistant.
Your core principle is DECISION SUPPORT with TRANSPARENT REASONING, NOT signal generation.
Do not calculate numbers yourself. All numerical calculations are provided by Python tools.

=== INPUT CONTEXT ===
SYMBOL: {symbol}
QUERY: {query}

[1] LIVE MARKET SNAPSHOT (Source of Truth for Numbers):
- LTP: {market_snapshot.get('ltp')}
- Open: {market_snapshot.get('open')}, High: {market_snapshot.get('high')}, Low: {market_snapshot.get('low')}
- IV (Implied Volatility): {market_snapshot.get('iv')}%
- PCR (Put-Call Ratio): {market_snapshot.get('pcr')}
- OI Trend: {market_snapshot.get('oi_trend')}
- Key Support: {market_snapshot.get('support')}
- Key Resistance: {market_snapshot.get('resistance')}
- Snapshot Time: {market_snapshot.get('timestamp')}

[2] RAG KNOWLEDGE & NEWS CHUNKS:
{rag_text}

[3] HISTORICAL BACKTEST METRICS:
- Strategy Sharpe Ratio: {backtest_data.get('sharpe')}
- Win Rate: {backtest_data.get('win_rate')}%
- Max Drawdown: {backtest_data.get('max_drawdown')}%
- Sample Size: {backtest_data.get('sample_size')} trades
- Caveat: {backtest_data.get('caveat')}

=== INSTRUCTIONS & OUTPUT FORMAT ===
Analyze the input data and provide a structured JSON response. You MUST follow this exact JSON schema:

{{
  "symbol": "{symbol}",
  "as_of": "{market_snapshot.get('timestamp')}",
  "current_data": {{
    "ltp": {market_snapshot.get('ltp', 0.0)},
    "iv": {market_snapshot.get('iv', 'null') if market_snapshot.get('iv') is not None else 'null'},
    "pcr": {market_snapshot.get('pcr', 'null') if market_snapshot.get('pcr') is not None else 'null'},
    "oi_trend": "{market_snapshot.get('oi_trend', 'neutral')}"
  }},
  "bullish_factors": [
    {{"point": "<evidence statement>", "source": "<source name>"}}
  ],
  "bearish_factors": [
    {{"point": "<evidence statement>", "source": "<source name>"}}
  ],
  "backtest_context": {{
    "sharpe": {backtest_data.get('sharpe', 'null') if backtest_data.get('sharpe') is not None else 'null'},
    "sample_size": {backtest_data.get('sample_size', 'null') if backtest_data.get('sample_size') is not None else 'null'},
    "win_rate": {backtest_data.get('win_rate', 'null') if backtest_data.get('win_rate') is not None else 'null'},
    "max_drawdown": {backtest_data.get('max_drawdown', 'null') if backtest_data.get('max_drawdown') is not None else 'null'},
    "caveat": "{backtest_data.get('caveat')}"
  }},
  "key_levels": {{
    "support": {market_snapshot.get('support', 0.0)},
    "resistance": {market_snapshot.get('resistance', 0.0)}
  }},
  "invalidation_conditions": "<specific price, PCR, or catalyst conditions that invalidate this view>",
  "confidence": "<low | moderate | high — with justification>",
  "explicit_note": "This is decision-support analysis, not a recommendation to buy or sell."
}}

Respond ONLY with valid JSON matching the schema above. No additional text before or after the JSON.
"""
    return prompt
