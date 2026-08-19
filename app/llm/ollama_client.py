import json
import logging
from typing import Any, Dict
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with local Ollama API running fine-tuned Plutus."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_analysis(
        self,
        prompt: str,
        symbol: str,
        market_snapshot: Dict[str, Any],
        backtest_data: Dict[str, Any],
        rag_chunks: list,
    ) -> Dict[str, Any]:
        """Call Ollama /api/generate REST endpoint, fallback to pre-structured reasoning if offline."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9},
        }

        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    resp_json = response.json()
                    response_text = resp_json.get("response", "")
                    # Clean json tags if present
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    return json.loads(response_text.strip())
        except Exception as e:
            logger.warning(
                f"Ollama connection error ({e}). Using offline reasoning synthesizer."
            )

        return self._generate_fallback_analysis(
            symbol, market_snapshot, backtest_data, rag_chunks
        )

    def _generate_fallback_analysis(
        self,
        symbol: str,
        market_snapshot: Dict[str, Any],
        backtest_data: Dict[str, Any],
        rag_chunks: list,
    ) -> Dict[str, Any]:
        """Synthesize structured Section 5 output offline when Ollama server is offline."""
        ltp = market_snapshot.get("ltp", 1000.0)
        pcr = market_snapshot.get("pcr", 1.0)
        iv = market_snapshot.get("iv", 20.0)
        oi_trend = market_snapshot.get("oi_trend", "neutral")
        support = market_snapshot.get("support", ltp * 0.985)
        resistance = market_snapshot.get("resistance", ltp * 1.015)

        bullish_factors = []
        bearish_factors = []

        if pcr > 1.0:
            bullish_factors.append({
                "point": f"PCR at {pcr} reflects solid put writing support beneath LTP {ltp}.",
                "source": "Option Chain Analytics",
            })
        else:
            bearish_factors.append({
                "point": f"PCR at {pcr} indicates muted put writing interest.",
                "source": "Option Chain Analytics",
            })

        if oi_trend == "long_buildup":
            bullish_factors.append({
                "point": f"OI trend shows Long Buildup alongside price stabilization at {ltp}.",
                "source": "F&O Market Snapshot",
            })
        elif oi_trend == "short_buildup":
            bearish_factors.append({
                "point": f"OI trend shows Short Buildup near resistance level {resistance}.",
                "source": "F&O Market Snapshot",
            })

        # Add evidence from RAG chunks if available
        for c in rag_chunks[:2]:
            content = c.get("content", "")
            source = c.get("source", "News/Books")
            if "bull" in content.lower() or "growth" in content.lower():
                bullish_factors.append(
                    {"point": content[:120] + "...", "source": source}
                )
            elif "bear" in content.lower() or "risk" in content.lower():
                bearish_factors.append(
                    {"point": content[:120] + "...", "source": source}
                )

        if not bullish_factors:
            bullish_factors.append({
                "point": f"Price holding above key support at {support}.",
                "source": "Technical Analysis",
            })
        if not bearish_factors:
            bearish_factors.append({
                "point": f"Overhead resistance capped near {resistance}.",
                "source": "Technical Analysis",
            })

        return {
            "symbol": symbol,
            "as_of": market_snapshot.get("timestamp", ""),
            "current_data": {
                "ltp": ltp,
                "iv": iv,
                "pcr": pcr,
                "oi_trend": oi_trend,
            },
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
            "backtest_context": {
                "sharpe": backtest_data.get("sharpe"),
                "sample_size": backtest_data.get("sample_size"),
                "win_rate": backtest_data.get("win_rate"),
                "max_drawdown": backtest_data.get("max_drawdown"),
                "caveat": backtest_data.get(
                    "caveat",
                    "Short backtest windows yield unreliable Sharpe ratios.",
                ),
            },
            "key_levels": {"support": support, "resistance": resistance},
            "invalidation_conditions": f"Break below support at {support} or PCR flipping below 0.85.",
            "confidence": "moderate — balanced technical levels with backing PCR support",
            "explicit_note": "This is decision-support analysis, not a recommendation to buy or sell.",
        }


ollama_client = OllamaClient()
