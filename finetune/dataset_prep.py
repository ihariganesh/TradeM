import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = """You are Plutus, an AI Trading Research Assistant.
Your core principle is DECISION SUPPORT with TRANSPARENT REASONING, NOT signal generation.
Do not calculate numbers yourself. Analyze provided numerical snapshot, RAG chunks, and backtest results.
Always output valid JSON conforming strictly to the Section 5 schema carrying bullish/bearish evidence with sources, key support/resistance levels, invalidation conditions, confidence justification, and mandatory decision-support disclaimer."""

# Diversity pools for realistic RAG context chunks
NEWS_POOL = [
    ("RELIANCE", "bull", "RELIANCE Q3 net profit rises 11% YoY on strong retail and telecom growth", "ET Markets"),
    ("RELIANCE", "bear", "Crude oil margin pressure may impact RELIANCE O2C segment earnings", "Reuters India"),
    ("NIFTY", "bull", "FII inflows resume as global central banks signal dovish stance", "Business Standard"),
    ("NIFTY", "bear", "Geopolitical tension in Middle East sparks volatility across Asian markets", "Moneycontrol"),
    ("BANKNIFTY", "bull", "RBI maintains accommodative liquidity stance; PSU banks lead rally", "ET Markets"),
    ("BANKNIFTY", "bear", "NPA concerns resurface in select regional lenders after audit review", "Reuters India"),
    ("TCS", "bull", "TCS signs $1.2B digital transformation deal with European retail giant", "ET Markets"),
    ("TCS", "bear", "Tech spending slowdown in North America may cap IT sector margins", "Moneycontrol"),
    ("INFY", "bull", "Infosys raises full-year revenue guidance to 4.5%-5.0%", "Business Standard"),
    ("INFY", "bear", "Attrition spikes slightly in Q3; wage hikes to pressure operating margin", "ET Markets"),
    ("HDFCBANK", "bull", "HDFC Bank deposit growth accelerates to 18% YoY", "ET Markets"),
    ("HDFCBANK", "bear", "Net Interest Margin (NIM) compresses by 12 bps quarter-on-quarter", "Reuters India"),
]

BOOK_PRINCIPLES = [
    ("PCR above 1.30 reflects strong put writing floor; PCR below 0.70 signals call resistance floor", "Option Volatility & Pricing Book"),
    ("Breakout above resistance requires volume > 1.8x 20-day average for institutional confirmation", "Price Action Dynamics"),
    ("Price increase coupled with OI increase confirms aggressive long accumulation", "F&O Analytics Manual"),
    ("Price decline with rising OI signals active short buildup near supply zones", "F&O Analytics Manual"),
    ("High Implied Volatility (IV percentile > 80) favors credit spreads over long option purchases", "Option Strategy Guide"),
    ("Never rely on a single technical catalyst; always verify contradicting option chain signals", "NSE Trading Risk Management"),
]


def generate_varied_synthetic_example(symbol: str, base_price: float) -> Dict[str, Any]:
    """Generate rich, non-memorized training pair with signal conflicts, dynamic confidence, varying backtest data, and RAG text context."""

    # 1. Vary Price & Numerical Snapshot
    variation = random.uniform(-0.025, 0.025)
    ltp = round(base_price * (1 + variation), 2)
    support = round(ltp * random.uniform(0.975, 0.990), 2)
    resistance = round(ltp * random.uniform(1.010, 1.025), 2)
    iv = round(random.uniform(16.0, 32.0), 1)

    # 2. Introduce Signal Agreement OR Conflict
    conflict_mode = random.choice(["aligned_bullish", "aligned_bearish", "conflicting_mixed", "extreme_divergence"])

    if conflict_mode == "aligned_bullish":
        pcr = round(random.uniform(1.15, 1.45), 2)
        oi_trend = "long_buildup"
        news_bias = "bull"
    elif conflict_mode == "aligned_bearish":
        pcr = round(random.uniform(0.65, 0.85), 2)
        oi_trend = "short_buildup"
        news_bias = "bear"
    elif conflict_mode == "conflicting_mixed":
        pcr = round(random.uniform(1.10, 1.30), 2)
        oi_trend = random.choice(["short_buildup", "long_unwinding"])
        news_bias = random.choice(["bull", "bear"])
    else:  # extreme divergence
        pcr = round(random.uniform(0.70, 0.80), 2)
        oi_trend = "long_buildup"
        news_bias = "bull"

    # 3. Dynamic RAG Chunk Injection (2 to 4 chunks)
    selected_rag = []
    symbol_news = [n for n in NEWS_POOL if n[0] == symbol and n[1] == news_bias]
    if not symbol_news:
        symbol_news = [n for n in NEWS_POOL if n[1] == news_bias]

    chosen_news = random.choice(symbol_news)
    selected_rag.append({
        "content": chosen_news[2],
        "corpus_type": "news",
        "source": chosen_news[3],
        "timestamp": "2026-08-19T10:30:00Z",
    })

    # Add a book principle chunk
    chosen_book = random.choice(BOOK_PRINCIPLES)
    selected_rag.append({
        "content": chosen_book[0],
        "corpus_type": "static",
        "source": chosen_book[1],
        "timestamp": "2026-01-01T00:00:00Z",
    })

    # Add second news chunk if possible
    other_news = [n for n in NEWS_POOL if n != chosen_news]
    if other_news and random.random() > 0.4:
        second_n = random.choice(other_news)
        selected_rag.append({
            "content": second_n[2],
            "corpus_type": "news",
            "source": second_n[3],
            "timestamp": "2026-08-19T11:15:00Z",
        })

    # 4. Dynamic Backtest Context (Varying per example)
    sharpe = round(random.uniform(0.85, 2.10), 2)
    sample_size = random.randint(28, 160)
    win_rate = round(random.uniform(49.0, 67.0), 1)
    max_dd = round(random.uniform(-22.0, -7.5), 1)
    backtest_caveat = (
        f"Backtest based on {sample_size} historical trade windows on {symbol}. "
        "Short backtest windows yield unreliable Sharpe ratios and unstable forward returns."
    )

    # 5. Build Dynamic Bullish & Bearish Factors
    bullish_factors = []
    bearish_factors = []

    # PCR Evidence
    if pcr >= 1.10:
        bullish_factors.append({
            "point": f"PCR at {pcr} indicates strong put writing interest and downside floor.",
            "source": "Option Chain Analytics",
        })
    else:
        bearish_factors.append({
            "point": f"Subdued PCR at {pcr} suggests lack of put writing support.",
            "source": "Option Chain Analytics",
        })

    # OI Trend Evidence
    if oi_trend == "long_buildup":
        bullish_factors.append({
            "point": f"Open Interest expansion alongside price stabilization indicates fresh long accumulation.",
            "source": "F&O Market Snapshot",
        })
    elif oi_trend == "short_buildup":
        bearish_factors.append({
            "point": f"Rising Open Interest accompanied by price weakness signals aggressive short buildup.",
            "source": "F&O Market Snapshot",
        })
    else:
        bearish_factors.append({
            "point": f"Short covering / long unwinding pattern shows profit-taking at higher levels.",
            "source": "F&O Market Snapshot",
        })

    # RAG Chunks Integration into Factors
    for r in selected_rag:
        content_snippet = r["content"]
        src = r["source"]
        if r["corpus_type"] == "news":
            if chosen_news[1] == "bull":
                bullish_factors.append({"point": f"News catalyst: {content_snippet}", "source": src})
            else:
                bearish_factors.append({"point": f"News catalyst: {content_snippet}", "source": src})
        elif r["corpus_type"] == "static":
            if "PCR" in content_snippet and pcr >= 1.10:
                bullish_factors.append({"point": f"Theory alignment: {content_snippet}", "source": src})
            else:
                bearish_factors.append({"point": f"Risk guideline: {content_snippet}", "source": src})

    # Ensure both lists are non-empty
    if not bullish_factors:
        bullish_factors.append({"point": f"Price holding above key support at {support}.", "source": "Technical Analysis"})
    if not bearish_factors:
        bearish_factors.append({"point": f"Overhead resistance capping gains near {resistance}.", "source": "Technical Analysis"})

    # 6. Dynamic Confidence Determination
    if conflict_mode == "aligned_bullish":
        confidence = f"high — technical levels, bullish PCR ({pcr}), and positive news align"
    elif conflict_mode == "aligned_bearish":
        confidence = f"high — bearish short buildup, low PCR ({pcr}), and headwind catalyst align"
    elif conflict_mode == "conflicting_mixed":
        confidence = f"moderate — PCR ({pcr}) is supportive but offset by short buildup and overhead resistance"
    else:
        confidence = f"low — sharp divergence between low PCR ({pcr}) and rising price action; low conviction"

    # 7. Formulate Input User Prompt matching Orchestrator Prompt Structure
    rag_text_lines = [f"- [{c['corpus_type'].upper()} | Source: {c['source']}] {c['content']}" for c in selected_rag]
    rag_formatted = "\n".join(rag_text_lines)

    user_input_prompt = f"""=== INPUT CONTEXT ===
SYMBOL: {symbol}
QUERY: Analyze risk-reward profile and technical levels for {symbol} options.

[1] LIVE MARKET SNAPSHOT:
- LTP: {ltp}
- Open: {round(ltp*0.998, 2)}, High: {round(ltp*1.01, 2)}, Low: {round(ltp*0.99, 2)}
- IV: {iv}%
- PCR: {pcr}
- OI Trend: {oi_trend}
- Key Support: {support}
- Key Resistance: {resistance}

[2] RAG KNOWLEDGE & NEWS CHUNKS:
{rag_formatted}

[3] HISTORICAL BACKTEST METRICS:
- Strategy Sharpe Ratio: {sharpe}
- Win Rate: {win_rate}%
- Max Drawdown: {max_dd}%
- Sample Size: {sample_size} trades
- Caveat: {backtest_caveat}"""

    # 8. Formulate Output matching Section 5 Schema
    output_schema = {
        "symbol": symbol,
        "as_of": "2026-08-19T12:00:00Z",
        "current_data": {
            "ltp": ltp,
            "iv": iv,
            "pcr": pcr,
            "oi_trend": oi_trend,
        },
        "bullish_factors": bullish_factors,
        "bearish_factors": bearish_factors,
        "backtest_context": {
            "sharpe": sharpe,
            "sample_size": sample_size,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "caveat": backtest_caveat,
        },
        "key_levels": {"support": support, "resistance": resistance},
        "invalidation_conditions": f"Break below support at {support} or PCR flip below {round(max(0.65, pcr - 0.25), 2)}.",
        "confidence": confidence,
        "explicit_note": "This is decision-support analysis, not a recommendation to buy or sell.",
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input_prompt},
            {"role": "assistant", "content": json.dumps(output_schema)},
        ]
    }


def generate_dataset(output_path: Path, num_examples: int = 500) -> Path:
    """Generate dataset with rich reasoning diversity, signal conflicts, dynamic confidence, and RAG context."""
    symbols = [
        ("RELIANCE", 2940.0),
        ("NIFTY", 24500.0),
        ("BANKNIFTY", 52000.0),
        ("TCS", 4200.0),
        ("INFY", 1850.0),
        ("HDFCBANK", 1650.0),
        ("ICICIBANK", 1220.0),
        ("SBIN", 840.0),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_examples):
            sym, base_p = random.choice(symbols)
            ex = generate_varied_synthetic_example(sym, base_p)
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {num_examples} diverse fine-tuning examples at {output_path}")
    return output_path


if __name__ == "__main__":
    out = Path(__file__).parent / "plutus_finetune_dataset.jsonl"
    generate_dataset(out, num_examples=250)
