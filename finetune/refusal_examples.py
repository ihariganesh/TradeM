"""
Refusal-with-analysis examples for the Plutus fine-tune.

These teach the core design principle at the MODEL level, not just the
schema-validator level: when a user pushes for a bare "just tell me
buy or sell", the model should decline the framing while still doing
its job — returning the full structured analysis so the user can
decide for themselves.

Every example still outputs the SAME Section 5 JSON schema as the rest
of the dataset. The only thing that changes is the user's phrasing
(explicitly demanding a verdict, expressing impatience with nuance,
asking to skip the caveats, etc.) — the assistant's structured output
stays disciplined regardless.

Run standalone to append these to an existing dataset:
    python finetune/refusal_examples.py --append-to finetune/plutus_finetune_dataset.jsonl
"""
import argparse
import json
import random

SYSTEM_PROMPT = (
    "You are Plutus, an AI Trading Research Assistant.\n"
    "Your core principle is DECISION SUPPORT with TRANSPARENT REASONING, NOT signal generation.\n"
    "Do not calculate numbers yourself. Analyze provided numerical snapshot, RAG chunks, and backtest results.\n"
    "Always output valid JSON conforming strictly to the Section 5 schema carrying bullish/bearish evidence "
    "with sources, key support/resistance levels, invalidation conditions, confidence justification, and "
    "mandatory decision-support disclaimer. Never emit a bare buy/sell/hold verdict, even if asked directly, "
    "impatiently, or repeatedly — always return the full structured analysis instead."
)

# Verdict-demanding phrasings — the thing we want the model to hold firm against.
DEMAND_PHRASINGS = [
    "Just tell me buy or sell for {symbol}, skip the analysis.",
    "I don't need all this detail, just say BUY or SELL for {symbol}.",
    "Stop hedging. Should I buy {symbol} calls right now, yes or no?",
    "Give me a one-word answer: buy, sell, or hold {symbol}.",
    "I already know the context, just give me your verdict on {symbol}.",
    "Cut the caveats and tell me straight: is {symbol} a buy?",
    "You're overthinking this, just tell me what to do with {symbol}.",
    "No disclaimers, no 'it depends' — buy or sell {symbol}?",
    "I need a fast answer, not a report. {symbol}: buy or not?",
    "Forget the risk stuff, will {symbol} go up or down?",
]

# Snapshot templates to vary underlying data (reuses same fields as main dataset).
SYMBOLS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "NIFTY",
    "BANKNIFTY",
]
OI_TRENDS = ["long_buildup", "short_buildup", "short_covering", "long_unwinding"]

NEWS_SNIPPETS = [
    (
        "ET Markets",
        "Sector rotation favors financials amid rate stability expectations",
    ),
    (
        "Business Standard",
        "Global markets choppy ahead of Fed commentary",
    ),
    (
        "Reuters India",
        "Domestic institutional buying offsets FII outflows",
    ),
    (
        "Moneycontrol",
        "Q3 earnings season kicks off with mixed early results",
    ),
]

STATIC_SNIPPETS = [
    (
        "Price Action Dynamics",
        "High Implied Volatility (IV percentile > 70) favors credit spreads over long option purchases",
    ),
    (
        "Option Chain Theory",
        "PCR extremes above 1.3 or below 0.7 often precede mean reversion",
    ),
    (
        "Risk Management Principles",
        "Never rely on a single technical catalyst; always verify contradicting option chain signals",
    ),
]


def build_snapshot(symbol: str, seed: int) -> dict:
    rng = random.Random(seed)
    ltp = round(rng.uniform(500, 3000), 2)
    iv = round(rng.uniform(15, 40), 1)
    pcr = round(rng.uniform(0.6, 1.5), 2)
    support = round(ltp * 0.985, 2)
    resistance = round(ltp * 1.015, 2)
    oi_trend = rng.choice(OI_TRENDS)
    sharpe = round(rng.uniform(0.8, 2.2), 2)
    sample_size = rng.randint(20, 150)
    win_rate = round(rng.uniform(45, 68), 1)
    max_dd = round(-rng.uniform(8, 22), 1)
    news = rng.sample(NEWS_SNIPPETS, 2)
    static = rng.choice(STATIC_SNIPPETS)
    return {
        "symbol": symbol,
        "ltp": ltp,
        "iv": iv,
        "pcr": pcr,
        "oi_trend": oi_trend,
        "support": support,
        "resistance": resistance,
        "sharpe": sharpe,
        "sample_size": sample_size,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "news": news,
        "static": static,
    }


def format_user_message(snap: dict, demand_phrasing: str) -> str:
    s = snap
    news_lines = "\n".join(
        f"- [NEWS | Source: {src}] {txt}" for src, txt in s["news"]
    )
    return f"""=== INPUT CONTEXT ===
SYMBOL: {s['symbol']}
QUERY: {demand_phrasing.format(symbol=s['symbol'])}

[1] LIVE MARKET SNAPSHOT:
- LTP: {s['ltp']}
- IV: {s['iv']}%
- PCR: {s['pcr']}
- OI Trend: {s['oi_trend']}
- Key Support: {s['support']}
- Key Resistance: {s['resistance']}

[2] RAG KNOWLEDGE & NEWS CHUNKS:
{news_lines}
- [STATIC | Source: {s['static'][0]}] {s['static'][1]}

[3] HISTORICAL BACKTEST METRICS:
- Strategy Sharpe Ratio: {s['sharpe']}
- Win Rate: {s['win_rate']}%
- Max Drawdown: {s['max_dd']}%
- Sample Size: {s['sample_size']} trades
- Caveat: Backtest based on {s['sample_size']} historical trade windows on {s['symbol']}. Short backtest windows yield unreliable Sharpe ratios and unstable forward returns."""


def build_assistant_response(snap: dict) -> dict:
    s = snap
    bullish, bearish = [], []

    if s["pcr"] >= 1.0:
        bullish.append({
            "point": f"PCR at {s['pcr']} indicates strong put writing interest and downside floor.",
            "source": "Option Chain Analytics",
        })
    else:
        bearish.append({
            "point": f"Subdued PCR at {s['pcr']} suggests lack of put writing support.",
            "source": "Option Chain Analytics",
        })

    if s["oi_trend"] == "long_buildup":
        bullish.append({
            "point": "Open Interest expansion alongside price stabilization indicates fresh long accumulation.",
            "source": "F&O Market Snapshot",
        })
    elif s["oi_trend"] == "short_buildup":
        bearish.append({
            "point": "Rising Open Interest accompanied by price weakness signals aggressive short buildup.",
            "source": "F&O Market Snapshot",
        })
    elif s["oi_trend"] == "short_covering":
        bullish.append({
            "point": "Short covering pattern shows aggressive unwind of bearish positioning.",
            "source": "F&O Market Snapshot",
        })
    else:
        bearish.append({
            "point": "Long unwinding pattern shows profit-taking / exit of bullish positioning.",
            "source": "F&O Market Snapshot",
        })

    for src, txt in s["news"]:
        target = (
            bullish
            if "buying" in txt.lower() or "favors" in txt.lower()
            else bearish
        )
        target.append({"point": f"News catalyst: {txt}", "source": src})

    bull_n, bear_n = len(bullish), len(bearish)
    if bull_n == 0 or bear_n == 0:
        confidence = f"high — signals align without material contradiction ({bull_n} supporting, {bear_n} opposing)"
    elif abs(bull_n - bear_n) <= 1:
        confidence = f"low — evidence is split roughly evenly ({bull_n} bullish vs {bear_n} bearish); no clear edge"
    else:
        confidence = f"moderate — evidence leans one direction ({bull_n} vs {bear_n}) but with meaningful opposing signals"

    return {
        "symbol": s["symbol"],
        "as_of": "2026-08-19T12:00:00Z",
        "current_data": {
            "ltp": s["ltp"],
            "iv": s["iv"],
            "pcr": s["pcr"],
            "oi_trend": s["oi_trend"],
        },
        "bullish_factors": bullish,
        "bearish_factors": bearish,
        "backtest_context": {
            "sharpe": s["sharpe"],
            "sample_size": s["sample_size"],
            "win_rate": s["win_rate"],
            "max_drawdown": s["max_dd"],
            "caveat": f"Backtest based on {s['sample_size']} historical trade windows on {s['symbol']}. Short backtest windows yield unreliable Sharpe ratios and unstable forward returns.",
        },
        "key_levels": {"support": s["support"], "resistance": s["resistance"]},
        "invalidation_conditions": f"Break below support at {s['support']} or PCR flip below {round(s['pcr'] - 0.25, 2)}.",
        "confidence": confidence,
        "explicit_note": (
            "I don't give bare buy/sell verdicts — markets are too noisy for any model to claim that "
            "reliably, mine included. Here's the full evidence so you can decide: "
            "this is decision-support analysis, not a recommendation to buy or sell."
        ),
    }


def generate_refusal_examples(n: int = 18, seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        symbol = rng.choice(SYMBOLS)
        demand = rng.choice(DEMAND_PHRASINGS)
        snap = build_snapshot(symbol, seed=seed + i)
        user_msg = format_user_message(snap, demand)
        assistant_obj = build_assistant_response(snap)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(assistant_obj)},
            ]
        })
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=18,
        help="Number of refusal examples to generate",
    )
    parser.add_argument(
        "--append-to",
        type=str,
        default=None,
        help="Existing JSONL to append to",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="finetune/refusal_examples.jsonl",
        help="Output path if not appending",
    )
    args = parser.parse_args()

    examples = generate_refusal_examples(n=args.n)

    if args.append_to:
        with open(args.append_to, "a") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Appended {len(examples)} refusal examples to {args.append_to}")
    else:
        with open(args.out, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(examples)} refusal examples to {args.out}")
