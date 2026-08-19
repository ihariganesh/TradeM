import json
import random
from pathlib import Path
from typing import List

SYSTEM_PROMPT = """You are Plutus, an AI Trading Research Assistant.
Your principle is decision-support with transparent reasoning, NOT signal generation.
Always output valid JSON conforming strictly to the Section 5 schema with bullish/bearish evidence, key levels, invalidation conditions, confidence justification, and mandatory decision-support disclaimer."""


def generate_synthetic_example(symbol: str, base_price: float) -> dict:
    variation = random.uniform(-0.02, 0.02)
    ltp = round(base_price * (1 + variation), 2)
    support = round(ltp * 0.985, 2)
    resistance = round(ltp * 1.015, 2)
    iv = round(random.uniform(18.0, 26.0), 1)
    pcr = round(random.uniform(0.7, 1.4), 2)
    oi_trend = random.choice(["long_buildup", "short_buildup", "short_covering"])

    input_text = f"Analyze market context for {symbol}: LTP={ltp}, IV={iv}%, PCR={pcr}, OI_Trend={oi_trend}, Support={support}, Resistance={resistance}."

    bullish_factors = []
    bearish_factors = []

    if pcr >= 1.0:
        bullish_factors.append({
            "point": f"Put-Call Ratio at {pcr} signals firm downside put writing support.",
            "source": "Option Chain Analytics",
        })
    else:
        bearish_factors.append({
            "point": f"Put-Call Ratio at {pcr} indicates limited put writing enthusiasm.",
            "source": "Option Chain Analytics",
        })

    if oi_trend == "long_buildup":
        bullish_factors.append({
            "point": f"Open Interest rising alongside price indicates fresh long accumulation.",
            "source": "F&O Analytics",
        })
    elif oi_trend == "short_buildup":
        bearish_factors.append({
            "point": f"Open Interest rising alongside falling price indicates active short buildup.",
            "source": "F&O Analytics",
        })

    if not bullish_factors:
        bullish_factors.append({
            "point": f"Technical support holding firm at {support}.",
            "source": "Technical Analysis",
        })
    if not bearish_factors:
        bearish_factors.append({
            "point": f"Overhead supply zone capping price near resistance {resistance}.",
            "source": "Technical Analysis",
        })

    output_data = {
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
            "sharpe": 1.45,
            "sample_size": 50,
            "win_rate": 58.0,
            "max_drawdown": -11.5,
            "caveat": "Short backtest windows yield unreliable Sharpe ratios.",
        },
        "key_levels": {"support": support, "resistance": resistance},
        "invalidation_conditions": f"Break below key support {support} or PCR flip below 0.80.",
        "confidence": "moderate — balanced technical levels supported by options OI data",
        "explicit_note": "This is decision-support analysis, not a recommendation to buy or sell.",
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": json.dumps(output_data)},
        ]
    }


def generate_dataset(
    output_path: Path, num_examples: int = 500
) -> Path:
    symbols = [
        ("RELIANCE", 2940.0),
        ("NIFTY", 24500.0),
        ("BANKNIFTY", 52000.0),
        ("TCS", 4200.0),
        ("INFY", 1850.0),
        ("HDFCBANK", 1650.0),
        ("ICICIBANK", 1220.0),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_examples):
            sym, base_p = random.choice(symbols)
            ex = generate_synthetic_example(sym, base_p)
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {num_examples} training examples at {output_path}")
    return output_path


if __name__ == "__main__":
    out = Path(__file__).parent / "plutus_finetune_dataset.jsonl"
    generate_dataset(out, num_examples=200)
