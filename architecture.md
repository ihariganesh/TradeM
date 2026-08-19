# Trading Research Assistant — System Architecture
**Base model:** Plutus (Ollama, fine-tuned Llama 3.1 8B) · **Broker:** Kite/Upstox · **Dev:** Google Colab + local (i5-12450HX, RTX 3050 6GB)

---

## 0. Design principle (read this first)

The system's job is **decision support with transparent reasoning**, not signal generation. Every output must show its evidence and its uncertainty. This isn't a compliance disclaimer bolted on top — it's baked into the output schema (Section 5) because:

- No model — fine-tuned or not — has a durable edge at predicting short-term price direction from news/technicals. If it claims one, that's a bug.
- What *is* achievable: aggregating scattered information (news, IV, OI, technicals, historical backtests, book-derived theory) into one coherent, sourced view faster than you could manually — so you decide with better inputs, in less time.
- Every recommendation-shaped output must carry its reasoning chain, contradicting evidence, and invalidation conditions alongside it. No bare verdicts.

Keep this constraint in the fine-tuning dataset and the prompt templates — it's a design decision, not an add-on.

---

## 1. High-level architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                         │
│                    (Python service, FastAPI)                        │
└───────────┬───────────────┬───────────────┬─────────────┬──────────┘
            │               │               │             │
   ┌────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐ ┌────▼─────────┐
   │  LIVE MARKET   │ │  RAG LAYER  │ │  BACKTEST   │ │  FINE-TUNED  │
   │  DATA MODULE   │ │ (news/books)│ │   ENGINE    │ │    PLUTUS    │
   │                │ │             │ │             │ │  (reasoning  │
   │ Kite/Upstox API│ │ Chroma/FAISS│ │ your NSE    │ │  + output    │
   │ → price, OI,   │ │ + embeddings│ │ backtester  │ │  formatting) │
   │ IV, PCR, Greeks│ │             │ │             │ │              │
   └────────────────┘ └─────────────┘ └─────────────┘ └──────────────┘
            │               │               │             │
            └───────────────┴───────┬───────┴─────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   STRUCTURED OUTPUT   │
                          │  (Section 5 schema)   │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  SCANNER + ALERTING   │
                          │  (Section 6, runs on  │
                          │   its own schedule,   │
                          │   not request-driven) │
                          └───────────────────────┘
```

Four independent modules feed one orchestrator. This matters: **do not let the LLM compute anything numerical** (Greeks, Sharpe, PCR, backtest returns). Python computes; the LLM narrates and reasons over the computed output. LLMs are unreliable at arithmetic and you already have a working Python backtester — use it as the source of truth.

---

## 2. Module breakdown

### 2.1 Live Market Data Module
**Purpose:** structured, numerical, real-time — never goes through embeddings, injected directly into prompt context.

| Data | Source | Refresh |
|---|---|---|
| LTP, OHLC, volume | Kite Connect / Upstox API | tick / 1-min |
| Option chain (OI, IV, Greeks) | Kite Connect `instruments` + `quote` | 1-min |
| PCR | computed from option chain OI | 1-min |
| Historical candles (backtest input) | Kite `historical_data` | on-demand |
| Index data (Nifty/Sensex/sector) | same API | 1-min |

Build as a simple Python module (`market_data.py`) with a scheduler (APScheduler or a cron-triggered script) that writes latest snapshots to a local SQLite/Postgres table or even a JSON cache — no need for a vector DB here, this is structured data, not prose.

### 2.2 RAG Layer
**Purpose:** unstructured knowledge — news, geopolitical events, book/theory content.

- **Embedding model:** `bge-small-en-v1.5` or `nomic-embed-text` — both run comfortably on your 3050 or even CPU.
- **Vector DB:** ChromaDB (simplest, file-based, no server) for dev; consider Qdrant if you outgrow it.
- **Ingestion pipelines (two speeds):**
  - *Static corpus* (books, strategy docs, your own notes): chunk once (~500–800 tokens, overlap ~100), embed, store. Re-run only when you add new material.
  - *Live corpus* (news, geopolitical): scheduled job every 15–30 min — pull from NewsAPI / RSS (Moneycontrol, ET Markets, Reuters India, Business Standard) / GDELT for geopolitical, chunk, embed, upsert. Tag each chunk with a timestamp and **decay old news out of retrieval** (e.g., don't retrieve >72hr old news for a "what's happening now" query unless explicitly asked for background).

### 2.3 Backtest Engine
This is your **existing NSE Options Backtester** — reuse it, don't rebuild. The orchestrator calls it as a library/subprocess, gets back structured results (Sharpe, win rate, drawdown, sample size), and passes those numbers into the prompt as *evidence*, clearly labeled with the caveat you already noted: short backtest windows → unreliable Sharpe. Bake that caveat into the output template too (Section 5) so it's never dropped silently.

### 2.4 Fine-tuned Plutus
**What fine-tuning should and shouldn't do here:**

| Fine-tune for | Don't fine-tune for |
|---|---|
| Consistent structured output format (Section 5 schema) | Injecting facts/news (RAG's job) |
| Domain vocabulary (Greeks, PCR, IV, basis, roll) | Numerical computation (Python's job) |
| Reasoning style extracted from your books (how a good analyst weighs conflicting signals) | "Correct" buy/sell answers (none exist to train on — see below) |
| Refusing to give bare verdicts without evidence | — |

**Critical point on training data:** you cannot create a labeled dataset of "correct" buy/sell decisions — market outcomes are noisy, and training on historical "stock went up after news X" pairs teaches the model spurious correlations, not skill (this is a classic overfitting trap in retail algo-trading projects). Instead, build your fine-tuning dataset as **(context → structured analysis) pairs**, not (context → outcome) pairs. Example training example shape:

```json
{
  "input": "RELIANCE: LTP 2940, IV 22%, PCR 1.3, [news snippets], [technical levels]",
  "output": {
    "bullish_factors": ["...with source"],
    "bearish_factors": ["...with source"],
    "key_levels": {"support": 2900, "resistance": 2980},
    "invalidation": "if PCR flips below 0.8 or breaks 2900 support",
    "confidence_note": "moderate — mixed signals, no strong catalyst"
  }
}
```

You can generate a lot of this synthetically: take historical snapshots (data you already have from your backtester), and have a strong model (e.g., Claude or GPT-4 class) generate structured "at-the-time" analyses (not "then it went up" — the model shouldn't see the future outcome). This teaches *analysis structure and reasoning*, not prediction.

---

## 3. Fine-tuning pipeline (Google Colab)

### 3.1 Stack
- **Unsloth** — not raw HF PEFT. It's 2x faster and dramatically more memory-efficient for QLoRA; makes 8B fit comfortably in Colab's free T4 (16GB VRAM). This is the single biggest practical lever for your setup.
- **Base:** pull Plutus's underlying weights. Since Plutus is Llama 3.1 8B fine-tuned, you'll fine-tune on top of it — check if the Plutus authors published it on Hugging Face (if it's Ollama-only/GGUF, you may need to fine-tune base Llama 3.1 8B instead and re-apply Plutus's tuning approach, or fine-tune Plutus's GGUF-adjacent safetensors if available — I'd verify this before building the dataset).
- **Method:** QLoRA (4-bit), rank 16–32, target `q_proj,k_proj,v_proj,o_proj`.

### 3.2 Colab notebook structure
1. Install: `unsloth`, `transformers`, `trl`, `peft`, `bitsandbytes`
2. Load base model in 4-bit via Unsloth's `FastLanguageModel`
3. Load your dataset (JSONL, the shape above) from Google Drive (mount it — keeps data persistent across Colab sessions)
4. Format into chat template matching Plutus's expected prompt format
5. Train with `SFTTrainer` (from `trl`) — 2–3 epochs is usually enough for a few thousand examples; watch for overfitting given a small dataset
6. Save LoRA adapter to Drive
7. Merge adapter into base weights (`model.merge_and_unload()`)
8. Convert merged model to GGUF (`llama.cpp` conversion script) for Ollama import
9. `ollama create` with a Modelfile pointing at the new GGUF

### 3.3 Local role (your laptop)
- Dataset prep (chunking books, formatting JSONL) — CPU work, no GPU needed
- Running the final GGUF model via Ollama for inference — your 3050 handles Q4/Q5 8B fine
- RAG embedding + retrieval — fine locally
- Do **not** attempt the training step locally; 6GB VRAM is too tight for reliable QLoRA runs even with Unsloth

---

## 4. Orchestration layer

FastAPI service tying it together:

```
POST /analyze
{
  "query": "Should I look at Reliance for this week?",
  "symbol": "RELIANCE",
  "instrument_type": "options" | "equity"
}
   ↓
1. market_data.get_snapshot(symbol)
2. rag.retrieve(query, symbol, k=8, recency_filter=72h for news, none for books)
3. backtest.run_if_applicable(symbol, strategy_context)
4. prompt = build_prompt(snapshot, rag_chunks, backtest_results)
5. response = plutus.generate(prompt)  # via Ollama local API
6. validate_output_schema(response)  # enforce Section 5 structure, reject bare verdicts
7. return structured response
```

Ollama exposes a local REST API (`localhost:11434/api/generate`) — your FastAPI service calls that directly, no need for the Anthropic API for the core loop (only use an external strong model during dataset generation in Section 2.4).

---

## 5. Output schema (enforced, not optional)

Every response — equity or options query — follows this shape:

```json
{
  "symbol": "...",
  "as_of": "timestamp",
  "current_data": { "ltp": ..., "iv": ..., "pcr": ..., "oi_trend": "..." },
  "bullish_factors": [{"point": "...", "source": "..."}],
  "bearish_factors": [{"point": "...", "source": "..."}],
  "backtest_context": { "sharpe": ..., "sample_size": ..., "caveat": "..." },
  "key_levels": { "support": ..., "resistance": ... },
  "invalidation_conditions": "...",
  "confidence": "low | moderate | high — with justification",
  "explicit_note": "This is decision-support analysis, not a recommendation to buy or sell."
}
```

Enforce this at the orchestrator level (reject/reformat any model output that skips fields) — don't rely on the model to remember every time.

---

## 6. Scanner + Alerting module

This is a **different execution model** from everything above. Sections 1–5 are request/response (you ask, it answers). This module runs unattended on a schedule, scans a watchlist or the whole NSE universe against defined criteria, and pushes a notification when something matches — you don't ask, it tells you.

### 6.1 Why this needs care in the design

An alert that fires is an implicit "this looks interesting" claim, repeated automatically, at scale, without you in the loop to sanity-check it in the moment. Two things follow:

- **Criteria must be explicit and inspectable**, not "the LLM decided this is good." The LLM's job here is narration/context-gathering within already-defined rules, not inventing the definition of "good stock" on the fly. You define the screen; the system finds matches and explains them.
- **Every alert carries the same evidence-and-caveats structure as Section 5** — no bare "BUY RELIANCE" push. Alert fatigue from frequent overconfident-sounding pings is a real failure mode; better to alert less often with more substance than constantly with none.

### 6.2 Architecture

```
Scheduled job (every N min during market hours)
   ↓
1. Pull full/watchlist universe snapshot (market_data module)
2. Apply quantitative screen (Python, not LLM):
   e.g. volume spike > 2x avg, breakout above N-day high,
   IV crush, unusual OI buildup, RSI/MACD crossover,
   PCR shift, whatever criteria you define
   ↓
3. Candidates that pass the screen → RAG retrieval for context
   (news on that stock/sector in last few hours)
   ↓
4. Plutus generates the Section-5-style structured writeup
   PER candidate — bullish/bearish factors, levels,
   invalidation, confidence — not a bare signal
   ↓
5. Push notification (Telegram bot is the easiest path — free,
   simple API, works well for personal alerting; email/desktop
   notify as alternatives)
   ↓
6. Log every alert (symbol, criteria matched, full writeup,
   timestamp) to a table — this becomes your own track record
   you can later measure against outcomes, which is the closest
   thing to validating whether the system is actually useful
```

### 6.3 Screening criteria — define these explicitly, don't outsource to the LLM

Start with a small, well-understood set rather than a vague "find good stocks":

| Category | Example screens |
|---|---|
| Technical | breakout above 20/50-day high, volume > 2x 20-day avg, RSI crossing 30/70, MA crossovers |
| Options-specific | unusual OI buildup, IV spike/crush, PCR extreme + reversal, large block deals in F&O |
| News-driven | high-relevance news volume spike on a symbol (from your RAG ingestion), sentiment shift |
| Fundamental (slower-moving, daily not intraday) | earnings beat/miss vs. estimates, results-day screens |

Each screen is a plain Python function returning true/false or a score — testable and debuggable on its own, independent of the LLM. This also makes your backtester directly useful here: you can backtest a screen's historical hit rate *before* wiring it into live alerting, which is a more honest way to build confidence in a criterion than trusting an LLM's judgment on it. Same sample-size caveat as your existing backtester applies.

### 6.4 Notification delivery

Telegram is the practical choice for a solo project — free bot API, push notifications on your phone, trivial to integrate (`python-telegram-bot`), and you can reply to the bot to trigger a full Section-5 deep-dive on an alerted symbol. Desktop notification (local script) as a fallback for when you're at your laptop.

---

## 7. Realistic build order

1. **Market data module** — get Kite API pulling live snapshots, store structured (2–3 days)
2. **RAG layer** — static corpus first (your books), then live news ingestion (3–5 days)
3. **Wire orchestrator + base Plutus (no fine-tune yet)** — validate the whole pipeline works end-to-end with the *unmodified* model and a strong prompt template (1 week) — you'll learn a lot about what fine-tuning actually needs to fix
4. **Backtest engine integration** — plug in your existing project (2–3 days)
5. **Scanner + alerting** — start with 2–3 well-defined screens, backtest their historical hit rate, then wire to Telegram (1 week)
6. **Fine-tuning dataset generation + Colab training** — only after steps 3–5 show you concretely where the base model's outputs fall short of your schema (1–2 weeks)

Step 3 before fine-tuning is still the key sequencing call — prompt engineering + RAG will get you most of the way, and fine-tuning mainly locks in formatting/style consistency. Scanner/alerting (step 5) is a natural extension of the same request/response pipeline plus a scheduler, so it slots in before fine-tuning — you'll want the screens running and tuned against real output before baking a style/format into model weights.
