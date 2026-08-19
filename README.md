# Trading Research Assistant — TradeM

> **Base model:** Plutus (Ollama, fine-tuned Llama 3.1 8B) · **Broker:** Angel One SmartAPI / Kite · **Dev:** Google Colab + Local (i5-12450HX, RTX 3050 6GB)

---

## 📌 Core Design Principle

TradeM's job is **decision support with transparent reasoning**, not signal generation.
- No bare buy/sell verdicts.
- All numerical computations (LTP, IV, PCR, Greeks, Sharpe, Drawdown) are calculated in **Python**, never by the LLM.
- LLM narrates and reasons over pre-computed structured evidence.
- Every analysis carries bullish factors, bearish factors, key support/resistance levels, invalidation conditions, confidence justification, and mandatory backtest caveats.

---

## 🏗️ Project Architecture & Components

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
   │AngelOne / Kite │ │ VectorStore │ │ NSE Options │ │  + output    │
   │ → price, OI,   │ │ + embeddings│ │ Backtester  │ │  formatting) │
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
                          │   SCANNER & ALERTING  │
                          │  (Explicit screens +  │
                          │   Telegram bot)       │
                          └───────────────────────┘
```

### Module Overview

1. **Market Data Module (`app/market_data/`)**: Live prices, OHLC, volume, Option Chain (OI, IV, Greeks, PCR), support & resistance levels. Persistence via SQLite (`market_db`). Primary integration: **Angel One SmartAPI** (TOTP authentication), secondary: KiteConnect, fallback: Mock provider.
2. **RAG Layer (`app/rag/`)**: Vector database with SentenceTransformers (`bge-small-en-v1.5`) and deterministic MD5 fallback vectorizer:
   - **Static Corpus**: Books and strategy documents chunked (~500–800 tokens) and stored.
   - **Live News Corpus**: News articles tagged with timestamps and filtered with a **72-hour recency decay window**.
3. **Backtest Engine (`app/backtest/`)**: Reuses NSE Options Backtester interface, computing Sharpe, Win Rate, Max Drawdown, Sample Size, and carrying the mandatory caveat (*"Short backtest windows yield unreliable Sharpe ratios"*).
4. **Ollama / Plutus LLM (`app/llm/`)**: Interacts with local Ollama API (`http://localhost:11434/api/generate`) with offline reasoning fallback engine.
5. **Section 5 Output Schema (`app/schemas/analysis.py`)**: Enforced Pydantic validation for structured outputs.
6. **Scanner & Alerting Module (`app/scanner/`)**: Scheduled Python-driven quantitative screens (volume breakout, PCR extreme, IV spike) that push alerts via Telegram and log to SQLite.
7. **Fine-Tuning Pipeline (`finetune/`)**: Unsloth QLoRA fine-tuning scripts and synthetic dataset generator for Colab.

---

## 🚀 Quick Start

### 1. Installation & Environment Setup

```bash
# Clone repository
git clone https://github.com/ihariganesh/TradeM.git
cd TradeM

# Create & activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies (packages finding configured for app & finetune)
pip install -e .
```

### 2. Configure Angel One SmartAPI Credentials (.env)

Edit `.env` (derived from `.env.example`):
```env
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_CODE=your_client_code
ANGEL_PASSWORD=your_password
ANGEL_TOTP_KEY=your_totp_key
```

### 3. Running the FastAPI Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API Documentation available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Running Unit Tests

```bash
pytest -v
```

---

## 📡 API Endpoints Summary

- `POST /analyze`: Main decision-support analysis endpoint.
- `POST /rag/ingest/news`: Ingest live news article into vector store.
- `POST /rag/ingest/book`: Ingest static strategy book/document into vector store.
- `POST /scanner/trigger`: Execute quantitative screens against watchlist and trigger alerts.
- `GET /scanner/watchlist`: View active scanner watchlist.
