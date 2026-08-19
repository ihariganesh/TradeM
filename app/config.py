from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System configuration for Trading Research Assistant (TradeM)."""

    # Service configuration
    APP_NAME: str = "TradeM - Trading Research Assistant"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Data directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "tradem.db"

    # Ollama / Plutus LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "plutus:latest"
    LLM_TIMEOUT: float = 60.0

    # Broker API Settings (Angel One SmartAPI & Kite fallback)
    ANGEL_API_KEY: str = ""
    ANGEL_CLIENT_CODE: str = ""
    ANGEL_PASSWORD: str = ""
    ANGEL_TOTP_KEY: str = ""

    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    KITE_ACCESS_TOKEN: str = ""

    # RAG Settings
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    NEWS_RECENCY_HOURS: float = 72.0

    # Telegram Alerting Settings
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Scanner Settings
    SCANNER_INTERVAL_MINUTES: int = 15
    WATCHLIST: list[str] = [
        "RELIANCE",
        "NIFTY",
        "BANKNIFTY",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
    ]

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
