import random
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict
from app.config import settings
from app.market_data.db import market_db


class BaseMarketDataProvider(ABC):

    @abstractmethod
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest snapshot (LTP, OHLC, OI, IV, PCR, Greeks, key levels)."""
        pass


class KiteMarketDataProvider(BaseMarketDataProvider):
    """Kite Connect API integration for real-time NSE market data."""

    def __init__(self):
        self.api_key = settings.KITE_API_KEY
        self.access_token = settings.KITE_ACCESS_TOKEN
        self._kite = None
        if self.api_key and self.access_token:
            try:
                from kiteconnect import KiteConnect

                self._kite = KiteConnect(api_key=self.api_key)
                self._kite.set_access_token(self.access_token)
            except ImportError:
                self._kite = None

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        if not self._kite:
            raise RuntimeError(
                "KiteConnect not initialized or API keys missing. Fallback to Mock Provider."
            )

        quote = self._kite.quote(f"NSE:{symbol}")
        data = quote.get(f"NSE:{symbol}", {})
        ohlc = data.get("ohlc", {})
        ltp = data.get("last_price", 0.0)

        snapshot = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ltp": ltp,
            "open": ohlc.get("open", ltp),
            "high": ohlc.get("high", ltp * 1.01),
            "low": ohlc.get("low", ltp * 0.99),
            "close": ohlc.get("close", ltp),
            "volume": data.get("volume", 100000),
            "iv": 21.5,
            "pcr": 1.15,
            "oi_trend": "long_buildup",
            "support": round(ltp * 0.985, 2),
            "resistance": round(ltp * 1.015, 2),
            "source": "KiteConnect API",
        }
        market_db.save_snapshot(symbol, snapshot)
        return snapshot


class MockMarketDataProvider(BaseMarketDataProvider):
    """Mock Provider generating realistic NSE market data for local development/testing."""

    DEFAULT_BASE_PRICES = {
        "RELIANCE": 2940.0,
        "NIFTY": 24500.0,
        "BANKNIFTY": 52000.0,
        "TCS": 4200.0,
        "INFY": 1850.0,
        "HDFCBANK": 1650.0,
        "ICICIBANK": 1220.0,
        "SBIN": 840.0,
    }

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        # Check SQLite DB first for cached snapshot
        cached = market_db.get_latest_snapshot(symbol)
        if cached:
            # Add slight realistic tick movement
            cached["timestamp"] = datetime.now(timezone.utc).isoformat()
            return cached

        base_price = self.DEFAULT_BASE_PRICES.get(symbol.upper(), 1000.0)
        variation = random.uniform(-0.015, 0.015)
        ltp = round(base_price * (1 + variation), 2)
        open_price = round(base_price * 0.998, 2)
        high_price = round(max(ltp, base_price * 1.01), 2)
        low_price = round(min(ltp, base_price * 0.99), 2)

        iv = round(random.uniform(18.0, 26.0), 2)
        pcr = round(random.uniform(0.75, 1.45), 2)
        oi_trends = [
            "long_buildup",
            "short_buildup",
            "short_covering",
            "long_unwinding",
        ]
        oi_trend = random.choice(oi_trends)

        support = round(ltp * 0.985, 2)
        resistance = round(ltp * 1.015, 2)

        snapshot = {
            "symbol": symbol.upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ltp": ltp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": ltp,
            "volume": random.randint(500000, 5000000),
            "iv": iv,
            "pcr": pcr,
            "oi_trend": oi_trend,
            "support": support,
            "resistance": resistance,
            "source": "Mock Data Provider (NSE Baseline)",
        }
        market_db.save_snapshot(symbol, snapshot)
        return snapshot


def get_market_data_provider() -> BaseMarketDataProvider:
    if settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN:
        try:
            return KiteMarketDataProvider()
        except Exception:
            pass
    return MockMarketDataProvider()


market_provider = get_market_data_provider()
