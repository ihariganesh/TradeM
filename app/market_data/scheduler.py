import logging
from typing import List
from app.config import settings
from app.market_data.provider import market_provider

logger = logging.getLogger(__name__)


def refresh_watchlist_snapshots(symbols: List[str] | None = None) -> List[dict]:
    """Poll live market data for symbols and update local DB cache."""
    symbols = symbols or settings.WATCHLIST
    snapshots = []
    for symbol in symbols:
        try:
            snap = market_provider.get_snapshot(symbol)
            snapshots.append(snap)
            logger.info(f"Updated market snapshot for {symbol}: LTP={snap.get('ltp')}")
        except Exception as e:
            logger.error(f"Failed updating snapshot for {symbol}: {e}")
    return snapshots
