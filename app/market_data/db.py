import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from app.config import settings


class MarketDataDB:

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ltp REAL NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    iv REAL,
                    pcr REAL,
                    oi_trend TEXT,
                    raw_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_ts 
                ON market_snapshots(symbol, timestamp);
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    screen_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    full_writeup TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_snapshot(self, symbol: str, data: Dict[str, Any]) -> None:
        timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())
        ltp = float(data.get("ltp", 0.0))
        open_price = data.get("open")
        high_price = data.get("high")
        low_price = data.get("low")
        close_price = data.get("close")
        volume = data.get("volume")
        iv = data.get("iv")
        pcr = data.get("pcr")
        oi_trend = data.get("oi_trend", "neutral")
        raw_data = json.dumps(data)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO market_snapshots 
                (symbol, timestamp, ltp, open, high, low, close, volume, iv, pcr, oi_trend, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    timestamp,
                    ltp,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    iv,
                    pcr,
                    oi_trend,
                    raw_data,
                ),
            )
            conn.commit()

    def get_latest_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT raw_data FROM market_snapshots
                WHERE symbol = ?
                ORDER BY id DESC LIMIT 1
            """,
                (symbol,),
            )
            row = cursor.fetchone()
            if row and row["raw_data"]:
                return json.loads(row["raw_data"])
        return None

    def log_alert(
        self, symbol: str, screen_name: str, summary: str, full_writeup: str
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts_log (symbol, screen_name, timestamp, summary, full_writeup)
                VALUES (?, ?, ?, ?, ?)
            """,
                (symbol, screen_name, ts, summary, full_writeup),
            )
            conn.commit()


market_db = MarketDataDB()
