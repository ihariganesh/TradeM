import logging
from typing import Any, Dict, List
from app.config import settings
from app.market_data.db import market_db
from app.market_data.provider import market_provider
from app.scanner.notifier import notifier
from app.scanner.screens import run_all_screens

logger = logging.getLogger(__name__)


class ScannerService:
    """Unattended scanner service running quantitative screens and dispatching structured alerts."""

    def run_scan(
        self, watchlist: List[str] | None = None, orchestrator_service: Any = None
    ) -> List[Dict[str, Any]]:
        watchlist = watchlist or settings.WATCHLIST
        alerts_triggered = []

        for symbol in watchlist:
            try:
                # 1. Pull market snapshot
                snapshot = market_provider.get_snapshot(symbol)

                # 2. Run explicit Python quantitative screens
                passed_screens = run_all_screens(snapshot)

                # 3. For candidates passing any screen -> trigger analysis & alert
                for screen_res in passed_screens:
                    writeup = None
                    if orchestrator_service:
                        analysis_res = orchestrator_service.analyze_symbol(
                            query=f"Automated scanner alert context for screen '{screen_res.screen_name}' on {symbol}",
                            symbol=symbol,
                        )
                        writeup = analysis_res.model_dump()
                    else:
                        writeup = {
                            "symbol": symbol,
                            "current_data": {"ltp": snapshot.get("ltp")},
                            "confidence": "moderate",
                            "key_levels": {
                                "support": snapshot.get("support"),
                                "resistance": snapshot.get("resistance"),
                            },
                            "invalidation_conditions": f"Break below support {snapshot.get('support')}",
                            "explicit_note": "This is decision-support analysis, not a recommendation to buy or sell.",
                        }

                    # 4. Push notification
                    notifier.send_alert(
                        symbol=symbol,
                        screen_name=screen_res.screen_name,
                        reason=screen_res.reason,
                        analysis_response=writeup,
                    )

                    # 5. Log alert to SQLite
                    market_db.log_alert(
                        symbol=symbol,
                        screen_name=screen_res.screen_name,
                        summary=screen_res.reason,
                        full_writeup=str(writeup),
                    )

                    alerts_triggered.append({
                        "symbol": symbol,
                        "screen": screen_res.screen_name,
                        "reason": screen_res.reason,
                        "analysis": writeup,
                    })

            except Exception as e:
                logger.error(f"Error scanning symbol {symbol}: {e}")

        return alerts_triggered


scanner_service = ScannerService()
