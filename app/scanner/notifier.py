import logging
from typing import Any, Dict
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:

    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    def send_alert(
        self,
        symbol: str,
        screen_name: str,
        reason: str,
        analysis_response: Dict[str, Any],
    ) -> bool:
        """Send formatted alert message with Section 5 evidence and caveats."""

        ltp = analysis_response.get("current_data", {}).get("ltp")
        confidence = analysis_response.get("confidence", "moderate")
        support = analysis_response.get("key_levels", {}).get("support")
        resistance = analysis_response.get("key_levels", {}).get("resistance")
        invalidation = analysis_response.get("invalidation_conditions", "N/A")
        explicit_note = analysis_response.get(
            "explicit_note",
            "This is decision-support analysis, not a recommendation to buy or sell.",
        )

        message = (
            f"🚨 <b>TRADEM SCANNER ALERT: {symbol}</b> 🚨\n\n"
            f"<b>Matched Screen:</b> {screen_name}\n"
            f"<b>Trigger Reason:</b> {reason}\n\n"
            f"<b>LTP:</b> ₹{ltp}\n"
            f"<b>Support:</b> ₹{support} | <b>Resistance:</b> ₹{resistance}\n"
            f"<b>Confidence:</b> {confidence}\n"
            f"<b>Invalidation:</b> {invalidation}\n\n"
            f"⚠️ <i>{explicit_note}</i>"
        )

        # Log alert to system output
        logger.info(
            f"[ALERT SENT] Symbol={symbol} Screen='{screen_name}' Reason='{reason}'"
        )

        if not self.bot_token or not self.chat_id:
            logger.info(
                "Telegram BOT_TOKEN or CHAT_ID not set. Alert printed to logs."
            )
            return True

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, json=payload)
                return res.status_code == 200
        except Exception as e:
            logger.error(f"Failed sending Telegram notification: {e}")
            return False


notifier = TelegramNotifier()
