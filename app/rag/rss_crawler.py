import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from app.rag.ingestion import ingest_news_article

logger = logging.getLogger(__name__)

# Primary Financial RSS Feed Endpoints (India Markets)
DEFAULT_RSS_FEEDS = [
    {
        "name": "ET Markets - Stock News",
        "url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "source": "ET Markets",
    },
    {
        "name": "Moneycontrol - Markets",
        "url": "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "source": "Moneycontrol",
    },
    {
        "name": "Livemint - Markets",
        "url": "https://www.livemint.com/rss/markets",
        "source": "Livemint",
    },
]

SYMBOL_KEYWORDS = {
    "RELIANCE": ["reliance", "ril", "jio", "ambani", "o2c"],
    "NIFTY": ["nifty", "nifty50", "nse", "dalal street", "sensex", "market rally"],
    "BANKNIFTY": ["banknifty", "bank nifty", "rbi", "psu bank", "banking index"],
    "TCS": ["tcs", "tata consultancy", "it sector", "tata group"],
    "INFY": ["infosys", "infy", "salil parekh", "it spending"],
    "HDFCBANK": ["hdfc bank", "hdfc", "net interest margin"],
    "ICICIBANK": ["icici bank", "icici", "sandeep bakhshi"],
    "SBIN": ["sbin", "state bank", "sbi", "dinesh khara"],
}


class RSSNewsCrawler:
    """Automated RSS Crawler fetching live financial market news into TradeM 72h RAG vector store."""

    def __init__(self, feeds: Optional[List[Dict[str, str]]] = None):
        self.feeds = feeds or DEFAULT_RSS_FEEDS
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def crawl_feed(self, feed_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch and parse single RSS feed using httpx and feedparser fallback."""
        url = feed_info["url"]
        source_name = feed_info["source"]
        articles = []

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch RSS feed {url}: HTTP {response.status_code}")
                    return articles

                content = response.text

            try:
                import feedparser
                parsed = feedparser.parse(content)
                entries = parsed.entries
            except Exception:
                # Basic regex fallback if feedparser unavailable
                entries = []
                items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
                for item in items:
                    title_match = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
                    desc_match = re.search(r'<description>(.*?)</description>', item, re.DOTALL)
                    link_match = re.search(r'<link>(.*?)</link>', item, re.DOTALL)
                    if title_match:
                        entries.append({
                            'title': title_match.group(1).strip(),
                            'summary': desc_match.group(1).strip() if desc_match else "",
                            'link': link_match.group(1).strip() if link_match else "",
                        })

            for entry in entries:
                headline = getattr(entry, "title", "") or entry.get("title", "")
                summary = getattr(entry, "summary", "") or entry.get("summary", "") or headline
                headline = re.sub(r'<[^>]+>', '', headline).strip()
                summary = re.sub(r'<[^>]+>', '', summary).strip()

                if not headline:
                    continue

                # Match symbols in headline and summary
                matched_symbols = self._detect_symbols(f"{headline} {summary}")
                if not matched_symbols:
                    matched_symbols = ["NIFTY"]  # Default market index tagging

                for sym in matched_symbols:
                    articles.append({
                        "headline": headline,
                        "body": summary[:400],
                        "symbol": sym,
                        "source": source_name,
                        "url": getattr(entry, "link", "") or entry.get("link", ""),
                    })

        except Exception as e:
            logger.warning(f"Error crawling RSS feed {url}: {e}")

        return articles

    def _detect_symbols(self, text: str) -> List[str]:
        """Detect referenced stock/index symbols from headline and body text."""
        text_lower = text.lower()
        matched = []
        for symbol, keywords in SYMBOL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                matched.append(symbol)
        return matched

    def crawl_and_ingest_all(self) -> Dict[str, Any]:
        """Crawl all registered feeds and ingest news items into RAG vector store."""
        total_ingested = 0
        ingested_details = []

        for feed in self.feeds:
            articles = self.crawl_feed(feed)
            for art in articles:
                try:
                    ingest_news_article(
                        headline=art["headline"],
                        body=art["body"],
                        symbol=art["symbol"],
                        source=art["source"],
                    )
                    total_ingested += 1
                    ingested_details.append(art)
                except Exception as e:
                    logger.warning(f"Failed to ingest crawler article '{art['headline']}': {e}")

        logger.info(f"RSS News Crawler finished. Ingested {total_ingested} live news items into RAG store.")
        return {
            "status": "success",
            "total_ingested": total_ingested,
            "articles": ingested_details[:10],  # Return preview of top 10
        }

    def start_background_crawler(self, interval_seconds: int = 300) -> None:
        """Start non-blocking daemon thread crawling feeds periodically."""
        if self._running:
            return

        self._running = True

        def _loop():
            logger.info(f"RSS Crawler daemon started (interval={interval_seconds}s).")
            while self._running:
                try:
                    self.crawl_and_ingest_all()
                except Exception as e:
                    logger.error(f"Error in RSS Crawler daemon loop: {e}")
                time.sleep(interval_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_background_crawler(self) -> None:
        """Stop background crawler thread."""
        self._running = False


rss_crawler = RSSNewsCrawler()
