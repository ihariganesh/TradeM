from app.rag.rss_crawler import rss_crawler


def test_symbol_detection():
    text1 = "Reliance Industries Q3 earnings beat estimates on strong retail growth."
    symbols1 = rss_crawler._detect_symbols(text1)
    assert "RELIANCE" in symbols1

    text2 = "Nifty 50 and Bank Nifty hit fresh highs amid RBI policy decision."
    symbols2 = rss_crawler._detect_symbols(text2)
    assert "NIFTY" in symbols2
    assert "BANKNIFTY" in symbols2


def test_rss_crawler_ingest():
    res = rss_crawler.crawl_and_ingest_all()
    assert res["status"] == "success"
    assert "total_ingested" in res
    assert isinstance(res["articles"], list)
