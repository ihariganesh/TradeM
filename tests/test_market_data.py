from app.market_data.provider import AngelOneMarketDataProvider, MockMarketDataProvider, get_market_data_provider


def test_mock_market_data_provider():
    provider = MockMarketDataProvider()
    snapshot = provider.get_snapshot("RELIANCE")

    assert snapshot["symbol"] == "RELIANCE"
    assert "ltp" in snapshot
    assert "iv" in snapshot
    assert "pcr" in snapshot
    assert "support" in snapshot
    assert "resistance" in snapshot
    assert snapshot["ltp"] > 0


def test_get_market_data_provider_fallback():
    provider = get_market_data_provider()
    assert provider is not None
    snapshot = provider.get_snapshot("NIFTY")
    assert snapshot["symbol"] == "NIFTY"
