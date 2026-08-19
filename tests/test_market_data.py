from app.market_data.provider import MockMarketDataProvider


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
