from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_analyze_endpoint():
    payload = {
        "query": "Should I look at Reliance for this week?",
        "symbol": "RELIANCE",
        "instrument_type": "options",
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["symbol"] == "RELIANCE"
    assert "current_data" in data
    assert "bullish_factors" in data
    assert "bearish_factors" in data
    assert "backtest_context" in data
    assert "key_levels" in data
    assert "invalidation_conditions" in data
    assert "confidence" in data
    assert "explicit_note" in data
    assert (
        "This is decision-support analysis, not a recommendation to buy or sell."
        in data["explicit_note"]
    )
