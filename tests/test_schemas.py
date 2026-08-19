from app.schemas.analysis import (
    BacktestContextSchema,
    CurrentDataSchema,
    FactorSchema,
    KeyLevelsSchema,
    SymbolAnalysisResponse,
)


def test_section_5_schema_valid():
    data = {
        "symbol": "RELIANCE",
        "as_of": "2026-08-19T12:00:00Z",
        "current_data": {
            "ltp": 2940.0,
            "iv": 22.5,
            "pcr": 1.15,
            "oi_trend": "long_buildup",
        },
        "bullish_factors": [
            {
                "point": "PCR 1.15 reflects put writing support",
                "source": "Option Chain",
            }
        ],
        "bearish_factors": [
            {
                "point": "Overhead resistance at 2980",
                "source": "Technical Analysis",
            }
        ],
        "backtest_context": {
            "sharpe": 1.45,
            "sample_size": 48,
            "win_rate": 58.5,
            "max_drawdown": -12.4,
            "caveat": "Short backtest windows yield unreliable Sharpe ratios.",
        },
        "key_levels": {"support": 2900.0, "resistance": 2980.0},
        "invalidation_conditions": "Break below 2900 support or PCR flipping below 0.85.",
        "confidence": "moderate — balanced technical levels",
        "explicit_note": "This is decision-support analysis, not a recommendation to buy or sell.",
    }

    res = SymbolAnalysisResponse(**data)
    assert res.symbol == "RELIANCE"
    assert res.current_data.ltp == 2940.0
    assert (
        "decision-support analysis" in res.explicit_note
    )
    assert len(res.bullish_factors) == 1
    assert len(res.bearish_factors) == 1
