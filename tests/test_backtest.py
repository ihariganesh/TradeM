from app.backtest.engine import backtest_engine, MANDATORY_SHARPE_CAVEAT


def test_options_backtest_engine():
    res = backtest_engine.run_backtest(
        symbol="RELIANCE",
        strategy="long_straddle",
        days_history=60,
    )
    assert res["symbol"] == "RELIANCE"
    assert res["strategy"] == "long_straddle"
    assert "sharpe" in res
    assert "win_rate" in res
    assert "max_drawdown" in res
    assert "profit_factor" in res
    assert len(res["equity_curve"]) > 0
    assert len(res["trades"]) > 0
    assert MANDATORY_SHARPE_CAVEAT in res["caveat"]


def test_multi_strategy_backtest():
    for strat in ["iron_condor", "bull_put_spread", "directional_momentum"]:
        res = backtest_engine.run_backtest(symbol="NIFTY", strategy=strat, days_history=30)
        assert res["strategy"] == strat
        assert res["sample_size"] >= 15
        assert res["win_rate"] >= 0.0
