from app.scanner.screens import run_all_screens, screen_volume_breakout


def test_scanner_screens():
    snapshot_breakout = {
        "symbol": "RELIANCE",
        "ltp": 2980.0,
        "resistance": 2980.0,
        "volume": 2500000,  # 2.5x baseline
        "pcr": 1.15,
        "iv": 22.0,
    }

    results = run_all_screens(snapshot_breakout)
    assert len(results) >= 1
    screen_names = [r.screen_name for r in results]
    assert "Volume Breakout Screen" in screen_names
