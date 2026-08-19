from typing import Any, Dict, List


class QuantitativeScreenResult:

    def __init__(
        self,
        symbol: str,
        screen_name: str,
        passed: bool,
        reason: str,
        metrics: Dict[str, Any],
    ):
        self.symbol = symbol
        self.screen_name = screen_name
        self.passed = passed
        self.reason = reason
        self.metrics = metrics


def screen_volume_breakout(snapshot: Dict[str, Any]) -> QuantitativeScreenResult:
    """Check for price breakout near resistance with strong volume multiplier."""
    ltp = snapshot.get("ltp", 0.0)
    resistance = snapshot.get("resistance", ltp * 1.05)
    volume = snapshot.get("volume", 0)
    avg_vol = 1000000  # 20-day avg volume baseline

    vol_mult = round(volume / avg_vol, 2) if avg_vol > 0 else 1.0
    near_resistance = ltp >= (resistance * 0.995)

    passed = (vol_mult >= 1.8) and near_resistance
    reason = (
        f"LTP {ltp} testing resistance {resistance} with {vol_mult}x avg volume."
        if passed
        else f"No volume breakout (vol={vol_mult}x, dist to resistance={round(resistance - ltp, 2)})."
    )

    return QuantitativeScreenResult(
        symbol=snapshot.get("symbol", ""),
        screen_name="Volume Breakout Screen",
        passed=passed,
        reason=reason,
        metrics={
            "vol_multiplier": vol_mult,
            "ltp": ltp,
            "resistance": resistance,
        },
    )


def screen_pcr_extreme_reversal(
    snapshot: Dict[str, Any],
) -> QuantitativeScreenResult:
    """Check for PCR extreme levels (< 0.70 oversold bullish reversal or > 1.35 overbought bearish reversal)."""
    pcr = snapshot.get("pcr", 1.0)
    passed = (pcr <= 0.75) or (pcr >= 1.35)

    if pcr <= 0.75:
        reason = f"Extreme low PCR ({pcr}) indicating oversold sentiment and potential bullish reversal."
    elif pcr >= 1.35:
        reason = f"Extreme high PCR ({pcr}) indicating heavy put writing support or overbought options sentiment."
    else:
        reason = f"PCR ({pcr}) within normal range [0.75, 1.35]."

    return QuantitativeScreenResult(
        symbol=snapshot.get("symbol", ""),
        screen_name="PCR Extreme Reversal Screen",
        passed=passed,
        reason=reason,
        metrics={"pcr": pcr},
    )


def screen_iv_spike(snapshot: Dict[str, Any]) -> QuantitativeScreenResult:
    """Check for IV spike indicating impending news/earnings volatility expansion."""
    iv = snapshot.get("iv", 20.0)
    passed = iv >= 25.0
    reason = (
        f"IV expanded to {iv}% (threshold 25%). High volatility option strategy opportunity."
        if passed
        else f"IV at normal level ({iv}%)."
    )

    return QuantitativeScreenResult(
        symbol=snapshot.get("symbol", ""),
        screen_name="IV Spike Screen",
        passed=passed,
        reason=reason,
        metrics={"iv": iv},
    )


ALL_SCREENS = [
    screen_volume_breakout,
    screen_pcr_extreme_reversal,
    screen_iv_spike,
]


def run_all_screens(snapshot: Dict[str, Any]) -> List[QuantitativeScreenResult]:
    """Run all quantitative python screens against a symbol market snapshot."""
    results = []
    for screen_func in ALL_SCREENS:
        res = screen_func(snapshot)
        if res.passed:
            results.append(res)
    return results
