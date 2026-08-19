from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class CurrentDataSchema(BaseModel):
    ltp: float = Field(..., description="Last Traded Price")
    iv: float | None = Field(default=None, description="Implied Volatility (%)")
    pcr: float | None = Field(default=None, description="Put-Call Ratio")
    oi_trend: str = Field(default="neutral", description="Open Interest Trend (e.g. long buildup, short covering)")


class FactorSchema(BaseModel):
    point: str = Field(..., description="Observed factor or evidence point")
    source: str = Field(..., description="Source of the evidence (e.g., News, Option Chain, Technicals, Books)")


class BacktestContextSchema(BaseModel):
    sharpe: float | None = Field(default=None, description="Sharpe ratio from backtest")
    sample_size: int | None = Field(default=None, description="Number of historical trades/samples")
    win_rate: float | None = Field(default=None, description="Win rate percentage")
    max_drawdown: float | None = Field(default=None, description="Max drawdown percentage")
    caveat: str = Field(
        default="Short backtest windows yield unreliable Sharpe ratios.",
        description="Mandatory caveat on backtest reliability",
    )


class KeyLevelsSchema(BaseModel):
    support: float = Field(..., description="Key support level")
    resistance: float = Field(..., description="Key resistance level")


class SymbolAnalysisRequest(BaseModel):
    query: str = Field(..., description="User question or research topic")
    symbol: str = Field(..., description="NSE Trading symbol (e.g. RELIANCE, NIFTY)")
    instrument_type: Literal["equity", "options"] = Field(
        default="options", description="Instrument type"
    )


class SymbolAnalysisResponse(BaseModel):
    """Enforced Section 5 structured output schema from architecture.md."""

    symbol: str
    as_of: str
    current_data: CurrentDataSchema
    bullish_factors: list[FactorSchema] = Field(default_factory=list)
    bearish_factors: list[FactorSchema] = Field(default_factory=list)
    backtest_context: BacktestContextSchema
    key_levels: KeyLevelsSchema
    invalidation_conditions: str = Field(
        ..., description="Specific market conditions under which this view becomes invalid"
    )
    confidence: str = Field(
        ..., description="low | moderate | high — with clear justification"
    )
    explicit_note: str = Field(
        default="This is decision-support analysis, not a recommendation to buy or sell.",
        description="Mandatory decision-support disclaimer",
    )

    @field_validator("explicit_note", mode="before")

    def validate_disclaimer(cls, v: Any) -> str:
        standard_disclaimer = "This is decision-support analysis, not a recommendation to buy or sell."
        if not v or standard_disclaimer not in str(v):
            return standard_disclaimer
        return v
