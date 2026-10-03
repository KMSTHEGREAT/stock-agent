"""
Pydantic schemas for structured output from the LLM.
These define the exact JSON structure the LLM must produce.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from datetime import datetime


class Recommendation(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class MarketCapCategory(str, Enum):
    LARGE_CAP = "large_cap"      # > ₹1,00,000 crore
    MID_CAP = "mid_cap"          # ₹25,000 - ₹1,00,000 crore
    SMALL_CAP = "small_cap"      # < ₹25,000 crore


class TechnicalMetrics(BaseModel):
    """Technical analysis output."""
    symbol: str
    current_price: float = Field(..., description="Current price in INR")
    rsi: float = Field(..., ge=0, le=100, description="Relative Strength Index")
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = Field(None, description="Average True Range in INR")
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_position: Optional[float] = Field(None, ge=0, le=1, description="Position within Bollinger Bands")
    volume_ma_20: Optional[float] = None
    volume_ratio: Optional[float] = Field(None, description="Current volume vs 20-day average")
    trend: TrendDirection
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    near_circuit: bool = Field(False, description="Near circuit limit")
    price_change_pct_1d: Optional[float] = None
    price_change_pct_1w: Optional[float] = None
    price_change_pct_1m: Optional[float] = None


class FundamentalMetrics(BaseModel):
    """Fundamental analysis output."""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    
    # Growth
    revenue_growth_1y: Optional[float] = None
    revenue_growth_3y_cagr: Optional[float] = None
    profit_growth_1y: Optional[float] = None
    
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roce: Optional[float] = Field(None, description="Return on Capital Employed (more relevant in India)")
    
    # Leverage
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    # India-specific
    promoter_holding: Optional[float] = Field(None, description="Promoter holding percentage")
    promoter_holding_change: Optional[float] = None
    pledged_pct: Optional[float] = Field(None, description="Percentage of promoter shares pledged")
    fii_holding: Optional[float] = None
    dii_holding: Optional[float] = None
    market_cap_cr: Optional[float] = Field(None, description="Market cap in INR crores")
    market_cap_category: Optional[MarketCapCategory] = None
    
    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None


class NewsItem(BaseModel):
    """Single news item."""
    title: str
    source: str
    summary: str = Field("", max_length=300)
    published: Optional[str] = None
    sentiment: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment score -1 to 1")
    relevance: Optional[float] = Field(None, ge=0, le=1)


class MarketContext(BaseModel):
    """Overall market context for Indian markets."""
    nifty_trend: TrendDirection
    sensex_trend: TrendDirection
    nifty_change_pct: Optional[float] = None
    vix_level: Optional[float] = None
    market_regime: str = Field("neutral", description="bull/bear/neutral")
    fii_flow_cr: Optional[float] = Field(None, description="FII net flow in INR crores")
    dii_flow_cr: Optional[float] = Field(None, description="DII net flow in INR crores")
    crude_oil_trend: Optional[TrendDirection] = None
    usd_inr_trend: Optional[TrendDirection] = None


class RiskFlags(BaseModel):
    """India-specific risk indicators."""
    promoter_pledge_risk: bool = Field(False, description="High promoter pledging")
    fii_outflow_risk: bool = Field(False, description="FII outflow pressure")
    liquidity_risk: bool = Field(False, description="Low liquidity")
    regulatory_risk: Optional[str] = None
    corporate_governance_risk: Optional[str] = None


class StockAnalysis(BaseModel):
    """Complete stock analysis output."""
    # Identity
    symbol: str = Field(..., description="NSE symbol (e.g., RELIANCE)")
    company_name: str
    sector: Optional[str] = None
    
    # Scores (0-100)
    technical_score: int = Field(..., ge=0, le=100)
    fundamental_score: int = Field(..., ge=0, le=100)
    news_score: int = Field(..., ge=0, le=100)
    macro_score: int = Field(..., ge=0, le=100)
    overall_score: int = Field(..., ge=0, le=100)
    
    # Technical metrics (computed)
    technical_metrics: TechnicalMetrics
    
    # Fundamental metrics (computed)
    fundamental_metrics: FundamentalMetrics
    
    # Recommendation
    recommendation: Recommendation
    entry_price: Optional[float] = Field(None, description="Suggested entry price in INR")
    target_price: Optional[float] = Field(None, description="Target price in INR")
    stop_loss: Optional[float] = Field(None, description="Stop loss price in INR")
    confidence_level: float = Field(..., ge=0, le=1)
    
    # Risk flags
    risk_flags: RiskFlags
    
    # LLM-generated content
    analysis_summary: str = Field(..., max_length=500, description="Brief analysis summary")
    key_risks: list[str] = Field(default_factory=list, max_length=5)
    key_catalysts: list[str] = Field(default_factory=list, max_length=5)
    
    # Context
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    market_context: Optional[MarketContext] = None
    
    @field_validator('overall_score')
    @classmethod
    def validate_overall_score(cls, v, info):
        """Ensure overall score is consistent with sub-scores."""
        # This will be computed deterministically, not by LLM
        return v


class WatchlistItem(BaseModel):
    """Watchlist entry."""
    symbol: str
    company_name: str
    added_date: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
    target_score: Optional[int] = Field(None, ge=0, le=100)


class AnalysisReport(BaseModel):
    """Complete analysis report for a watchlist."""
    watchlist_name: str
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    total_stocks: int
    analyses: list[StockAnalysis]
    disclaimer: str = Field(default="For personal research only. Not investment advice.")