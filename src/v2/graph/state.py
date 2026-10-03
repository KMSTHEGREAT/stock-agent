"""
LangGraph state definitions for the V2 analysis pipeline.
These define the data that flows between agent nodes.
"""
from typing import TypedDict, Literal, Optional, Annotated
from pydantic import BaseModel, Field
from enum import Enum
import operator


class Rating(str, Enum):
    """Five-tier rating system (TradingAgents style)."""
    BUY = "buy"                # Strong conviction, initiate position
    OVERWEIGHT = "overweight"  # Positive, increase position
    HOLD = "hold"              # Neutral, maintain current
    UNDERWEIGHT = "underweight" # Negative, reduce position
    SELL = "sell"              # Strong conviction, exit position


class AnalystOutput(BaseModel):
    """Base output from any analyst agent."""
    agent_name: str
    symbol: str
    score: float = Field(ge=0, le=100, description="Score 0-100")
    confidence: float = Field(ge=0, le=1)
    key_findings: list[str] = Field(default_factory=list, max_items=5)
    data_sources: list[str] = Field(default_factory=list)
    timestamp: str


class TechnicalOutput(AnalystOutput):
    """Technical analyst output."""
    current_price: Optional[float] = None
    trend: Optional[str] = None
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    atr: Optional[float] = None
    price_change_1w: Optional[float] = None
    price_change_1m: Optional[float] = None


class FundamentalOutput(AnalystOutput):
    """Fundamental analyst output."""
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roce: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    promoter_holding: Optional[float] = None
    pledged_pct: Optional[float] = None
    market_cap_cr: Optional[float] = None
    revenue_growth: Optional[float] = None


class NewsItem(BaseModel):
    """Single news item."""
    title: str
    source: str
    summary: str = ""
    url: str = ""
    published: str = ""


class NewsOutput(AnalystOutput):
    """News analyst output."""
    news_items: list[NewsItem] = Field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    key_events: list[str] = Field(default_factory=list)


class SentimentOutput(AnalystOutput):
    """Sentiment analyst output."""
    overall_sentiment: str = "neutral"
    social_sentiment: Optional[str] = None
    analyst_sentiment: Optional[str] = None
    institutional_flow: Optional[str] = None


class ResearchThesis(BaseModel):
    """Researcher's argument (bull or bear)."""
    researcher_type: Literal["bull", "bear"]
    symbol: str
    thesis: str
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_arguments_addressed: list[str] = Field(default_factory=list)
    conviction_level: float = Field(default=0.5, ge=0, le=1)


class DebateResult(BaseModel):
    """Output from debate moderation."""
    bull_thesis: str
    bear_thesis: str
    winner: Literal["bull", "bear", "inconclusive"]
    reasoning: str
    sentiment_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)


class RiskAssessment(BaseModel):
    """Risk manager output."""
    risk_score: float = Field(ge=0, le=100)
    volatility_risk: str = "medium"
    liquidity_risk: str = "low"
    promoter_risk: str = "none"
    regulatory_risk: str = "none"
    max_position_size_pct: float = 5.0
    suggested_stop_loss: Optional[float] = None
    risk_warnings: list[str] = Field(default_factory=list)


class FinalRecommendation(BaseModel):
    """Portfolio manager's final output."""
    symbol: str
    company_name: str
    rating: Rating
    overall_score: float = Field(ge=0, le=100)
    conviction: float = Field(ge=0, le=1)
    
    # Pricing
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    # Reasoning
    bull_thesis: str = ""
    bear_thesis: str = ""
    debate_summary: str = ""
    key_risks: list[str] = Field(default_factory=list)
    key_catalysts: list[str] = Field(default_factory=list)
    
    # Metadata
    analysis_timestamp: str = ""
    processing_time_seconds: float = 0.0
    data_quality_score: float = Field(default=0.5, ge=0, le=1)


# ============ LANGGRAPH STATE ============

class AnalysisState(TypedDict):
    """
    Main state that flows through the LangGraph workflow.
    
    The `operator.add` annotation means these lists accumulate
    across nodes rather than being overwritten.
    """
    # Input (set once at start)
    symbol: str
    company_name: str
    
    # Analyst outputs (set by analyst nodes, run in parallel)
    technical_output: Optional[TechnicalOutput]
    fundamental_output: Optional[FundamentalOutput]
    news_output: Optional[NewsOutput]
    sentiment_output: Optional[SentimentOutput]
    
    # Researcher outputs (set by bull/bear nodes, run in parallel)
    bull_thesis: Optional[ResearchThesis]
    bear_thesis: Optional[ResearchThesis]
    
    # Debate output
    debate_result: Optional[DebateResult]
    
    # Risk assessment
    risk_assessment: Optional[RiskAssessment]
    
    # Final output
    final_recommendation: Optional[FinalRecommendation]
    
    # Accumulating lists (use operator.add)
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    agent_logs: Annotated[list[dict], operator.add]
    
    # Metadata
    start_time: str
    current_step: str