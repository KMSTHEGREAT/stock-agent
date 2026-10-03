"""
Technical Analyst Agent for V2.
Reuses V1's technical analysis engine but outputs structured data for LangGraph.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import TechnicalOutput, TrendDirection
from src.v1.analysis.technical import TechnicalAnalyzer
from src.v1.analysis.technical_scorer import TechnicalScorer

logger = logging.getLogger(__name__)


def technical_analyst_node(state: dict) -> dict:
    """
    LangGraph node: Technical Analyst.
    
    Computes technical indicators and produces a TechnicalOutput.
    This is deterministic — no LLM involved in calculations.
    """
    symbol = state["symbol"]
    logger.info(f"[Technical Analyst] Analyzing {symbol}")
    
    try:
        # Reuse V1's technical analyzer
        analyzer = TechnicalAnalyzer()
        scorer = TechnicalScorer()
        
        # Get metrics
        metrics = analyzer.analyze(symbol, period="1y")
        
        if metrics is None:
            logger.warning(f"[Technical Analyst] No data for {symbol}")
            return {
                "technical_output": None,
                "errors": [f"Technical analysis failed for {symbol}"],
            }
        
        # Score it
        scores = scorer.score(metrics)
        
        # Build structured output
        output = TechnicalOutput(
            agent_name="technical_analyst",
            symbol=symbol,
            score=scores.get("total", 50.0),
            confidence=_calc_confidence(scores),
            key_findings=_extract_findings(metrics, scores),
            data_sources=["yfinance"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            
            # Technical specifics
            current_price=metrics.current_price,
            trend=metrics.trend.value if metrics.trend else "neutral",
            rsi=metrics.rsi,
            macd_signal=_macd_signal(metrics),
            support_level=metrics.support_level,
            resistance_level=metrics.resistance_level,
            atr=metrics.atr,
            price_change_1w=metrics.price_change_pct_1w,
            price_change_1m=metrics.price_change_pct_1m,
        )
        
        logger.info(f"[Technical Analyst] {symbol}: score={output.score}, "
                   f"trend={output.trend}, RSI={output.rsi}")
        
        return {"technical_output": output}
        
    except Exception as e:
        logger.error(f"[Technical Analyst] Error: {e}")
        return {
            "technical_output": None,
            "errors": [f"Technical analyst error: {str(e)}"],
        }


def _calc_confidence(scores: dict) -> float:
    """Calculate confidence based on score consistency."""
    sub_scores = [
        scores.get("trend", 50),
        scores.get("momentum", 50),
        scores.get("volatility", 50),
        scores.get("volume", 50),
    ]
    
    # Low variance = high confidence
    avg = sum(sub_scores) / len(sub_scores)
    variance = sum((s - avg) ** 2 for s in sub_scores) / len(sub_scores)
    std = variance ** 0.5
    
    # Lower std = higher confidence (max 1.0)
    confidence = max(0.3, min(1.0, 1.0 - (std / 25)))
    
    return round(confidence, 2)


def _extract_findings(metrics, scores: dict) -> list:
    """Extract key findings from technical analysis."""
    findings = []
    
    if metrics.rsi is not None:
        if metrics.rsi > 70:
            findings.append(f"RSI {metrics.rsi:.0f} — overbought, potential reversal risk")
        elif metrics.rsi < 30:
            findings.append(f"RSI {metrics.rsi:.0f} — oversold, potential bounce opportunity")
        else:
            findings.append(f"RSI {metrics.rsi:.0f} — neutral zone")
    
    if metrics.trend:
        findings.append(f"Trend: {metrics.trend.value}")
    
    if metrics.sma_20 and metrics.sma_50:
        if metrics.sma_20 > metrics.sma_50:
            findings.append("Golden cross: SMA20 > SMA50 (bullish)")
        else:
            findings.append("Death cross: SMA20 < SMA50 (bearish)")
    
    if metrics.current_price and metrics.sma_200:
        if metrics.current_price > metrics.sma_200:
            findings.append("Price above 200-day SMA (long-term bullish)")
        else:
            findings.append("Price below 200-day SMA (long-term bearish)")
    
    if metrics.price_change_pct_1m is not None:
        findings.append(f"1-month change: {metrics.price_change_pct_1m:+.1f}%")
    
    if metrics.atr and metrics.current_price:
        atr_pct = (metrics.atr / metrics.current_price) * 100
        findings.append(f"Volatility (ATR): {atr_pct:.1f}% of price")
    
    return findings[:5]  # Max 5 findings


def _macd_signal(metrics) -> str:
    """Determine MACD signal."""
    if metrics.macd is not None and metrics.macd_signal is not None:
        if metrics.macd > metrics.macd_signal:
            return "bullish"
        else:
            return "bearish"
    return "neutral"