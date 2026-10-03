"""
Sentiment Analyst Agent for V2.
Uses LLM to assess market sentiment from multiple signals.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import SentimentOutput
from src.v2.llm import call_llm, extract_json

logger = logging.getLogger(__name__)


def sentiment_analyst_node(state: dict) -> dict:
    """
    LangGraph node: Sentiment Analyst.
    
    Analyzes overall market sentiment for the stock using LLM,
    considering technical, fundamental, and news data from other analysts.
    """
    symbol = state["symbol"]
    
    logger.info(f"[Sentiment Analyst] Analyzing {symbol}")
    
    try:
        # Gather data from other analysts (if available)
        tech = state.get("technical_output")
        fund = state.get("fundamental_output")
        news = state.get("news_output")
        
        # Build context for LLM
        context = _build_context(tech, fund, news)
        
        # Use LLM to assess sentiment
        prompt = f"""Assess market sentiment for {symbol} (Indian stock market).

CONTEXT:
{context}

Consider:
1. Technical momentum (trend, RSI, price action)
2. Fundamental strength (P/E, ROCE, growth, promoter holding)
3. News sentiment (positive/negative events)
4. India-specific factors (FII flows, RBI policy, sector trends)

Respond with ONLY this JSON:
{{
  "overall_sentiment": "positive or negative or neutral",
  "social_sentiment": "positive or negative or neutral",
  "analyst_sentiment": "positive or negative or neutral",
  "institutional_flow": "buying or selling or neutral",
  "score": 50,
  "confidence": 0.5,
  "findings": ["key finding 1", "key finding 2", "key finding 3"]
}}

Scoring: 0=very negative, 50=neutral, 100=very positive.
"""
        
        result = extract_json(call_llm(prompt, temperature=0.3, num_predict=400))
        
        if result:
            score = result.get("score", 50)
            if not isinstance(score, (int, float)):
                score = 50
            score = max(0, min(100, float(score)))
            
            output = SentimentOutput(
                agent_name="sentiment_analyst",
                symbol=symbol,
                score=round(score, 1),
                confidence=_safe_float(result.get("confidence", 0.5)),
                key_findings=_safe_list(result.get("findings")),
                data_sources=["llm_analysis"],
                timestamp=datetime.now(timezone.utc).isoformat(),
                
                overall_sentiment=_validate_sentiment(result.get("overall_sentiment")),
                social_sentiment=_validate_sentiment(result.get("social_sentiment")),
                analyst_sentiment=_validate_sentiment(result.get("analyst_sentiment")),
                institutional_flow=_validate_flow(result.get("institutional_flow")),
            )
            
            logger.info(f"[Sentiment Analyst] {symbol}: score={output.score}, "
                       f"sentiment={output.overall_sentiment}")
            
            return {"sentiment_output": output}
        
        # Fallback: compute from sub-analyst scores
        return {"sentiment_output": _fallback_sentiment(symbol, tech, fund, news)}
        
    except Exception as e:
        logger.error(f"[Sentiment Analyst] Error: {e}")
        return {
            "sentiment_output": _neutral_sentiment(symbol),
            "errors": [f"Sentiment analyst error: {str(e)}"],
        }


def _build_context(tech, fund, news) -> str:
    """Build context string from other analyst outputs."""
    parts = []
    
    if tech:
        parts.append(f"Technical: Score {tech.score}/100, Trend: {tech.trend}, RSI: {tech.rsi}")
        if tech.key_findings:
            parts.append(f"  Key: {', '.join(tech.key_findings[:2])}")
    
    if fund:
        parts.append(f"Fundamental: Score {fund.score}/100, P/E: {fund.pe_ratio}, ROCE: {fund.roce}")
        if fund.key_findings:
            parts.append(f"  Key: {', '.join(fund.key_findings[:2])}")
    
    if news:
        parts.append(f"News: Score {news.score}/100, Items: {len(news.news_items)}")
        if news.key_findings:
            parts.append(f"  Key: {', '.join(news.key_findings[:2])}")
    
    if not parts:
        parts.append("No data available from other analysts")
    
    return "\n".join(parts)


def _validate_sentiment(value) -> str:
    """Validate sentiment string."""
    if value and isinstance(value, str):
        value = value.lower()
        if value in ("positive", "negative", "neutral"):
            return value
    return "neutral"


def _validate_flow(value) -> str:
    """Validate institutional flow."""
    if value and isinstance(value, str):
        value = value.lower()
        if value in ("buying", "selling", "neutral"):
            return value
    return "neutral"


def _fallback_sentiment(symbol: str, tech, fund, news) -> SentimentOutput:
    """Compute sentiment from sub-analyst scores without LLM."""
    scores = []
    weights = []
    
    if tech:
        scores.append(tech.score)
        weights.append(0.35)
    if fund:
        scores.append(fund.score)
        weights.append(0.40)
    if news:
        scores.append(news.score)
        weights.append(0.25)
    
    if scores:
        # Weighted average
        total = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        score = total / weight_sum if weight_sum > 0 else 50
    else:
        score = 50
    
    if score > 60:
        sentiment = "positive"
    elif score < 40:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return SentimentOutput(
        agent_name="sentiment_analyst",
        symbol=symbol,
        score=round(max(0, min(100, score)), 1),
        confidence=0.4,  # Low confidence for fallback
        key_findings=["Computed from sub-analyst scores (no LLM)"],
        data_sources=["computed"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_sentiment=sentiment,
    )


def _neutral_sentiment(symbol: str) -> SentimentOutput:
    """Neutral sentiment output."""
    return SentimentOutput(
        agent_name="sentiment_analyst",
        symbol=symbol,
        score=50.0,
        confidence=0.3,
        key_findings=["Sentiment analysis unavailable"],
        data_sources=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value) if value is not None else default
    except:
        return default


def _safe_list(value) -> list:
    if isinstance(value, list):
        return [str(v)[:100] for v in value[:5]]
    return []