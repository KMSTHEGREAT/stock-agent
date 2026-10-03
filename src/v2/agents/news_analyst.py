"""
News Analyst Agent for V2.
Uses SearXNG instead of DDGS, with LLM for sentiment scoring.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import NewsOutput, NewsItem
from src.v2.search.searxng import SearXNGSearch
from src.v2.llm import call_llm, extract_json

logger = logging.getLogger(__name__)


def news_analyst_node(state: dict) -> dict:
    """
    LangGraph node: News Analyst.
    
    Fetches news via SearXNG and scores sentiment using LLM.
    """
    symbol = state["symbol"]
    company_name = state.get("company_name", symbol)
    
    logger.info(f"[News Analyst] Analyzing {symbol}")
    
    try:
        # Fetch news via SearXNG
        search = SearXNGSearch()
        news_items_raw = search.search_stock_news(symbol, company_name)
        
        if not news_items_raw:
            logger.warning(f"[News Analyst] No news for {symbol}")
            return {
                "news_output": _neutral_output(symbol, "No recent news found"),
            }
        
        # Convert to NewsItem objects
        news_items = []
        for item in news_items_raw[:10]:  # Limit to 10
            news_items.append(NewsItem(
                title=item.get("title") or "",
                source=item.get("source") or "unknown",
                summary=(item.get("summary") or "")[:200],
                url=item.get("url") or "",
                published=item.get("published") or "",
            ))
        
        # Score sentiment using LLM
        sentiment_result = _score_sentiment_with_llm(symbol, news_items)
        
        # Build output
        output = NewsOutput(
            agent_name="news_analyst",
            symbol=symbol,
            score=sentiment_result.get("score", 50.0),
            confidence=sentiment_result.get("confidence", 0.5),
            key_findings=sentiment_result.get("findings", []),
            data_sources=["searxng"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            
            news_items=news_items,
            positive_count=sentiment_result.get("positive", 0),
            negative_count=sentiment_result.get("negative", 0),
            neutral_count=sentiment_result.get("neutral", 0),
            key_events=sentiment_result.get("events", []),
        )
        
        logger.info(f"[News Analyst] {symbol}: score={output.score}, "
                   f"{len(news_items)} items, sentiment={output.positive_count}pos/{output.negative_count}neg")
        
        return {"news_output": output}
        
    except Exception as e:
        logger.error(f"[News Analyst] Error: {e}")
        return {
            "news_output": _neutral_output(symbol, f"News analysis error: {str(e)}"),
            "errors": [f"News analyst error: {str(e)}"],
        }


def _score_sentiment_with_llm(symbol: str, news_items: list) -> dict:
    """Use LLM to score news sentiment."""
    
    # Build news summary for LLM
    news_text = ""
    for i, item in enumerate(news_items[:5], 1):
        news_text += f"{i}. {item.title}\n"
        if item.summary:
            news_text += f"   {item.summary[:100]}\n"
    
    prompt = f"""Analyze news sentiment for {symbol} (Indian stock market):

{news_text}

Analyze the overall sentiment and respond with ONLY this JSON:
{{
  "sentiment_score": 50,
  "sentiment_label": "neutral",
  "positive_count": 0,
  "negative_count": 0,
  "neutral_count": 0,
  "key_events": ["event 1", "event 2"],
  "findings": ["finding 1", "finding 2"],
  "confidence": 0.5
}}

Scoring: 0=very negative, 50=neutral, 100=very positive.
Count positive/negative/neutral items.
List key events and findings as short strings.
"""
    
    # Call LLM
    result = extract_json(call_llm(prompt, temperature=0.2, num_predict=400))
    
    if result:
        # Validate and sanitize
        score = result.get("sentiment_score", 50)
        if not isinstance(score, (int, float)):
            score = 50
        score = max(0, min(100, float(score)))
        
        return {
            "score": round(score, 1),
            "confidence": _safe_float(result.get("confidence", 0.5), 0.5),
            "positive": int(_safe_float(result.get("positive_count", 0))),
            "negative": int(_safe_float(result.get("negative_count", 0))),
            "neutral": int(_safe_float(result.get("neutral_count", 0))),
            "events": _safe_list(result.get("key_events")),
            "findings": _safe_list(result.get("findings")),
        }
    
    # Fallback: keyword-based scoring
    return _keyword_fallback(symbol, news_items)


def _keyword_fallback(symbol: str, news_items: list) -> dict:
    """Fallback keyword-based sentiment analysis."""
    positive_words = ['surge', 'gain', 'rise', 'profit', 'growth', 'upgrade',
                     'bullish', 'strong', 'record', 'high', 'beat', 'positive']
    negative_words = ['fall', 'drop', 'decline', 'loss', 'downgrade', 'sell',
                     'bearish', 'weak', 'low', 'miss', 'negative', 'probe', 'crash']
    
    pos = 0
    neg = 0
    
    for item in news_items:
        text = (item.title + " " + item.summary).lower()
        pos += sum(1 for w in positive_words if w in text)
        neg += sum(1 for w in negative_words if w in text)
    
    total = pos + neg
    if total == 0:
        score = 50
    else:
        score = 50 + (pos / total - 0.5) * 100
    
    return {
        "score": round(max(0, min(100, score)), 1),
        "confidence": 0.4,  # Low confidence for fallback
        "positive": pos,
        "negative": neg,
        "neutral": len(news_items) - pos - neg if len(news_items) > pos + neg else 0,
        "events": [],
        "findings": [f"Keyword analysis: {pos} positive, {neg} negative signals"],
    }


def _neutral_output(symbol: str, reason: str) -> NewsOutput:
    """Create a neutral news output."""
    return NewsOutput(
        agent_name="news_analyst",
        symbol=symbol,
        score=50.0,
        confidence=0.3,
        key_findings=[reason],
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