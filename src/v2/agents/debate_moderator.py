"""
Debate Moderator Agent for V2.
Weighs bull vs bear arguments and determines the winning side.
This is the key innovation from the TradingAgents framework.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import DebateResult
from src.v2.llm import call_llm, extract_json

logger = logging.getLogger(__name__)


def debate_moderator_node(state: dict) -> dict:
    """
    LangGraph node: Debate Moderator.
    
    Evaluates both bull and bear arguments, considers the analyst
    scores, and determines the overall sentiment direction.
    """
    symbol = state["symbol"]
    
    logger.info(f"[Debate Moderator] Evaluating arguments for {symbol}")
    
    try:
        bull = state.get("bull_thesis")
        bear = state.get("bear_thesis")
        tech = state.get("technical_output")
        fund = state.get("fundamental_output")
        news = state.get("news_output")
        
        # Build the debate evaluation prompt
        prompt = _build_debate_prompt(symbol, bull, bear, tech, fund, news)
        
        # Call LLM to moderate
        result = extract_json(call_llm(
            prompt=prompt,
            system=_DEBATE_SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for consistent judgment
            num_predict=500
        ))
        
        if result:
            debate = DebateResult(
                bull_thesis=bull.thesis if bull else "",
                bear_thesis=bear.thesis if bear else "",
                winner=_validate_winner(result.get("winner", "inconclusive")),
                reasoning=_safe_str(result.get("reasoning", ""), 
                                   "Debate evaluation unavailable"),
                sentiment_score=_safe_score(result.get("sentiment_score", 50)),
                confidence=_safe_confidence(result.get("confidence", 0.5)),
            )
            
            logger.info(f"[Debate Moderator] {symbol}: winner={debate.winner}, "
                       f"score={debate.sentiment_score}, conf={debate.confidence}")
            
            return {"debate_result": debate}
        
        # Fallback: compute from analyst scores
        return {"debate_result": _fallback_debate(symbol, bull, bear, tech, fund, news)}
        
    except Exception as e:
        logger.error(f"[Debate Moderator] Error: {e}")
        return {
            "debate_result": _fallback_debate(symbol, None, None, None, None, None),
            "errors": [f"Debate moderator error: {str(e)}"],
        }


_DEBATE_SYSTEM_PROMPT = """You are a debate moderator evaluating investment theses for an Indian stock.

Your job is to:
1. Read both the BULL and BEAR arguments
2. Weigh the evidence quality and logical consistency
3. Consider the underlying data scores
4. Determine which side has the stronger case
5. Assign an overall sentiment score

RULES:
- Be objective — don't favor either side by default
- Higher conviction + better evidence = stronger argument
- The sentiment score should reflect the balance of evidence
- 50 = perfectly balanced, 80+ = strong bull, 20- = strong bear

Respond with ONLY this JSON:
{
  "winner": "bull or bear or inconclusive",
  "reasoning": "2-3 sentences explaining your judgment",
  "sentiment_score": 55,
  "confidence": 0.7
}"""


def _build_debate_prompt(symbol, bull, bear, tech, fund, news) -> str:
    """Build the debate evaluation prompt."""
    
    parts = []
    parts.append(f"DEBATE EVALUATION: {symbol}")
    parts.append(f"DATE: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Bull thesis
    parts.append("\n=== BULL ARGUMENT ===")
    if bull:
        parts.append(f"Thesis: {bull.thesis}")
        parts.append(f"Conviction: {bull.conviction_level:.0%}")
        if bull.supporting_evidence:
            parts.append("Evidence:")
            for e in bull.supporting_evidence[:4]:
                parts.append(f"  + {e}")
        if bull.counter_arguments_addressed:
            parts.append("Counter-arguments addressed:")
            for c in bull.counter_arguments_addressed[:2]:
                parts.append(f"  → {c}")
    else:
        parts.append("No bull thesis available.")
    
    # Bear thesis
    parts.append("\n=== BEAR ARGUMENT ===")
    if bear:
        parts.append(f"Thesis: {bear.thesis}")
        parts.append(f"Conviction: {bear.conviction_level:.0%}")
        if bear.supporting_evidence:
            parts.append("Evidence:")
            for e in bear.supporting_evidence[:4]:
                parts.append(f"  - {e}")
        if bear.counter_arguments_addressed:
            parts.append("Counter-arguments addressed:")
            for c in bear.counter_arguments_addressed[:2]:
                parts.append(f"  → {c}")
    else:
        parts.append("No bear thesis available.")
    
    # Underlying data scores for context
    parts.append("\n=== DATA CONTEXT ===")
    if tech:
        parts.append(f"Technical Score: {tech.score:.0f}/100 (confidence: {tech.confidence:.0%})")
    if fund:
        parts.append(f"Fundamental Score: {fund.score:.0f}/100 (confidence: {fund.confidence:.0%})")
    if news:
        parts.append(f"News Score: {news.score:.0f}/100")
    
    context = "\n".join(parts)
    
    prompt = f"""{context}

TASK: Evaluate this debate and determine the winner.

Consider:
1. Quality of evidence (specific data > vague claims)
2. Logical consistency of arguments
3. Whether counter-arguments adequately address concerns
4. Conviction levels (high conviction + weak evidence = overconfident)
5. Alignment with underlying data scores

Respond with ONLY valid JSON."""
    
    return prompt


def _fallback_debate(symbol, bull, bear, tech, fund, news) -> DebateResult:
    """Compute debate result from scores without LLM."""
    
    # Weighted score from analysts
    scores = []
    weights = []
    
    if tech:
        scores.append(tech.score)
        weights.append(0.30)
    if fund:
        scores.append(fund.score)
        weights.append(0.40)
    if news:
        scores.append(news.score)
        weights.append(0.30)
    
    if scores:
        total = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        sentiment_score = total / weight_sum if weight_sum > 0 else 50
    else:
        sentiment_score = 50
    
    # Adjust based on conviction levels
    if bull and bear:
        # If bull is more confident than bear, nudge score up
        conviction_diff = bull.conviction_level - bear.conviction_level
        sentiment_score += conviction_diff * 10  # Max ±10 adjustment
    
    sentiment_score = max(0, min(100, sentiment_score))
    
    # Determine winner
    if sentiment_score > 55:
        winner = "bull"
    elif sentiment_score < 45:
        winner = "bear"
    else:
        winner = "inconclusive"
    
    return DebateResult(
        bull_thesis=bull.thesis if bull else "",
        bear_thesis=bear.thesis if bear else "",
        winner=winner,
        reasoning=f"Computed from analyst scores: Tech={tech.score if tech else 'N/A'}, "
                 f"Fund={fund.score if fund else 'N/A'}, News={news.score if news else 'N/A'}",
        sentiment_score=round(sentiment_score, 1),
        confidence=0.4,  # Low confidence for fallback
    )


def _validate_winner(value) -> str:
    """Validate winner string."""
    if value and isinstance(value, str):
        value = value.lower().strip()
        if value in ("bull", "bear", "inconclusive"):
            return value
    return "inconclusive"


def _safe_score(value) -> float:
    try:
        if value is None:
            return 50.0
        return max(0.0, min(100.0, float(value)))
    except:
        return 50.0


def _safe_confidence(value) -> float:
    try:
        if value is None:
            return 0.5
        return max(0.0, min(1.0, float(value)))
    except:
        return 0.5


def _safe_str(value, default: str) -> str:
    if value is None:
        return default
    return str(value)[:500]