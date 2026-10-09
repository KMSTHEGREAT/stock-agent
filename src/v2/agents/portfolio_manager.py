"""
Portfolio Manager Agent for V2.
Makes the final investment decision based on all inputs.
Uses deterministic rules with debate sentiment as primary input.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import FinalRecommendation, Rating
from src.v2.llm import call_llm, extract_json

logger = logging.getLogger(__name__)


def portfolio_manager_node(state: dict) -> dict:
    """
    LangGraph node: Portfolio Manager.
    
    The final decision-maker. Considers:
    - Debate result (sentiment score)
    - Risk assessment (risk score, position limits)
    - Individual analyst scores
    - Bull/bear conviction levels
    
    Produces: Buy/Overweight/Hold/Underweight/Sell
    """
    symbol = state["symbol"]
    company_name = state.get("company_name", symbol)
    
    logger.info(f"[Portfolio Manager] Making final decision for {symbol}")
    
    try:
        debate = state.get("debate_result")
        risk = state.get("risk_assessment")
        tech = state.get("technical_output")
        fund = state.get("fundamental_output")
        bull = state.get("bull_thesis")
        bear = state.get("bear_thesis")
        
        # Compute overall score from all inputs
        overall_score = _compute_overall_score(debate, tech, fund, risk)
        
        # Determine rating based on score AND risk
        rating = _determine_rating(overall_score, risk, debate)
        
        # Calculate conviction
        conviction = _calculate_conviction(debate, risk, tech, fund)
        
        # Calculate price levels
        entry_price = _calculate_entry_price(tech, rating)
        target_price = _calculate_target_price(tech, rating, risk)
        stop_loss = risk.suggested_stop_loss if risk else None
        
        # Calculate risk/reward
        risk_reward = _calculate_risk_reward(entry_price, target_price, stop_loss)
        
        # Generate LLM summary
        llm_summary = _generate_llm_summary(
            symbol, company_name, rating, overall_score,
            debate, bull, bear, risk
        )
        
        # Build final recommendation
        recommendation = FinalRecommendation(
            symbol=symbol,
            company_name=company_name,
            rating=rating,
            overall_score=round(overall_score, 1),
            conviction=round(conviction, 2),
            
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            
            bull_thesis=bull.thesis if bull else "",
            bear_thesis=bear.thesis if bear else "",
            debate_summary=llm_summary,
            key_risks=risk.risk_warnings if risk else [],
            key_catalysts=_extract_catalysts(bull),
            
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            data_quality_score=_assess_data_quality(tech, fund, debate),
        )
        
        logger.info(f"[Portfolio Manager] {symbol}: {rating.value.upper()} "
                   f"(score={overall_score:.1f}, conviction={conviction:.0%})")
        
        return {"final_recommendation": recommendation}
        
    except Exception as e:
        logger.error(f"[Portfolio Manager] Error: {e}")
        return {
            "final_recommendation": _error_recommendation(symbol, company_name),
            "errors": [f"Portfolio manager error: {str(e)}"],
        }


def _compute_overall_score(debate, tech, fund, risk) -> float:
    """Compute the final overall score from all components."""
    scores = []
    weights = []
    
    # Debate sentiment is the primary signal (highest weight)
    if debate:
        scores.append(debate.sentiment_score)
        weights.append(0.45)
    
    # Individual analyst scores
    if fund:
        scores.append(fund.score)
        weights.append(0.25)
    
    if tech:
        scores.append(tech.score)
        weights.append(0.20)
    
    # Risk adjustment (inverse — low risk = good)
    if risk:
        scores.append(100 - risk.risk_score)  # Invert: low risk → high score
        weights.append(0.10)
    
    # Weighted average
    if scores and weights:
        total = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        return total / weight_sum if weight_sum > 0 else 50
    
    return 50.0


def _determine_rating(overall_score: float, risk, debate) -> Rating:
    """Determine the five-tier rating."""
    
    # Risk overrides — high risk caps the rating
    if risk and risk.risk_score > 75:
        if overall_score > 60:
            return Rating.UNDERWEIGHT  # Cap at underweight for very high risk
        else:
            return Rating.SELL
    
    # Standard thresholds
    if overall_score >= 75:
        return Rating.BUY
    elif overall_score >= 62:
        return Rating.OVERWEIGHT
    elif overall_score >= 45:
        # If bear won debate decisively, downgrade
        if debate and debate.winner == "bear" and debate.sentiment_score < 40:
            return Rating.UNDERWEIGHT
        return Rating.HOLD
    elif overall_score >= 32:
        return Rating.UNDERWEIGHT
    else:
        return Rating.SELL


def _calculate_conviction(debate, risk, tech, fund) -> float:
    """Calculate conviction level for the recommendation."""
    conviction = 0.5  # Base
    
    if debate:
        # Higher debate confidence = higher conviction
        conviction += debate.confidence * 0.2
        
        # Decisive winner = higher conviction
        if debate.winner in ("bull", "bear"):
            conviction += 0.1
        else:
            conviction -= 0.1  # Inconclusive = lower conviction
    
    if risk:
        # High risk reduces conviction
        if risk.risk_score > 60:
            conviction -= 0.15
        elif risk.risk_score < 30:
            conviction += 0.1
    
    # Data quality affects conviction
    if tech and tech.confidence > 0.7:
        conviction += 0.05
    if fund and fund.confidence > 0.7:
        conviction += 0.05
    
    # Consistency between tech and fund
    if tech and fund:
        diff = abs(tech.score - fund.score)
        if diff < 15:
            conviction += 0.05  # Consistent signals
        elif diff > 35:
            conviction -= 0.1  # Conflicting signals
    
    return max(0.1, min(0.95, conviction))


def _calculate_entry_price(tech, rating) -> float:
    """Calculate suggested entry price."""
    if not tech or not tech.current_price:
        return None
    
    price = tech.current_price
    
    if rating in (Rating.BUY, Rating.OVERWEIGHT):
        # Entry at or near current price for buy ratings
        return round(price * 0.98, 2)  # 2% below current
    elif rating == Rating.HOLD:
        return round(price * 0.94, 2)  # 6% below current
    else:
        return None  # No entry for sell ratings


def _calculate_target_price(tech, rating, risk) -> float:
    """Calculate target price."""
    if not tech or not tech.current_price:
        return None
    
    price = tech.current_price
    atr = tech.atr or price * 0.02
    
    # Multipliers by rating
    multipliers = {
        Rating.BUY: 3.0, Rating.OVERWEIGHT: 2.5,
        Rating.HOLD: 2.0, Rating.UNDERWEIGHT: 1.5,
        Rating.SELL: 1.0
    }
    multiplier = multipliers.get(rating, 2.0)
    
    # Reduce target for high risk
    if risk and risk.risk_score > 60:
        multiplier *= 0.8
    
    return round(price + atr * multiplier, 2)

def _calculate_risk_reward(entry, target, stop) -> float:
    """Calculate risk/reward ratio."""
    if not all([entry, target, stop]):
        return None
    
    if entry <= stop:
        return None
    
    risk = entry - stop
    reward = target - entry
    
    if risk <= 0:
        return None
    
    return round(reward / risk, 2)


def _extract_catalysts(bull) -> list:
    """Extract key catalysts from bull thesis."""
    if bull and bull.supporting_evidence:
        return bull.supporting_evidence[:4]
    return []


def _generate_llm_summary(symbol, company_name, rating, score,
                         debate, bull, bear, risk) -> str:
    """Generate a brief LLM summary of the recommendation."""
    
    # Build a simple prompt
    bull_thesis = bull.thesis[:200] if bull else "N/A"
    bear_thesis = bear.thesis[:200] if bear else "N/A"
    risk_level = f"{risk.risk_score:.0f}/100" if risk else "N/A"
    debate_winner = debate.winner if debate else "N/A"
    
    prompt = f"""Summarize this stock recommendation in 2-3 sentences.

STOCK: {symbol} ({company_name})
RATING: {rating.value.upper()} (Score: {score:.0f}/100)
DEBATE WINNER: {debate_winner}
RISK LEVEL: {risk_level}

BULL VIEW: {bull_thesis}
BEAR VIEW: {bear_thesis}

Respond with ONLY this JSON:
{{"summary": "2-3 sentence recommendation summary"}}"""
    
    result = extract_json(call_llm(prompt, temperature=0.3, num_predict=300))
    
    if result and result.get("summary"):
        return str(result["summary"])[:500]
    
    # Fallback summary
    return (f"{symbol} rated {rating.value.upper()} with score {score:.0f}/100. "
            f"Debate winner: {debate_winner}. Risk level: {risk_level}.")


def _assess_data_quality(tech, fund, debate) -> float:
    """Assess overall data quality."""
    quality = 0.5
    
    if tech and tech.confidence > 0.6:
        quality += 0.15
    if fund and fund.confidence > 0.6:
        quality += 0.15
    if debate and debate.confidence > 0.6:
        quality += 0.1
    
    return round(min(1.0, quality), 2)


def _error_recommendation(symbol, company_name) -> FinalRecommendation:
    """Error case recommendation."""
    return FinalRecommendation(
        symbol=symbol,
        company_name=company_name,
        rating=Rating.HOLD,
        overall_score=50.0,
        conviction=0.2,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        debate_summary=f"Analysis error occurred for {symbol}. Defaulting to HOLD.",
        key_risks=["Analysis pipeline error — treat with caution"],
    )