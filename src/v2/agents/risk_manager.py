"""
Risk Manager Agent for V2.
Assesses risk levels, determines position sizing, calculates stop-loss.
Uses deterministic rules based on technical and fundamental data.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import RiskAssessment

logger = logging.getLogger(__name__)


def risk_manager_node(state: dict) -> dict:
    """
    LangGraph node: Risk Manager.
    
    Assesses risk from multiple dimensions and produces
    actionable risk parameters (stop-loss, position size).
    Deterministic — no LLM involved.
    """
    symbol = state["symbol"]
    
    logger.info(f"[Risk Manager] Assessing risk for {symbol}")
    
    try:
        tech = state.get("technical_output")
        fund = state.get("fundamental_output")
        debate = state.get("debate_result")
        news = state.get("news_output")
        
        # Compute risk components
        volatility_risk = _assess_volatility_risk(tech)
        liquidity_risk = _assess_liquidity_risk(fund)
        promoter_risk = _assess_promoter_risk(fund)
        regulatory_risk = _assess_regulatory_risk(news)
        
        # Calculate overall risk score (0=low risk, 100=high risk)
        risk_score = _calculate_risk_score(
            volatility_risk, liquidity_risk, promoter_risk, regulatory_risk,
            tech, fund, debate
        )
        
        # Determine position sizing based on risk
        max_position_pct = _position_size_from_risk(risk_score)
        
        # Calculate stop-loss from technical data
        stop_loss = _calculate_stop_loss(tech)
        
        # Collect risk warnings
        warnings = _collect_warnings(
            volatility_risk, liquidity_risk, promoter_risk, regulatory_risk,
            fund, tech, debate
        )
        
        assessment = RiskAssessment(
            risk_score=risk_score,
            volatility_risk=volatility_risk,
            liquidity_risk=liquidity_risk,
            promoter_risk=promoter_risk,
            regulatory_risk=regulatory_risk,
            max_position_size_pct=max_position_pct,
            suggested_stop_loss=stop_loss,
            risk_warnings=warnings,
        )
        
        logger.info(f"[Risk Manager] {symbol}: risk={risk_score:.0f}/100, "
                   f"max_position={max_position_pct:.1f}%, stop_loss={stop_loss}")
        
        return {"risk_assessment": assessment}
        
    except Exception as e:
        logger.error(f"[Risk Manager] Error: {e}")
        return {
            "risk_assessment": _default_risk(),
            "errors": [f"Risk manager error: {str(e)}"],
        }


def _assess_volatility_risk(tech) -> str:
    """Assess volatility risk from technical data."""
    if not tech:
        return "medium"
    
    if tech.atr and tech.current_price:
        atr_pct = (tech.atr / tech.current_price) * 100
        
        if atr_pct > 5:
            return "high"
        elif atr_pct > 3:
            return "medium"
        else:
            return "low"
    
    # Use price change as proxy
    if tech.price_change_1m is not None:
        if abs(tech.price_change_1m) > 15:
            return "high"
        elif abs(tech.price_change_1m) > 8:
            return "medium"
    
    return "low"


def _assess_liquidity_risk(fund) -> str:
    """Assess liquidity risk from market cap."""
    if not fund or fund.market_cap_cr is None:
        return "medium"
    
    if fund.market_cap_cr > 50000:  # > ₹50,000 Cr
        return "low"
    elif fund.market_cap_cr > 10000:  # > ₹10,000 Cr
        return "medium"
    else:
        return "high"


def _assess_promoter_risk(fund) -> str:
    """Assess promoter-related risk."""
    if not fund:
        return "none"
    
    risk = "none"
    
    # Pledged shares
    if fund.pledged_pct is not None:
        if fund.pledged_pct > 30:
            risk = "high"
        elif fund.pledged_pct > 10:
            risk = "medium" if risk != "high" else risk
    
    # Low promoter holding
    if fund.promoter_holding is not None:
        if fund.promoter_holding < 15:
            risk = "medium" if risk == "none" else risk
    
    return risk


def _assess_regulatory_risk(news) -> str:
    """Assess regulatory risk from news."""
    if not news:
        return "none"
    
    # Check for regulatory-related keywords in news
    regulatory_keywords = [
        'regulatory', 'sebi', 'probe', 'investigation', 'penalty',
        'fine', 'fraud', 'scam', 'compliance', 'ban'
    ]
    
    for item in news.news_items:
        text = (item.title + " " + item.summary).lower()
        if any(keyword in text for keyword in regulatory_keywords):
            return "medium"
    
    return "none"


def _calculate_risk_score(vol_risk, liq_risk, prom_risk, reg_risk,
                         tech, fund, debate) -> float:
    """Calculate overall risk score (0=low, 100=high)."""
    score = 20  # Base risk
    
    # Volatility contribution (0-30)
    vol_scores = {"low": 5, "medium": 15, "high": 30}
    score += vol_scores.get(vol_risk, 15)
    
    # Liquidity contribution (0-20)
    liq_scores = {"low": 2, "medium": 10, "high": 20}
    score += liq_scores.get(liq_risk, 10)
    
    # Promoter risk contribution (0-25)
    prom_scores = {"none": 0, "medium": 15, "high": 25}
    score += prom_scores.get(prom_risk, 0)
    
    # Regulatory risk contribution (0-15)
    reg_scores = {"none": 0, "medium": 10, "high": 15}
    score += reg_scores.get(reg_risk, 0)
    
    # Adjust based on technical weakness
    if tech and tech.score < 35:
        score += 10
    elif tech and tech.score > 65:
        score -= 5
    
    # Adjust based on fundamental weakness
    if fund and fund.score < 40:
        score += 10
    elif fund and fund.score > 70:
        score -= 5
    
    # Adjust based on debate outcome
    if debate:
        if debate.winner == "bear":
            score += 8
        elif debate.winner == "bull":
            score -= 5
    
    return max(0, min(100, round(score, 1)))


def _position_size_from_risk(risk_score: float) -> float:
    """Determine maximum position size based on risk."""
    if risk_score < 30:
        return 8.0   # Low risk — can take larger position
    elif risk_score < 50:
        return 5.0   # Medium risk — standard position
    elif risk_score < 70:
        return 3.0   # High risk — reduced position
    else:
        return 1.5   # Very high risk — minimal position


def _calculate_stop_loss(tech):
    """Calculate stop loss BELOW current price."""
    if not tech or not tech.current_price:
        return None
    
    price = tech.current_price
    
    if tech.atr:
        # 1.5x ATR below current price
        stop = price - (tech.atr * 1.5)
    elif tech.support_level and tech.support_level < price:
        # 2% below support
        stop = tech.support_level * 0.98
    else:
        # 7% below current price
        stop = price * 0.93
    
    # Ensure stop is always below current price
    return round(min(stop, price * 0.95), 2)

def _collect_warnings(vol_risk, liq_risk, prom_risk, reg_risk,
                     fund, tech, debate) -> list:
    """Collect human-readable risk warnings."""
    warnings = []
    
    if vol_risk == "high":
        warnings.append("⚠ High volatility — significant price swings expected")
    if liq_risk == "high":
        warnings.append("⚠ Low liquidity — difficult to exit position quickly")
    
    if fund:
        if fund.pledged_pct and fund.pledged_pct > 20:
            warnings.append(f"⚠ Promoter pledging at {fund.pledged_pct:.0f}% — major governance risk")
        if fund.debt_to_equity and fund.debt_to_equity > 2:
            warnings.append(f"⚠ High debt-to-equity ratio ({fund.debt_to_equity:.2f})")
    
    if tech:
        if tech.trend == "down" and tech.rsi and tech.rsi < 30:
            warnings.append("⚠ Downtrend with oversold RSI — falling knife risk")
    
    if debate and debate.winner == "bear":
        warnings.append("⚠ Bear thesis won the debate — exercise caution")
    
    if not warnings:
        warnings.append("No significant risk warnings")
    
    return warnings[:6]


def _default_risk() -> RiskAssessment:
    """Default risk assessment for error cases."""
    return RiskAssessment(
        risk_score=50.0,
        volatility_risk="medium",
        liquidity_risk="medium",
        promoter_risk="none",
        regulatory_risk="none",
        max_position_size_pct=3.0,
        suggested_stop_loss=None,
        risk_warnings=["Risk assessment defaulted due to error"],
    )