"""
Fundamental Analyst Agent for V2.
Reuses V1's fundamental analysis but outputs structured data.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import FundamentalOutput
from src.v1.analysis.fundamental import FundamentalAnalyzer
from src.v1.analysis.fundamental_scorer import FundamentalScorer

logger = logging.getLogger(__name__)


def fundamental_analyst_node(state: dict) -> dict:
    """
    LangGraph node: Fundamental Analyst.
    
    Computes fundamental metrics and produces a FundamentalOutput.
    Deterministic — no LLM involved in calculations.
    """
    symbol = state["symbol"]
    logger.info(f"[Fundamental Analyst] Analyzing {symbol}")
    
    try:
        # Reuse V1's fundamental analyzer
        analyzer = FundamentalAnalyzer()
        scorer = FundamentalScorer()
        
        # Get metrics
        metrics = analyzer.analyze(symbol)
        
        if metrics is None:
            logger.warning(f"[Fundamental Analyst] No data for {symbol}")
            return {
                "fundamental_output": None,
                "errors": [f"Fundamental analysis failed for {symbol}"],
            }
        
        # Score it
        scores = scorer.score(metrics)
        
        # Build structured output
        output = FundamentalOutput(
            agent_name="fundamental_analyst",
            symbol=symbol,
            score=scores.get("total", 50.0),
            confidence=_calc_confidence(scores, metrics),
            key_findings=_extract_findings(metrics),
            data_sources=["screener.in", "yfinance"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            
            # Fundamental specifics
            pe_ratio=metrics.pe_ratio,
            pb_ratio=metrics.pb_ratio,
            roce=metrics.roce,
            roe=metrics.roe,
            debt_to_equity=metrics.debt_to_equity,
            promoter_holding=metrics.promoter_holding,
            pledged_pct=metrics.pledged_pct,
            market_cap_cr=metrics.market_cap_cr,
            revenue_growth=metrics.revenue_growth_1y,
        )
        
        logger.info(f"[Fundamental Analyst] {symbol}: score={output.score}, "
                   f"P/E={output.pe_ratio}, ROCE={output.roce}")
        
        return {"fundamental_output": output}
        
    except Exception as e:
        logger.error(f"[Fundamental Analyst] Error: {e}")
        return {
            "fundamental_output": None,
            "errors": [f"Fundamental analyst error: {str(e)}"],
        }


def _calc_confidence(scores: dict, metrics) -> float:
    """Calculate confidence based on data completeness."""
    confidence = 0.5  # Base
    
    # Check data availability
    data_points = [
        metrics.pe_ratio is not None,
        metrics.pb_ratio is not None,
        metrics.roce is not None,
        metrics.roe is not None,
        metrics.promoter_holding is not None,
        metrics.pledged_pct is not None,
        metrics.debt_to_equity is not None,
    ]
    
    available = sum(data_points)
    confidence += (available / len(data_points)) * 0.3
    
    # Check sub-score consistency
    sub_scores = [
        scores.get("valuation", 50),
        scores.get("growth", 50),
        scores.get("profitability", 50),
        scores.get("leverage", 50),
    ]
    
    avg = sum(sub_scores) / len(sub_scores)
    variance = sum((s - avg) ** 2 for s in sub_scores) / len(sub_scores)
    std = variance ** 0.5
    
    # Lower variance = higher confidence
    confidence += max(0, 0.2 - (std / 100))
    
    return round(min(1.0, confidence), 2)


def _extract_findings(metrics) -> list:
    """Extract key findings from fundamental analysis."""
    findings = []
    
    # Valuation
    if metrics.pe_ratio is not None:
        if metrics.pe_ratio < 15:
            findings.append(f"P/E {metrics.pe_ratio:.1f} — undervalued relative to market")
        elif metrics.pe_ratio > 30:
            findings.append(f"P/E {metrics.pe_ratio:.1f} — premium valuation")
        else:
            findings.append(f"P/E {metrics.pe_ratio:.1f} — fairly valued")
    
    # Profitability
    if metrics.roce is not None:
        if metrics.roce > 20:
            findings.append(f"ROCE {metrics.roce:.1f}% — excellent capital efficiency")
        elif metrics.roce > 15:
            findings.append(f"ROCE {metrics.roce:.1f}% — good capital efficiency")
        elif metrics.roce < 10:
            findings.append(f"ROCE {metrics.roce:.1f}% — weak capital efficiency")
    
    # India-specific
    if metrics.promoter_holding is not None:
        if metrics.promoter_holding > 50:
            findings.append(f"Promoter holding {metrics.promoter_holding:.0f}% — strong skin in the game")
        elif metrics.promoter_holding < 20:
            findings.append(f"Promoter holding {metrics.promoter_holding:.0f}% — low promoter interest")
    
    if metrics.pledged_pct is not None and metrics.pledged_pct > 10:
        findings.append(f"⚠ Promoter pledging {metrics.pledged_pct:.0f}% — significant risk")
    
    # Growth
    if metrics.revenue_growth_1y is not None:
        if metrics.revenue_growth_1y > 15:
            findings.append(f"Revenue growth {metrics.revenue_growth_1y:.1f}% — strong growth")
        elif metrics.revenue_growth_1y < 0:
            findings.append(f"Revenue growth {metrics.revenue_growth_1y:.1f}% — declining revenue")
    
    # Leverage
    if metrics.debt_to_equity is not None:
        if metrics.debt_to_equity > 1.5:
            findings.append(f"Debt/Equity {metrics.debt_to_equity:.2f} — high leverage")
        elif metrics.debt_to_equity < 0.3:
            findings.append(f"Debt/Equity {metrics.debt_to_equity:.2f} — low debt")
    
    return findings[:5]