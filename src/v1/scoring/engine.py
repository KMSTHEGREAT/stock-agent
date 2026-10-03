"""
Combined scoring engine that produces the final 0-100 score.
Integrates technical, fundamental, news, and macro scores.
"""
from typing import Dict, Optional
from datetime import datetime

from src.v1.schemas.stock_analysis import (
    StockAnalysis, TechnicalMetrics, FundamentalMetrics,
    Recommendation, RiskFlags
)
from src.v1.analysis.technical_scorer import TechnicalScorer
from src.v1.analysis.fundamental_scorer import FundamentalScorer
from src.v1.utils.logger import setup_logger
from config.settings import SCORING, RISK

logger = setup_logger(__name__)


class ScoringEngine:
    """
    Produces the final 0-100 score for a stock.
    Technical: 30%, Fundamental: 35%, News: 20%, Macro: 15%
    """
    
    def __init__(self):
        self.technical_scorer = TechnicalScorer()
        self.fundamental_scorer = FundamentalScorer()
    
    def compute_score(
        self,
        technical_metrics: TechnicalMetrics,
        fundamental_metrics: FundamentalMetrics,
        news_score: Optional[float] = None,
        macro_score: Optional[float] = None
    ) -> Dict:
        """
        Compute the final composite score.
        
        Returns:
            Dictionary with all scores:
            - technical_score, fundamental_score, news_score, macro_score
            - overall_score
            - recommendation
        """
        # Compute sub-scores
        tech_scores = self.technical_scorer.score(technical_metrics)
        fund_scores = self.fundamental_scorer.score(fundamental_metrics)
        
        # News score (if not provided, default to neutral 50)
        if news_score is None:
            news_score = 50.0
            logger.debug("No news score provided, using neutral 50")
        
        # Macro score (if not provided, default to neutral 50)
        if macro_score is None:
            macro_score = 50.0
            logger.debug("No macro score provided, using neutral 50")
        
        # Compute weighted overall score
        weights = SCORING["weights"]
        overall = (
            tech_scores["total"] * weights["technical"] +
            fund_scores["total"] * weights["fundamental"] +
            news_score * weights["news_sentiment"] +
            macro_score * weights["macro_context"]
        )
        
        overall = round(overall, 1)
        
        # Determine recommendation
        recommendation = self._get_recommendation(overall, tech_scores, fund_scores)
        
        result = {
            "technical_score": round(tech_scores["total"], 1),
            "fundamental_score": round(fund_scores["total"], 1),
            "news_score": round(news_score, 1),
            "macro_score": round(macro_score, 1),
            "overall_score": overall,
            "recommendation": recommendation,
            "technical_sub_scores": tech_scores,
            "fundamental_sub_scores": fund_scores,
        }
        
        logger.info(f"Final score computed: {overall}/100 "
                   f"({recommendation.value})")
        
        return result
    
    def _get_recommendation(
        self, 
        overall_score: float,
        tech_scores: Dict,
        fund_scores: Dict
    ) -> Recommendation:
        """Determine recommendation based on score and sub-scores."""
        thresholds = SCORING["recommendation_thresholds"]
        
        # Check for red flags that override the score
        fund_total = fund_scores.get("total", 50)
        tech_total = tech_scores.get("total", 50)
        
        # Strong red flags downgrade recommendation
        if fund_total < 30:  # Very poor fundamentals
            if overall_score > thresholds["hold"]:
                overall_score = thresholds["hold"]
        
        # Strong signals upgrade
        if fund_total > 80 and tech_total > 70:
            # Both strong - upgrade to strong buy if score is high
            if overall_score > thresholds["buy"]:
                return Recommendation.STRONG_BUY
        
        # Standard thresholds
        if overall_score >= thresholds["strong_buy"]:
            return Recommendation.STRONG_BUY
        elif overall_score >= thresholds["buy"]:
            return Recommendation.BUY
        elif overall_score >= thresholds["hold"]:
            return Recommendation.HOLD
        elif overall_score >= thresholds["sell"]:
            return Recommendation.SELL
        else:
            return Recommendation.STRONG_SELL