"""
Fundamental analysis scoring engine.
Scores based on valuation, growth, profitability, and India-specific factors.
"""
from typing import Dict
from src.v1.schemas.stock_analysis import FundamentalMetrics, MarketCapCategory
from src.v1.utils.logger import setup_logger

logger = setup_logger(__name__)


class FundamentalScorer:
    """Scores fundamental analysis on a 0-100 scale with India adjustments."""
    
    # Sector-average PE ratios for context (approximate, as of 2025)
    SECTOR_AVG_PE = {
        'IT': 25, 'Banking': 15, 'Pharma': 30, 'FMCG': 45,
        'Auto': 20, 'Metals': 12, 'Energy': 15, 'Realty': 25,
        'Infrastructure': 18, 'PSU Bank': 10, 'Media': 20,
        'Healthcare': 28, 'Consumer Durables': 35, 'Oil & Gas': 15,
    }
    
    def score(self, metrics: FundamentalMetrics, 
              sector: str = None) -> Dict[str, float]:
        """
        Calculate fundamental score.
        
        Args:
            metrics: FundamentalMetrics object
            sector: Sector name for relative valuation
        
        Returns:
            Dictionary with total score and sub-scores
        """
        scores = {}
        
        # 1. Valuation Score (0-100) - Weight: 30%
        scores["valuation"] = self._score_valuation(metrics, sector)
        
        # 2. Growth Score (0-100) - Weight: 20%
        scores["growth"] = self._score_growth(metrics)
        
        # 3. Profitability Score (0-100) - Weight: 25%
        scores["profitability"] = self._score_profitability(metrics)
        
        # 4. Leverage Score (0-100) - Weight: 10%
        scores["leverage"] = self._score_leverage(metrics)
        
        # 5. Promoter/Governance Score (0-100) - Weight: 15%
        # This is India-specific and critical
        scores["promoter_governance"] = self._score_promoter_governance(metrics)
        
        # Weighted total
        total = (
            scores["valuation"] * 0.30 +
            scores["growth"] * 0.20 +
            scores["profitability"] * 0.25 +
            scores["leverage"] * 0.10 +
            scores["promoter_governance"] * 0.15
        )
        
        scores["total"] = max(0, min(100, round(total, 1)))
        
        logger.info(f"Fundamental score for {metrics.symbol}: {scores['total']}/100")
        logger.debug(f"Sub-scores: {scores}")
        
        return scores
    
    def _score_valuation(self, m: FundamentalMetrics, sector: str = None) -> float:
        """Score valuation metrics relative to sector."""
        score = 50  # Neutral base
        
        # Get sector average PE if available
        sector_pe = self.SECTOR_AVG_PE.get(sector, 20)  # Default 20
        
        # PE Ratio scoring (relative to sector)
        if m.pe_ratio is not None and m.pe_ratio > 0:
            if m.pe_ratio < sector_pe * 0.5:
                score += 30  # Very cheap relative to sector
            elif m.pe_ratio < sector_pe * 0.7:
                score += 20  # Cheap
            elif m.pe_ratio < sector_pe:
                score += 10  # Below average
            elif m.pe_ratio < sector_pe * 1.3:
                score += 0   # Fair value
            elif m.pe_ratio < sector_pe * 1.5:
                score -= 10  # Slightly expensive
            elif m.pe_ratio < sector_pe * 2:
                score -= 20  # Expensive
            else:
                score -= 30  # Very expensive
            
            # Negative PE (loss-making) is very bad
            if m.pe_ratio < 0:
                score -= 40
        else:
            score -= 10  # No PE data available
        
        # PB Ratio scoring
        if m.pb_ratio is not None and m.pb_ratio > 0:
            if m.pb_ratio < 1:
                score += 15  # Below book value
            elif m.pb_ratio < 2:
                score += 10
            elif m.pb_ratio < 5:
                score += 5
            elif m.pb_ratio > 10:
                score -= 10  # Very high PB
        
        # EV/EBITDA scoring
        if m.ev_ebitda is not None and m.ev_ebitda > 0:
            if m.ev_ebitda < 8:
                score += 10
            elif m.ev_ebitda < 12:
                score += 5
            elif m.ev_ebitda > 20:
                score -= 10
        
        # Market cap bonus (larger = more stable)
        if m.market_cap_category == MarketCapCategory.LARGE_CAP:
            score += 5
        elif m.market_cap_category == MarketCapCategory.SMALL_CAP:
            score -= 5  # Small caps are riskier
        
        return max(0, min(100, score))
    
    def _score_growth(self, m: FundamentalMetrics) -> float:
        """Score growth metrics."""
        score = 50  # Neutral base
        
        # Revenue growth (1 year)
        if m.revenue_growth_1y is not None:
            if m.revenue_growth_1y > 25:
                score += 30
            elif m.revenue_growth_1y > 15:
                score += 20
            elif m.revenue_growth_1y > 10:
                score += 15
            elif m.revenue_growth_1y > 5:
                score += 10
            elif m.revenue_growth_1y > 0:
                score += 5
            elif m.revenue_growth_1y > -10:
                score -= 10
            else:
                score -= 20
        
        # Profit growth
        if m.profit_growth_1y is not None:
            if m.profit_growth_1y > 25:
                score += 20
            elif m.profit_growth_1y > 15:
                score += 15
            elif m.profit_growth_1y > 5:
                score += 10
            elif m.profit_growth_1y < -20:
                score -= 20
            elif m.profit_growth_1y < 0:
                score -= 10
        
        return max(0, min(100, score))
    
    def _score_profitability(self, m: FundamentalMetrics) -> float:
        """Score profitability metrics."""
        score = 50  # Neutral base
        
        # ROCE (most important in Indian context)
        if m.roce is not None:
            if m.roce > 25:
                score += 30  # Excellent capital efficiency
            elif m.roce > 20:
                score += 25
            elif m.roce > 15:
                score += 20
            elif m.roce > 10:
                score += 10
            elif m.roce > 5:
                score += 5
            elif m.roce < 0:
                score -= 30  # Negative ROCE is very bad
            else:
                score -= 10
        
        # ROE
        if m.roe is not None:
            if m.roe > 20:
                score += 15
            elif m.roe > 15:
                score += 10
            elif m.roe > 10:
                score += 5
            elif m.roe < 5:
                score -= 10
        
        # Net margin
        if m.net_margin is not None:
            if m.net_margin > 15:
                score += 10
            elif m.net_margin > 10:
                score += 5
            elif m.net_margin > 5:
                score += 2
            elif m.net_margin < 0:
                score -= 15
        
        # Operating margin
        if m.operating_margin is not None:
            if m.operating_margin > 20:
                score += 5
            elif m.operating_margin < 5:
                score -= 5
        
        return max(0, min(100, score))
    
    def _score_leverage(self, m: FundamentalMetrics) -> float:
        """Score financial leverage."""
        score = 50  # Neutral base
        
        # Debt to Equity
        if m.debt_to_equity is not None:
            if m.debt_to_equity < 0.1:
                score += 30  # Almost debt-free
            elif m.debt_to_equity < 0.3:
                score += 20  # Low debt
            elif m.debt_to_equity < 0.5:
                score += 10  # Manageable
            elif m.debt_to_equity < 1.0:
                score += 0   # Normal
            elif m.debt_to_equity < 2.0:
                score -= 15  # High debt
            else:
                score -= 25  # Very high debt
        
        # Current ratio (liquidity)
        if m.current_ratio is not None:
            if m.current_ratio > 2:
                score += 10  # Very liquid
            elif m.current_ratio > 1.5:
                score += 5
            elif m.current_ratio < 1:
                score -= 10  # Liquidity issues
        
        return max(0, min(100, score))
    
    def _score_promoter_governance(self, m: FundamentalMetrics) -> float:
        """
        Score promoter holding and governance (India-specific).
        This is CRITICAL for Indian markets.
        """
        score = 50  # Neutral base
        
        # Promoter holding (higher = more skin in the game)
        if m.promoter_holding is not None:
            if m.promoter_holding > 60:
                score += 30  # Very high promoter holding
            elif m.promoter_holding > 50:
                score += 25  # High
            elif m.promoter_holding > 40:
                score += 20
            elif m.promoter_holding > 30:
                score += 10
            elif m.promoter_holding > 20:
                score += 0   # Low
            elif m.promoter_holding > 10:
                score -= 10  # Very low
            else:
                score -= 20  # Minimal promoter interest
        
        # CRITICAL: Pledged shares (major red flag)
        if m.pledged_pct is not None:
            if m.pledged_pct == 0:
                score += 20  # No pledging - excellent
            elif m.pledged_pct < 5:
                score += 10  # Very low
            elif m.pledged_pct < 10:
                score += 0   # Acceptable
            elif m.pledged_pct < 20:
                score -= 15  # Warning
            elif m.pledged_pct < 50:
                score -= 30  # High risk
            else:
                score -= 40  # Very high risk - avoid
        
        # FII/DII holding (institutional confidence)
        if m.fii_holding is not None:
            if m.fii_holding > 20:
                score += 10  # High institutional confidence
            elif m.fii_holding > 10:
                score += 5
            elif m.fii_holding < 2:
                score -= 5   # Low institutional interest
        
        if m.dii_holding is not None:
            if m.dii_holding > 15:
                score += 5
        
        return max(0, min(100, score))