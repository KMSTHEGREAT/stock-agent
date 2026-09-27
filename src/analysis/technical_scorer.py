"""
Technical analysis scoring engine.
Converts technical metrics into a 0-100 score deterministically.
"""
from typing import Dict
from src.schemas.stock_analysis import TechnicalMetrics, TrendDirection
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TechnicalScorer:
    """Scores technical analysis on a 0-100 scale."""
    
    def score(self, metrics: TechnicalMetrics) -> Dict[str, float]:
        """
        Calculate technical score based on multiple sub-factors.
        
        Returns:
            Dictionary with total score and sub-scores
        """
        scores = {}
        
        # 1. Trend Score (0-100) - Weight: 40% of technical
        scores["trend"] = self._score_trend(metrics)
        
        # 2. Momentum Score (0-100) - Weight: 30% of technical
        scores["momentum"] = self._score_momentum(metrics)
        
        # 3. Volatility Score (0-100) - Weight: 15% of technical
        scores["volatility"] = self._score_volatility(metrics)
        
        # 4. Volume Score (0-100) - Weight: 15% of technical
        scores["volume"] = self._score_volume(metrics)
        
        # Weighted total
        total = (
            scores["trend"] * 0.40 +
            scores["momentum"] * 0.30 +
            scores["volatility"] * 0.15 +
            scores["volume"] * 0.15
        )
        
        # Penalty for circuit limit proximity
        if metrics.near_circuit:
            total -= 10
            logger.warning(f"{metrics.symbol}: Near circuit limit, score penalized")
        
        scores["total"] = max(0, min(100, round(total, 1)))
        
        logger.info(f"Technical score for {metrics.symbol}: {scores['total']}/100")
        logger.debug(f"Sub-scores: {scores}")
        
        return scores
    
    def _score_trend(self, m: TechnicalMetrics) -> float:
        """Score trend direction and strength."""
        score = 50  # Neutral base
        
        # Overall trend direction
        if m.trend == TrendDirection.UP:
            score += 25
        elif m.trend == TrendDirection.DOWN:
            score -= 25
        
        # Price relative to moving averages
        if m.sma_20 and m.current_price > m.sma_20:
            score += 5
        elif m.sma_20 and m.current_price < m.sma_20:
            score -= 5
        
        if m.sma_50 and m.current_price > m.sma_50:
            score += 5
        elif m.sma_50 and m.current_price < m.sma_50:
            score -= 5
        
        # Long-term trend (SMA 200)
        if m.sma_200 and m.current_price > m.sma_200:
            score += 10
        elif m.sma_200 and m.current_price < m.sma_200:
            score -= 10
        
        # Price change momentum
        if m.price_change_pct_1m is not None:
            if m.price_change_pct_1m > 10:
                score += 5
            elif m.price_change_pct_1m < -10:
                score -= 5
        
        return max(0, min(100, score))
    
    def _score_momentum(self, m: TechnicalMetrics) -> float:
        """Score momentum indicators (RSI, MACD, Stochastic)."""
        score = 50  # Neutral base
        
        # RSI scoring
        if m.rsi is not None:
            if 40 <= m.rsi <= 60:
                score += 10  # Neutral zone - stable
            elif 60 < m.rsi <= 70:
                score += 20  # Bullish but not overbought
            elif 30 <= m.rsi < 40:
                score -= 10  # Bearish but not oversold
            elif m.rsi > 70:
                score -= 15  # Overbought - potential reversal
            elif m.rsi < 30:
                score += 5   # Oversold - potential bounce
            elif m.rsi < 20:
                score += 10  # Deeply oversold - strong bounce potential
        
        # MACD scoring
        if m.macd is not None and m.macd_signal is not None:
            if m.macd > m.macd_signal:
                score += 15  # Bullish crossover
            else:
                score -= 10  # Bearish crossover
            
            if m.macd_histogram is not None:
                if m.macd_histogram > 0 and m.macd > 0:
                    score += 5  # Strong bullish
                elif m.macd_histogram < 0 and m.macd < 0:
                    score -= 5  # Strong bearish
        
        # Bollinger Bands position
        if m.bb_position is not None:
            if m.bb_position > 0.8:
                score -= 10  # Near upper band - extended
            elif m.bb_position < 0.2:
                score += 10  # Near lower band - potential bounce
            elif 0.4 <= m.bb_position <= 0.6:
                score += 5   # Middle zone - stable
        
        return max(0, min(100, score))
    
    def _score_volatility(self, m: TechnicalMetrics) -> float:
        """Score based on volatility (lower volatility = higher score)."""
        score = 50  # Neutral base
        
        if m.atr is not None and m.current_price > 0:
            # ATR as percentage of price
            atr_pct = (m.atr / m.current_price) * 100
            
            if atr_pct < 2:
                score += 30  # Very low volatility
            elif atr_pct < 3:
                score += 20  # Low volatility
            elif atr_pct < 5:
                score += 10  # Moderate volatility
            elif atr_pct < 7:
                score -= 5   # High volatility
            else:
                score -= 15  # Very high volatility
        
        # Price change stability
        if m.price_change_pct_1w is not None:
            if abs(m.price_change_pct_1w) < 3:
                score += 10  # Stable weekly movement
            elif abs(m.price_change_pct_1w) > 10:
                score -= 10  # High weekly volatility
        
        return max(0, min(100, score))
    
    def _score_volume(self, m: TechnicalMetrics) -> float:
        """Score based on volume patterns."""
        score = 50  # Neutral base
        
        if m.volume_ratio is not None:
            if m.volume_ratio > 2.0:
                score += 20  # Very high volume - strong interest
            elif m.volume_ratio > 1.5:
                score += 15  # High volume
            elif m.volume_ratio > 1.2:
                score += 10  # Above average
            elif m.volume_ratio > 0.8:
                score += 5   # Normal
            elif m.volume_ratio < 0.5:
                score -= 15  # Very low volume - lack of interest
            elif m.volume_ratio < 0.8:
                score -= 5   # Below average
        
        return max(0, min(100, score))