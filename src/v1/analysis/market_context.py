"""
Market context analyzer for Indian markets.
Fetches Nifty/Sensex trends and computes market regime.
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

from src.v1.schemas.stock_analysis import MarketContext, TrendDirection
from src.v1.data.cache import DataCache
from src.v1.utils.logger import setup_logger
from config.settings import MARKET

logger = setup_logger(__name__)


class MarketContextAnalyzer:
    """Analyzes broader Indian market context."""
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()
    
    def get_market_context(self) -> MarketContext:
        """Get current market context for Indian markets."""
        cache_key = "market_context"
        cached = self.cache.get(cache_key, max_age_hours=6)
        if cached:
            return MarketContext(**cached)
        
        context = {
            "nifty_trend": TrendDirection.SIDEWAYS,
            "sensex_trend": TrendDirection.SIDEWAYS,
            "nifty_change_pct": None,
            "vix_level": None,
            "market_regime": "neutral",
            "fii_flow_cr": None,
            "dii_flow_cr": None,
        }
        
        try:
            # Nifty 50
            nifty = yf.Ticker(MARKET["indices"]["nifty_50"])
            nifty_hist = nifty.history(period="1mo")
            
            if not nifty_hist.empty:
                current = nifty_hist["Close"].iloc[-1]
                month_ago = nifty_hist["Close"].iloc[0]
                change_pct = ((current - month_ago) / month_ago) * 100
                context["nifty_change_pct"] = round(change_pct, 2)
                
                if change_pct > 3:
                    context["nifty_trend"] = TrendDirection.UP
                elif change_pct < -3:
                    context["nifty_trend"] = TrendDirection.DOWN
        
        except Exception as e:
            logger.error(f"Nifty fetch error: {e}")
        
        try:
            # Sensex
            sensex = yf.Ticker(MARKET["indices"]["sensex"])
            sensex_hist = sensex.history(period="1mo")
            
            if not sensex_hist.empty:
                current = sensex_hist["Close"].iloc[-1]
                month_ago = sensex_hist["Close"].iloc[0]
                change_pct = ((current - month_ago) / month_ago) * 100
                
                if change_pct > 3:
                    context["sensex_trend"] = TrendDirection.UP
                elif change_pct < -3:
                    context["sensex_trend"] = TrendDirection.DOWN
        
        except Exception as e:
            logger.error(f"Sensex fetch error: {e}")
        
        try:
            # India VIX
            vix = yf.Ticker("^INDIAVIX")
            vix_hist = vix.history(period="5d")
            if not vix_hist.empty:
                context["vix_level"] = round(float(vix_hist["Close"].iloc[-1]), 2)
        
        except Exception as e:
            logger.debug(f"VIX fetch error: {e}")
        
        # Determine market regime
        context["market_regime"] = self._determine_regime(context)
        
        # Cache result
        self.cache.set(cache_key, context, ttl_hours=6, data_type="json")
        
        return MarketContext(**context)
    
    def get_macro_score(self, context: MarketContext) -> float:
        """Compute macro score (0-100) based on market context."""
        score = 50  # Neutral
        
        # Nifty trend
        if context.nifty_trend == TrendDirection.UP:
            score += 15
        elif context.nifty_trend == TrendDirection.DOWN:
            score -= 15
        
        # VIX level (inverse - low VIX is good)
        if context.vix_level:
            if context.vix_level < 12:
                score += 10  # Very low volatility
            elif context.vix_level < 15:
                score += 5   # Low volatility
            elif context.vix_level > 25:
                score -= 15  # High volatility
            elif context.vix_level > 20:
                score -= 10  # Elevated volatility
        
        # Nifty monthly change
        if context.nifty_change_pct is not None:
            if context.nifty_change_pct > 5:
                score += 10
            elif context.nifty_change_pct < -5:
                score -= 10
        
        return max(0, min(100, score))
    
    def _determine_regime(self, context: Dict) -> str:
        """Determine overall market regime."""
        nifty_trend = context.get("nifty_trend", TrendDirection.SIDEWAYS)
        sensex_trend = context.get("sensex_trend", TrendDirection.SIDEWAYS)
        vix = context.get("vix_level", 15)
        
        # Bull market: both indices up, low VIX
        if nifty_trend == TrendDirection.UP and sensex_trend == TrendDirection.UP:
            if vix < 15:
                return "bull"
            else:
                return "cautiously_bullish"
        
        # Bear market: both indices down, high VIX
        if nifty_trend == TrendDirection.DOWN and sensex_trend == TrendDirection.DOWN:
            if vix > 20:
                return "bear"
            else:
                return "cautiously_bearish"
        
        # Mixed or sideways
        return "neutral"