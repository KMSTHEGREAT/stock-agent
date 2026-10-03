"""
Recommendation generator that computes entry/exit prices and risk levels.
All price levels are computed deterministically from technical metrics.
"""
from typing import Optional, Dict
from v1.schemas.stock_analysis import (
    StockAnalysis, TechnicalMetrics, Recommendation
)
from v1.utils.logger import setup_logger
from config.settings import RISK

logger = setup_logger(__name__)


class RecommendationGenerator:
    """Generates actionable recommendations with entry/exit/stop-loss levels."""
    
    def generate(
        self,
        symbol: str,
        company_name: str,
        technical_metrics: TechnicalMetrics,
        recommendation: Recommendation,
        overall_score: int
    ) -> Dict:
        """
        Generate entry, target, and stop-loss prices.
        
        Returns:
            Dictionary with:
            - entry_price
            - target_price (conservative and aggressive)
            - stop_loss
            - risk_reward_ratio
            - position_size_suggestion
        """
        current_price = technical_metrics.current_price
        atr = technical_metrics.atr or (current_price * 0.02)  # Default 2% ATR
        support = technical_metrics.support_level
        resistance = technical_metrics.resistance_level
        
        result = {
            "symbol": symbol,
            "company_name": company_name,
            "current_price": current_price,
            "recommendation": recommendation.value,
            "overall_score": overall_score,
        }
        
        if recommendation in [Recommendation.STRONG_BUY, Recommendation.BUY]:
            result.update(self._generate_buy_levels(
                current_price, atr, support, resistance, overall_score
            ))
        elif recommendation == Recommendation.HOLD:
            result.update(self._generate_hold_levels(
                current_price, atr, support, resistance
            ))
        else:  # SELL or STRONG_SELL
            result.update(self._generate_sell_levels(
                current_price, atr, resistance
            ))
        
        # Calculate risk-reward ratio
        if result.get("entry_price") and result.get("stop_loss") and result.get("target_price"):
            entry = result["entry_price"]
            stop = result["stop_loss"]
            target = result["target_price"]
            
            risk = entry - stop
            reward = target - entry
            
            if risk > 0 and reward > 0:
                result["risk_reward_ratio"] = round(reward / risk, 2)
            else:
                result["risk_reward_ratio"] = None
        
        return result
    
    def _generate_buy_levels(
        self, current_price: float, atr: float,
        support: Optional[float], resistance: Optional[float],
        score: int
    ) -> Dict:
        """Generate buy recommendation levels."""
        levels = {}
        
        # Entry price: depends on score
        if score >= 80:  # Strong buy - can enter at market
            levels["entry_price"] = round(current_price, 2)
            levels["entry_strategy"] = "Enter at current market price (strong signal)"
        elif score >= 70:  # Buy - wait for slight pullback
            # Entry at support or 1 ATR below current
            if support and support < current_price:
                entry = max(support, current_price - atr * 0.5)
            else:
                entry = current_price - atr * 0.5
            levels["entry_price"] = round(entry, 2)
            levels["entry_strategy"] = f"Buy on pullback to ₹{entry:.2f}"
        
        # Stop loss: below support or 1.5 ATR below entry
        entry = levels.get("entry_price", current_price)
        if support and support < entry:
            stop = min(support - atr * 0.5, entry - atr * RISK["stop_loss_atr_multiplier"])
        else:
            stop = entry - atr * RISK["stop_loss_atr_multiplier"]
        levels["stop_loss"] = round(stop, 2)
        
        # Target price: based on resistance or ATR multiples
        if resistance and resistance > entry:
            # Conservative: resistance level
            levels["target_price_conservative"] = round(resistance, 2)
            # Aggressive: beyond resistance
            levels["target_price_aggressive"] = round(
                resistance + atr, 2
            )
        else:
            # No clear resistance - use ATR multiples
            levels["target_price_conservative"] = round(
                entry + atr * RISK["target_atr_multiplier_conservative"], 2
            )
            levels["target_price_aggressive"] = round(
                entry + atr * RISK["target_atr_multiplier_aggressive"], 2
            )
        
        # Use conservative as primary target
        levels["target_price"] = levels["target_price_conservative"]
        
        return levels
    
    def _generate_hold_levels(
        self, current_price: float, atr: float,
        support: Optional[float], resistance: Optional[float]
    ) -> Dict:
        """Generate hold recommendation levels."""
        levels = {
            "entry_price": None,
            "entry_strategy": "Hold - wait for better entry point",
            "stop_loss": None,
            "target_price": None
        }
        
        # Suggest monitoring levels
        if support:
            levels["watch_buy_level"] = round(support, 2)
            levels["entry_strategy"] += f" Watch for buy signal near ₹{support:.2f}"
        
        if resistance:
            levels["watch_sell_level"] = round(resistance, 2)
        
        return levels
    
    def _generate_sell_levels(
        self, current_price: float, atr: float,
        resistance: Optional[float]
    ) -> Dict:
        """Generate sell recommendation levels."""
        levels = {
            "entry_price": None,
            "entry_strategy": "Consider exiting position",
            "stop_loss": None,
        }
        
        # Exit target: current price or resistance
        if resistance and resistance > current_price:
            levels["exit_target"] = round(resistance, 2)
            levels["target_price"] = round(resistance, 2)
            levels["entry_strategy"] = f"Exit near ₹{resistance:.2f} (resistance)"
        else:
            levels["exit_target"] = round(current_price, 2)
            levels["target_price"] = round(current_price, 2)
            levels["entry_strategy"] = "Exit at current price"
        
        return levels