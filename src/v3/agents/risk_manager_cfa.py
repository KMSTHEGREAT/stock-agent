"""
Advanced Risk Manager implementing CFA Level 3 methodology.

Computes:
- Value at Risk (VaR) — parametric and historical
- Conditional VaR (CVaR/Expected Shortfall)
- Sharpe Ratio and Sortino Ratio
- Maximum Drawdown and Duration
- Beta, Alpha, R² (vs Nifty)
- Kelly Criterion for position sizing
- Risk-adjusted position recommendations

All calculations are deterministic — no LLM involved.
"""
import yfinance as yf
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdvancedRiskResult:
    """CFA-grade risk metrics."""
    symbol: str
    
    # Value at Risk (daily, %)
    var_95: float               # 95% confidence, 1-day VaR
    var_99: float               # 99% confidence, 1-day VaR
    cvar_95: float              # Expected Shortfall at 95%
    
    # Risk-adjusted returns (annualized)
    sharpe_ratio: float         # (Rp - Rf) / σp
    sortino_ratio: float        # (Rp - Rf) / σ_downside
    annual_return: float        # Historical annualized return
    annual_volatility: float    # Annualized standard deviation
    
    # Drawdown analysis
    max_drawdown: float         # Maximum peak-to-trough decline (%)
    max_drawdown_duration: int  # Days from peak to trough
    current_drawdown: float     # Current drawdown from recent high
    
    # Market-relative metrics
    beta: float                 # Sensitivity to Nifty
    alpha: float                # Jensen's Alpha (annualized)
    r_squared: float            # Correlation with Nifty (R²)
    correlation: float          # Pearson correlation
    
    # Position sizing (CFA-informed)
    recommended_position_pct: float  # Suggested portfolio weight
    kelly_fraction: float           # Kelly criterion optimal fraction
    risk_score: float               # 0-100 (higher = riskier)
    risk_category: str              # "low", "moderate", "high", "very_high"
    
    @property
    def risk_summary(self) -> str:
        """Concise risk summary for display."""
        return (f"VaR(95%): {self.var_95:.1f}%, "
                f"Max DD: {self.max_drawdown:.1f}%, "
                f"Beta: {self.beta:.2f}, "
                f"Sharpe: {self.sharpe_ratio:.2f}, "
                f"Risk: {self.risk_category}")


class AdvancedRiskManager:
    """
    CFA Level 3 risk management.
    
    Implements:
    - Parametric VaR (variance-covariance method)
    - Historical CVaR (expected shortfall)
    - Sharpe and Sortino ratios
    - Maximum drawdown analysis
    - CAPM beta and Jensen's alpha
    - Kelly criterion position sizing
    """
    
    TRADING_DAYS = 252
    RISK_FREE_RATE = 0.07  # Indian 10Y G-Sec
    
    def analyze(self, symbol: str, 
                benchmark: str = "^NSEI") -> Optional[AdvancedRiskResult]:
        """
        Perform comprehensive risk analysis.
        
        Args:
            symbol: NSE stock symbol
            benchmark: Benchmark index (default: Nifty 50)
        
        Returns:
            AdvancedRiskResult with all metrics
        """
        logger.info(f"[Risk Manager CFA] Analyzing {symbol}")
        
        # Fetch price data
        stock_data = self._fetch_returns(symbol)
        market_data = self._fetch_returns(benchmark, is_index=True)
        
        if stock_data is None or len(stock_data) < 60:
            logger.warning(f"[Risk Manager CFA] Insufficient data for {symbol}")
            return None
        
        # Compute all metrics
        try:
            # VaR and CVaR
            var_95, var_99, cvar_95 = self._calculate_var(stock_data)
            
            # Sharpe and Sortino
            sharpe, sortino, annual_ret, annual_vol = self._calculate_risk_adjusted(stock_data)
            
            # Drawdown
            max_dd, dd_duration, current_dd = self._calculate_drawdown(stock_data)
            
            # Beta, Alpha, R²
            beta, alpha, r_sq, correlation = self._calculate_market_metrics(
                stock_data, market_data
            )
            
            # Position sizing
            kelly, position_pct, risk_score, risk_cat = self._calculate_position_sizing(
                annual_ret, annual_vol, var_95, max_dd, beta, sharpe
            )
            
            result = AdvancedRiskResult(
                symbol=symbol,
                var_95=round(var_95, 2),
                var_99=round(var_99, 2),
                cvar_95=round(cvar_95, 2),
                sharpe_ratio=round(sharpe, 2),
                sortino_ratio=round(sortino, 2),
                annual_return=round(annual_ret * 100, 2),
                annual_volatility=round(annual_vol * 100, 2),
                max_drawdown=round(max_dd * 100, 2),
                max_drawdown_duration=dd_duration,
                current_drawdown=round(current_dd * 100, 2),
                beta=round(beta, 2),
                alpha=round(alpha * 100, 2),
                r_squared=round(r_sq, 3),
                correlation=round(correlation, 3),
                recommended_position_pct=round(position_pct, 1),
                kelly_fraction=round(kelly, 3),
                risk_score=round(risk_score, 1),
                risk_category=risk_cat,
            )
            
            logger.info(f"[Risk Manager CFA] {symbol}: {result.risk_summary}")
            
            return result
            
        except Exception as e:
            logger.error(f"[Risk Manager CFA] Error for {symbol}: {e}")
            return None
    
    def _fetch_returns(self, symbol: str, is_index: bool = False) -> Optional[pd.Series]:
        """Fetch daily returns for a symbol."""
        try:
            if is_index:
                ticker = yf.Ticker(symbol)
            else:
                ticker = yf.Ticker(f"{symbol}.NS")
            
            hist = ticker.history(period="2y", interval="1d")
            
            if hist.empty:
                return None
            
            # Calculate daily returns
            returns = hist["Close"].pct_change().dropna()
            
            return returns
            
        except Exception:
            return None
    
    def _calculate_var(self, returns: pd.Series) -> tuple:
        """
        Calculate VaR using parametric and historical methods.
        
        Parametric VaR: μ - z × σ
        Historical CVaR: Mean of worst 5% of returns
        """
        mean = returns.mean()
        std = returns.std()
        
        # Parametric VaR (one-day)
        var_95 = mean - 1.645 * std  # 95% confidence
        var_99 = mean - 2.326 * std  # 99% confidence
        
        # Historical CVaR (Expected Shortfall)
        sorted_returns = returns.sort_values()
        n_tail = max(1, int(len(returns) * 0.05))  # Worst 5%
        cvar_95 = sorted_returns[:n_tail].mean()
        
        # Return as negative percentages (losses)
        return abs(var_95), abs(var_99), abs(cvar_95)
    
    def _calculate_risk_adjusted(self, returns: pd.Series) -> tuple:
        """Calculate Sharpe, Sortino, and annualized metrics."""
        # Annualized return
        daily_mean = returns.mean()
        annual_return = daily_mean * self.TRADING_DAYS
        
        # Annualized volatility
        daily_std = returns.std()
        annual_vol = daily_std * np.sqrt(self.TRADING_DAYS)
        
        # Sharpe ratio
        excess_return = annual_return - self.RISK_FREE_RATE
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = downside.std()
            annual_downside = downside_std * np.sqrt(self.TRADING_DAYS)
        else:
            annual_downside = annual_vol  # No downside = use total vol
        
        sortino = excess_return / annual_downside if annual_downside > 0 else 0
        
        return sharpe, sortino, annual_return, annual_vol
    
    def _calculate_drawdown(self, returns: pd.Series) -> tuple:
        """Calculate maximum drawdown, duration, and current drawdown."""
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cumulative.expanding().max()
        
        # Drawdown series
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Duration (days from peak to trough)
        if max_dd < 0:
            dd_end_idx = drawdown.idxmin()
            dd_start_idx = cumulative[:dd_end_idx].idxmax()
            dd_duration = len(cumulative[dd_start_idx:dd_end_idx])
        else:
            dd_duration = 0
        
        # Current drawdown (from recent peak)
        current_peak = cumulative.max()
        current_value = cumulative.iloc[-1]
        current_dd = (current_value - current_peak) / current_peak
        
        return abs(max_dd), dd_duration, current_dd
    
    def _calculate_market_metrics(self, stock_returns: pd.Series,
                                 market_returns: Optional[pd.Series]) -> tuple:
        """Calculate beta, alpha, R², and correlation vs benchmark."""
        beta = 1.0
        alpha = 0.0
        r_squared = 0.0
        correlation = 0.0
        
        if market_returns is None or len(market_returns) < 60:
            return beta, alpha, r_squared, correlation
        
        # Align dates
        common_idx = stock_returns.index.intersection(market_returns.index)
        
        if len(common_idx) < 60:
            return beta, alpha, r_squared, correlation
        
        stock = stock_returns[common_idx]
        market = market_returns[common_idx]
        
        # Beta = Cov(stock, market) / Var(market)
        covariance = np.cov(stock, market)[0][1]
        market_var = np.var(market)
        beta = covariance / market_var if market_var > 0 else 1.0
        
        # Annualized returns
        stock_annual = stock.mean() * self.TRADING_DAYS
        market_annual = market.mean() * self.TRADING_DAYS
        
        # Jensen's Alpha = Actual - [Rf + β(Rm - Rf)]
        expected_return = self.RISK_FREE_RATE + beta * (market_annual - self.RISK_FREE_RATE)
        alpha = stock_annual - expected_return
        
        # Correlation and R²
        correlation = np.corrcoef(stock, market)[0][1]
        r_squared = correlation ** 2
        
        # Cap extreme values
        beta = max(0.1, min(3.0, beta))
        
        return beta, alpha, r_squared, correlation
    
    def _calculate_position_sizing(self, annual_return: float,
                                   annual_vol: float,
                                   var_95: float,
                                   max_dd: float,
                                   beta: float,
                                   sharpe: float) -> tuple:
        """
        Calculate recommended position size using multiple methods.
        
        Combines:
        1. Kelly Criterion (with cap)
        2. Risk budget approach
        3. Beta-adjusted sizing
        """
        # Kelly Criterion: f* = (μ - r) / σ²
        excess = annual_return - self.RISK_FREE_RATE
        if annual_vol > 0:
            kelly_full = excess / (annual_vol ** 2)
        else:
            kelly_full = 0
        
        # Use quarter-Kelly (more conservative)
        kelly = max(0, min(kelly_full * 0.25, 0.25))
        
        # Risk budget approach
        # Base: 5% position for average-risk stock
        base_position = 5.0
        
        # Adjust for VaR
        if var_95 < 0.02:  # Less than 2% daily VaR
            base_position += 2
        elif var_95 > 0.04:  # More than 4% daily VaR
            base_position -= 2
        
        # Adjust for max drawdown
        if max_dd < 0.20:  # Less than 20% max DD
            base_position += 1.5
        elif max_dd > 0.40:  # More than 40% max DD
            base_position -= 2
        
        # Adjust for Sharpe
        if sharpe > 1.5:
            base_position += 2
        elif sharpe > 1.0:
            base_position += 1
        elif sharpe < 0:
            base_position -= 2
        
        # Adjust for beta (high beta = smaller position)
        if beta > 1.5:
            base_position -= 1.5
        elif beta < 0.8:
            base_position += 1
        
        # Combine: average of risk-budget and Kelly-informed
        # Kelly as fraction (0-0.25) → convert to percentage (0-25%)
        kelly_pct = kelly * 100
        
        # Weighted blend (70% risk-budget, 30% Kelly)
        final_position = base_position * 0.7 + kelly_pct * 0.3
        
        # Enforce limits
        final_position = max(1.0, min(10.0, final_position))
        
        # Calculate overall risk score
        risk_score = self._risk_score(var_95, max_dd, beta, annual_vol, sharpe)
        
        # Risk category
        if risk_score < 30:
            risk_cat = "low"
        elif risk_score < 50:
            risk_cat = "moderate"
        elif risk_score < 70:
            risk_cat = "high"
        else:
            risk_cat = "very_high"
        
        return kelly, final_position, risk_score, risk_cat
    
    def _risk_score(self, var_95: float, max_dd: float, beta: float,
                    annual_vol: float, sharpe: float) -> float:
        """Calculate composite risk score (0-100, higher = riskier)."""
        score = 30  # Base
        
        # VaR contribution (0-20)
        score += min(20, var_95 * 400)  # 5% VaR = max score
        
        # Drawdown contribution (0-20)
        score += min(20, max_dd * 40)  # 50% DD = max score
        
        # Beta contribution (0-15)
        score += min(15, max(0, (beta - 0.5) * 12))
        
        # Volatility contribution (0-15)
        score += min(15, annual_vol * 100 * 0.2)  # 75% vol = max
        
        # Sharpe penalty (0-10)
        if sharpe < 0:
            score += 10
        elif sharpe < 0.5:
            score += 5
        
        return min(100, max(0, score))