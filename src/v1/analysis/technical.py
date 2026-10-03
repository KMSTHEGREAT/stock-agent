"""
Technical analysis engine using pandas-ta.
All computations are deterministic — no LLM involved.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta

from src.v1.data.price_fetcher import PriceFetcher
from src.v1.schemas.stock_analysis import TechnicalMetrics, TrendDirection
from src.v1.utils.logger import setup_logger
from config.settings import TECHNICAL, MARKET

logger = setup_logger(__name__)


class TechnicalAnalyzer:
    """Computes technical indicators and scores for Indian stocks."""
    
    def __init__(self, price_fetcher: Optional[PriceFetcher] = None):
        self.price_fetcher = price_fetcher or PriceFetcher()
    
    def analyze(self, nse_symbol: str, period: str = "1y") -> Optional[TechnicalMetrics]:
        """
        Perform complete technical analysis for a stock.
        
        Args:
            nse_symbol: NSE symbol (e.g., "RELIANCE")
            period: Historical data period
        
        Returns:
            TechnicalMetrics object or None if analysis fails
        """
        logger.info(f"Starting technical analysis for {nse_symbol}")
        
        # Fetch historical data
        df = self.price_fetcher.get_historical_data(nse_symbol, period=period)
        if df is None or df.empty:
            logger.error(f"No data available for {nse_symbol}")
            return None
        
        # Compute all indicators
        df = self._compute_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Determine trend
        trend = self._determine_trend(df)
        
        # Find support/resistance
        support, resistance = self._find_support_resistance(df)
        
        # Check circuit limit proximity
        near_circuit = self._check_circuit_limit(df)
        
        # Calculate price changes
        price_changes = self._calculate_price_changes(df)
        
        # Build metrics object
        metrics = TechnicalMetrics(
            symbol=nse_symbol,
            current_price=round(float(latest["Close"]), 2),
            rsi=round(float(latest.get("RSI", 50)), 2),
            macd=round(float(latest.get("MACD", 0)), 4) if pd.notna(latest.get("MACD")) else None,
            macd_signal=round(float(latest.get("MACD_signal", 0)), 4) if pd.notna(latest.get("MACD_signal")) else None,
            macd_histogram=round(float(latest.get("MACD_hist", 0)), 4) if pd.notna(latest.get("MACD_hist")) else None,
            atr=round(float(latest.get("ATR", 0)), 2) if pd.notna(latest.get("ATR")) else None,
            sma_20=round(float(latest.get("SMA_20", 0)), 2) if pd.notna(latest.get("SMA_20")) else None,
            sma_50=round(float(latest.get("SMA_50", 0)), 2) if pd.notna(latest.get("SMA_50")) else None,
            sma_200=round(float(latest.get("SMA_200", 0)), 2) if pd.notna(latest.get("SMA_200")) else None,
            bb_upper=round(float(latest.get("BB_upper", 0)), 2) if pd.notna(latest.get("BB_upper")) else None,
            bb_lower=round(float(latest.get("BB_lower", 0)), 2) if pd.notna(latest.get("BB_lower")) else None,
            bb_position=self._get_bb_position(latest),
            volume_ma_20=round(float(latest.get("Volume_MA_20", 0)), 0) if pd.notna(latest.get("Volume_MA_20")) else None,
            volume_ratio=round(float(latest.get("Volume_ratio", 1)), 2) if pd.notna(latest.get("Volume_ratio")) else None,
            trend=trend,
            support_level=round(support, 2) if support else None,
            resistance_level=round(resistance, 2) if resistance else None,
            near_circuit=near_circuit,
            price_change_pct_1d=price_changes.get("1d"),
            price_change_pct_1w=price_changes.get("1w"),
            price_change_pct_1m=price_changes.get("1m"),
        )
        
        logger.info(f"Technical analysis complete for {nse_symbol}: "
                   f"trend={trend}, RSI={metrics.rsi}")
        
        return metrics
    
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators on the DataFrame."""
        # RSI
        df["RSI"] = ta.rsi(df["Close"], length=TECHNICAL["rsi_period"])
        
        # MACD
        macd = ta.macd(df["Close"], 
                    fast=TECHNICAL["macd_fast"],
                    slow=TECHNICAL["macd_slow"],
                    signal=TECHNICAL["macd_signal"])
        if macd is not None:
            # Find columns by prefix instead of exact name
            for col in macd.columns:
                if col.startswith("MACD_"):
                    df["MACD"] = macd[col]
                elif col.startswith("MACDs_"):
                    df["MACD_signal"] = macd[col]
                elif col.startswith("MACDh_"):
                    df["MACD_hist"] = macd[col]
        
        # Moving Averages
        for period in TECHNICAL["sma_periods"]:
            df[f"SMA_{period}"] = ta.sma(df["Close"], length=period)
        
        # EMA
        df["EMA_12"] = ta.ema(df["Close"], length=12)
        df["EMA_26"] = ta.ema(df["Close"], length=26)
        
        # Bollinger Bands - handle dynamic column names
        bbands = ta.bbands(df["Close"], 
                        length=TECHNICAL["bb_period"],
                        std=TECHNICAL["bb_std"])
        if bbands is not None and not bbands.empty:
            for col in bbands.columns:
                if col.startswith("BBL_"):
                    df["BB_lower"] = bbands[col]
                elif col.startswith("BBM_"):
                    df["BB_mid"] = bbands[col]
                elif col.startswith("BBU_"):
                    df["BB_upper"] = bbands[col]
        
        # ATR (Average True Range)
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], 
                        length=TECHNICAL["atr_period"])
        
        # Volume indicators
        df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
        
        # Calculate volume ratio (current volume / 20-day average)
        if len(df) >= 20:
            df["Volume_ratio"] = df["Volume"] / df["Volume_MA_20"]
        
        # OBV (On Balance Volume)
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
        
        # Stochastic
        stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
        if stoch is not None and not stoch.empty:
            for col in stoch.columns:
                if col.startswith("STOCHk_"):
                    df["Stoch_K"] = stoch[col]
                elif col.startswith("STOCHd_"):
                    df["Stoch_D"] = stoch[col]
        
        return df
    
    def _determine_trend(self, df: pd.DataFrame) -> TrendDirection:
        """Determine overall trend based on multiple indicators."""
        latest = df.iloc[-1]
        price = latest["Close"]
        
        # Count bullish signals
        bullish_signals = 0
        total_signals = 0
        
        # 1. Price vs SMA_20
        if pd.notna(latest.get("SMA_20")):
            total_signals += 1
            if price > latest["SMA_20"]:
                bullish_signals += 1
        
        # 2. Price vs SMA_50
        if pd.notna(latest.get("SMA_50")):
            total_signals += 1
            if price > latest["SMA_50"]:
                bullish_signals += 1
        
        # 3. SMA_20 vs SMA_50 (Golden/Death cross)
        if pd.notna(latest.get("SMA_20")) and pd.notna(latest.get("SMA_50")):
            total_signals += 1
            if latest["SMA_20"] > latest["SMA_50"]:
                bullish_signals += 1
        
        # 4. Price vs SMA_200 (long-term trend)
        if pd.notna(latest.get("SMA_200")):
            total_signals += 1
            if price > latest["SMA_200"]:
                bullish_signals += 1
        
        # 5. MACD histogram positive
        if pd.notna(latest.get("MACD_hist")):
            total_signals += 1
            if latest["MACD_hist"] > 0:
                bullish_signals += 1
        
        # Determine trend based on signal ratio
        if total_signals > 0:
            ratio = bullish_signals / total_signals
            if ratio >= 0.7:
                return TrendDirection.UP
            elif ratio <= 0.3:
                return TrendDirection.DOWN
        
        return TrendDirection.SIDEWAYS
    
    def _find_support_resistance(self, df: pd.DataFrame, 
                                lookback: int = 20) -> tuple:
        """Find support and resistance levels using recent highs/lows."""
        recent = df.tail(lookback)
        
        if recent.empty:
            return None, None
        
        # Support: recent low that's below current price
        current_price = df["Close"].iloc[-1]
        
        # Find swing lows (local minimums)
        lows = recent["Low"].values
        supports = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                if lows[i] < current_price:
                    supports.append(lows[i])
        
        # Find swing highs (local maximums)
        highs = recent["High"].values
        resistances = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                if highs[i] > current_price:
                    resistances.append(highs[i])
        
        # Return closest support and resistance
        support = max(supports) if supports else None
        resistance = min(resistances) if resistances else None
        
        # Fallback to SMA levels if no clear S/R
        if support is None and pd.notna(df["SMA_50"].iloc[-1]):
            sma_50 = df["SMA_50"].iloc[-1]
            if sma_50 < current_price:
                support = sma_50
        
        if resistance is None and pd.notna(df["BB_upper"].iloc[-1]):
            bb_upper = df["BB_upper"].iloc[-1]
            if bb_upper > current_price:
                resistance = bb_upper
        
        return support, resistance
    
    def _check_circuit_limit(self, df: pd.DataFrame) -> bool:
        """Check if stock is near circuit limit (>18% daily move)."""
        if len(df) < 2:
            return False
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        if prev["Close"] == 0:
            return False
        
        change_pct = abs((latest["Close"] - prev["Close"]) / prev["Close"] * 100)
        return change_pct > 18  # Circuit limit threshold
    
    def _get_bb_position(self, latest: pd.Series) -> Optional[float]:
        """Get position within Bollinger Bands (0 = lower, 1 = upper)."""
        bb_upper = latest.get("BB_upper")
        bb_lower = latest.get("BB_lower")
        price = latest.get("Close")
        
        if pd.isna(bb_upper) or pd.isna(bb_lower) or bb_upper == bb_lower:
            return None
        
        position = (price - bb_lower) / (bb_upper - bb_lower)
        return round(max(0, min(1, position)), 2)
    
    def _calculate_price_changes(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate price changes over different periods."""
        if df.empty:
            return {"1d": None, "1w": None, "1m": None}
        
        current = df["Close"].iloc[-1]
        changes = {}
        
        # 1 day
        if len(df) >= 2:
            prev = df["Close"].iloc[-2]
            if prev != 0:
                changes["1d"] = round((current - prev) / prev * 100, 2)
        
        # 1 week (5 trading days)
        if len(df) >= 6:
            week_ago = df["Close"].iloc[-6]
            if week_ago != 0:
                changes["1w"] = round((current - week_ago) / week_ago * 100, 2)
        
        # 1 month (21 trading days)
        if len(df) >= 22:
            month_ago = df["Close"].iloc[-22]
            if month_ago != 0:
                changes["1m"] = round((current - month_ago) / month_ago * 100, 2)
        
        return changes

