"""
Price data fetcher for Indian stocks.
Primary: yfinance (.NS suffix)
Fallback: NSEPython for live quotes
"""
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

from src.data.cache import DataCache
from src.data.symbol_mapper import SymbolMapper
from src.utils.logger import setup_logger
from src.utils.validation import sanitize_price, is_market_open

logger = setup_logger(__name__)


class PriceFetcher:
    """Fetches historical and current price data for Indian stocks."""
    
    def __init__(self, cache: Optional[DataCache] = None,
                 symbol_mapper: Optional[SymbolMapper] = None):
        self.cache = cache or DataCache()
        self.symbol_mapper = symbol_mapper or SymbolMapper()
    
    def get_historical_data(self, nse_symbol: str, period: str = "1y",
                           interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data with corporate action adjustment.
        
        Args:
            nse_symbol: NSE symbol (e.g., "RELIANCE")
            period: Time period ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
            interval: Data interval ("1d", "1wk", "1mo")
        
        Returns:
            DataFrame with OHLCV data, or None if fetch fails
        """
        cache_key = f"{nse_symbol}_hist_{period}_{interval}"
        
        # Check cache (24-hour TTL for daily data)
        cached = self.cache.get(cache_key, max_age_hours=24)
        if cached is not None:
            logger.debug(f"Cache hit for {nse_symbol} historical data")
            return pd.DataFrame(cached)
        
        try:
            # Convert to Yahoo Finance symbol
            yahoo_symbol = self.symbol_mapper.nse_to_yahoo(nse_symbol)
            
            # Fetch data with auto_adjust=True for corporate actions
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=interval, 
                              auto_adjust=True)
            
            if df.empty:
                logger.warning(f"No data returned for {yahoo_symbol}")
                return None
            
            # Clean and validate data
            df = self._clean_price_data(df)
            
            # Cache the data
            self.cache.set(cache_key, df.to_dict(), ttl_hours=24, 
                          data_type="json")
            
            logger.info(f"Fetched {len(df)} rows for {nse_symbol} ({period})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {nse_symbol}: {e}")
            return None
    
    def get_current_price(self, nse_symbol: str) -> Optional[float]:
        """Get current/latest price for a stock."""
        cache_key = f"{nse_symbol}_current_price"
        
        # Cache for 6 hours if market closed, 15 minutes if open
        ttl = 0.25 if is_market_open() else 6
        
        cached = self.cache.get(cache_key, max_age_hours=ttl)
        if cached is not None:
            return cached
        
        try:
            yahoo_symbol = self.symbol_mapper.nse_to_yahoo(nse_symbol)
            ticker = yf.Ticker(yahoo_symbol)
            
            # Try fast_info first
            try:
                price = ticker.fast_info.last_price
            except:
                # Fallback to history
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                else:
                    return None
            
            price = sanitize_price(price)
            if price:
                self.cache.set(cache_key, price, ttl_hours=ttl, data_type="json")
            
            return price
            
        except Exception as e:
            logger.error(f"Error fetching current price for {nse_symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Fetch current prices for multiple symbols efficiently."""
        results = {}
        
        # Check which need fetching
        to_fetch = []
        for sym in symbols:
            cache_key = f"{sym}_current_price"
            cached = self.cache.get(cache_key, max_age_hours=1)
            if cached is not None:
                results[sym] = cached
            else:
                to_fetch.append(sym)
        
        # Batch fetch using yf.download for efficiency
        if to_fetch:
            yahoo_symbols = [self.symbol_mapper.nse_to_yahoo(s) for s in to_fetch]
            
            try:
                data = yf.download(yahoo_symbols, period="1d", 
                                 progress=False, auto_adjust=True)
                
                if not data.empty:
                    for i, sym in enumerate(to_fetch):
                        try:
                            if len(to_fetch) == 1:
                                price = data["Close"].iloc[-1]
                            else:
                                price = data["Close"].iloc[-1, i]
                            
                            price = sanitize_price(float(price))
                            results[sym] = price
                            
                            # Cache
                            self.cache.set(
                                f"{sym}_current_price", price, 
                                ttl_hours=1, data_type="json"
                            )
                        except (IndexError, KeyError, ValueError):
                            results[sym] = None
                else:
                    for sym in to_fetch:
                        results[sym] = None
                        
            except Exception as e:
                logger.error(f"Batch price fetch error: {e}")
                for sym in to_fetch:
                    results[sym] = None
        
        return results
    
    def get_market_indices(self) -> Dict[str, Optional[float]]:
        """Get current values for major Indian market indices."""
        from config.settings import MARKET
        
        indices = {}
        for name, yahoo_symbol in MARKET["indices"].items():
            try:
                ticker = yf.Ticker(yahoo_symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    indices[name] = sanitize_price(float(hist["Close"].iloc[-1]))
                else:
                    indices[name] = None
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                indices[name] = None
        
        return indices
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data."""
        # Remove rows with null close prices
        df = df.dropna(subset=["Close"])
        
        # Ensure numeric types
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove rows with invalid prices
        df = df[(df["Close"] > 0) & (df["Open"] > 0)]
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def get_52_week_high_low(self, nse_symbol: str) -> Dict[str, Optional[float]]:
        """Get 52-week high and low for a stock."""
        df = self.get_historical_data(nse_symbol, period="1y")
        if df is None or df.empty:
            return {"high_52w": None, "low_52w": None}
        
        return {
            "high_52w": sanitize_price(float(df["High"].max())),
            "low_52w": sanitize_price(float(df["Low"].min()))
        }