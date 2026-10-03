"""
Data validation utilities for Indian stock market data.
"""
import re
from typing import Optional


def validate_nse_symbol(symbol: str) -> bool:
    """
    Validate NSE symbol format.
    NSE symbols are uppercase alphanumeric, 1-10 characters.
    Examples: RELIANCE, TCS, HDFCBANK, INFY, M&M, BAJAJ-AUTO
    """
    # Allow alphanumeric, hyphen, ampersand (M&M, BAJAJ-AUTO)
    pattern = r'^[A-Z0-9&\-]{1,10}$'
    return bool(re.match(pattern, symbol))


def validate_bse_code(code: str) -> bool:
    """
    Validate BSE code format.
    BSE codes are numeric, typically 5-6 digits.
    Examples: 500325, 532540
    """
    pattern = r'^[0-9]{5,6}$'
    return bool(re.match(pattern, code))


def sanitize_price(price: Optional[float]) -> Optional[float]:
    """Sanitize price values - ensure positive and reasonable."""
    if price is None:
        return None
    if price <= 0:
        return None
    if price > 1000000:  # > ₹10 lakh - probably data error
        return None
    return round(price, 2)


def sanitize_percentage(pct: Optional[float]) -> Optional[float]:
    """Sanitize percentage values - ensure reasonable range."""
    if pct is None:
        return None
    if abs(pct) > 100:  # > 100% change is suspicious
        return None
    return round(pct, 2)


def is_market_open() -> bool:
    """Check if Indian market is currently open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
    from datetime import datetime, time
    import zoneinfo
    
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Check if within market hours
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    return market_open <= now.time() <= market_close