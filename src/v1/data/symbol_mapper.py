"""
Symbol mapper for Indian stock exchanges.
Handles mapping between NSE symbols, BSE codes, Yahoo Finance symbols, and ISINs.
"""
import sqlite3
import logging
from typing import Optional
from pathlib import Path

from config.settings import DATA_DIR
from v1.utils.validation import validate_nse_symbol, validate_bse_code
from v1.utils.logger import setup_logger

logger = setup_logger(__name__)


class SymbolMapper:
    """Maps between different Indian stock symbol conventions."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATA_DIR / "symbols.db"
        self._init_db()
        self._populate_common_symbols()
    
    def _init_db(self):
        """Initialize the symbols database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                nse_symbol TEXT PRIMARY KEY,
                bse_code TEXT,
                isin TEXT,
                company_name TEXT,
                yahoo_nse TEXT,
                yahoo_bse TEXT,
                sector TEXT,
                market_cap_cr REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def _populate_common_symbols(self):
        """Populate with common NSE symbols for validation."""
        common_symbols = {
            # Nifty 50 constituents (partial list for validation)
            "RELIANCE": {"bse": "500325", "name": "Reliance Industries"},
            "TCS": {"bse": "532540", "name": "Tata Consultancy Services"},
            "HDFCBANK": {"bse": "500180", "name": "HDFC Bank"},
            "INFY": {"bse": "500209", "name": "Infosys"},
            "ICICIBANK": {"bse": "532174", "name": "ICICI Bank"},
            "HINDUNILVR": {"bse": "500696", "name": "Hindustan Unilever"},
            "ITC": {"bse": "500875", "name": "ITC Limited"},
            "SBIN": {"bse": "500112", "name": "State Bank of India"},
            "BHARTIARTL": {"bse": "532454", "name": "Bharti Airtel"},
            "KOTAKBANK": {"bse": "500247", "name": "Kotak Mahindra Bank"},
            "LT": {"bse": "500510", "name": "Larsen & Toubro"},
            "AXISBANK": {"bse": "532215", "name": "Axis Bank"},
            "ASIANPAINT": {"bse": "500820", "name": "Asian Paints"},
            "MARUTI": {"bse": "532500", "name": "Maruti Suzuki"},
            "SUNPHARMA": {"bse": "524715", "name": "Sun Pharmaceutical"},
            "TITAN": {"bse": "500114", "name": "Titan Company"},
            "ULTRACEMCO": {"bse": "532538", "name": "UltraTech Cement"},
            "NESTLEIND": {"bse": "500790", "name": "Nestle India"},
            "BAJFINANCE": {"bse": "500034", "name": "Bajaj Finance"},
            "WIPRO": {"bse": "507685", "name": "Wipro"},
            "M&M": {"bse": "500520", "name": "Mahindra & Mahindra"},
            "BAJAJFINSV": {"bse": "532978", "name": "Bajaj Finserv"},
            "ADANIENT": {"bse": "512599", "name": "Adani Enterprises"},
            "ADANIPORTS": {"bse": "532921", "name": "Adani Ports & SEZ"},
            "TATAMOTORS": {"bse": "500570", "name": "Tata Motors"},
            "POWERGRID": {"bse": "532898", "name": "Power Grid Corporation"},
            "NTPC": {"bse": "532555", "name": "NTPC Limited"},
            "ONGC": {"bse": "500312", "name": "Oil & Natural Gas Corp"},
            "COALINDIA": {"bse": "533278", "name": "Coal India"},
            "JSWSTEEL": {"bse": "500228", "name": "JSW Steel"},
            "HCLTECH": {"bse": "532281", "name": "HCL Technologies"},
            "TECHM": {"bse": "532755", "name": "Tech Mahindra"},
            "TATASTEEL": {"bse": "500470", "name": "Tata Steel"},
            "GRASIM": {"bse": "500300", "name": "Grasim Industries"},
            "BAJAJ-AUTO": {"bse": "532977", "name": "Bajaj Auto"},
            "HEROMOTOCO": {"bse": "500182", "name": "Hero MotoCorp"},
            "EICHERMOT": {"bse": "505200", "name": "Eicher Motors"},
            "DRREDDY": {"bse": "500124", "name": "Dr. Reddy's Laboratories"},
            "CIPLA": {"bse": "500087", "name": "Cipla"},
            "DIVISLAB": {"bse": "532488", "name": "Divi's Laboratories"},
            "BRITANNIA": {"bse": "500825", "name": "Britannia Industries"},
            "INDUSINDBK": {"bse": "532187", "name": "IndusInd Bank"},
            "APOLLOHOSP": {"bse": "508869", "name": "Apollo Hospitals"},
            "SHRIRAMFIN": {"bse": "511218", "name": "Shriram Finance"},
            "TATACONSUM": {"bse": "500800", "name": "Tata Consumer Products"},
            "HINDALCO": {"bse": "500440", "name": "Hindalco Industries"},
            "SBILIFE": {"bse": "540719", "name": "SBI Life Insurance"},
            "HDFCLIFE": {"bse": "540697", "name": "HDFC Life Insurance"},
            "BPCL": {"bse": "500547", "name": "Bharat Petroleum"},
        }
        
        conn = sqlite3.connect(self.db_path)
        for nse_sym, info in common_symbols.items():
            conn.execute("""
                INSERT OR IGNORE INTO symbols 
                (nse_symbol, bse_code, company_name, yahoo_nse, yahoo_bse)
                VALUES (?, ?, ?, ?, ?)
            """, (
                nse_sym, 
                info["bse"], 
                info["name"],
                f"{nse_sym}.NS",
                f"{info['bse']}.BO"
            ))
        conn.commit()
        conn.close()
        logger.info(f"Populated {len(common_symbols)} common symbols")
    
    def nse_to_yahoo(self, nse_symbol: str) -> str:
        """Convert NSE symbol to Yahoo Finance format: RELIANCE -> RELIANCE.NS"""
        if not validate_nse_symbol(nse_symbol):
            raise ValueError(f"Invalid NSE symbol: {nse_symbol}")
        return f"{nse_symbol}.NS"
    
    def bse_to_yahoo(self, bse_code: str) -> str:
        """Convert BSE code to Yahoo Finance format: 500325 -> 500325.BO"""
        if not validate_bse_code(bse_code):
            raise ValueError(f"Invalid BSE code: {bse_code}")
        return f"{bse_code}.BO"
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol information from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM symbols WHERE nse_symbol = ? OR bse_code = ?",
            (symbol, symbol)
        ).fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def validate_symbol(self, symbol: str) -> dict:
        """
        Validate a symbol and return its information.
        Raises ValueError if symbol is invalid.
        """
        info = self.get_symbol_info(symbol)
        
        if info:
            return {
                "valid": True,
                "nse_symbol": info["nse_symbol"],
                "bse_code": info["bse_code"],
                "company_name": info["company_name"],
                "yahoo_nse": info["yahoo_nse"],
                "yahoo_bse": info["yahoo_bse"],
                "isin": info.get("isin"),
            }
        
        # If not in database, validate format
        if validate_nse_symbol(symbol):
            return {
                "valid": True,
                "nse_symbol": symbol,
                "bse_code": None,
                "company_name": None,
                "yahoo_nse": f"{symbol}.NS",
                "yahoo_bse": None,
                "isin": None,
            }
        elif validate_bse_code(symbol):
            return {
                "valid": True,
                "nse_symbol": None,
                "bse_code": symbol,
                "company_name": None,
                "yahoo_nse": None,
                "yahoo_bse": f"{symbol}.BO",
                "isin": None,
            }
        else:
            raise ValueError(f"Invalid symbol format: {symbol}")
    
    def add_symbol(self, nse_symbol: str, bse_code: str = None, 
                   company_name: str = None, isin: str = None,
                   sector: str = None, market_cap_cr: float = None):
        """Add or update a symbol in the database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO symbols 
            (nse_symbol, bse_code, isin, company_name, yahoo_nse, yahoo_bse, sector, market_cap_cr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            nse_symbol, bse_code, isin, company_name,
            f"{nse_symbol}.NS", 
            f"{bse_code}.BO" if bse_code else None,
            sector, market_cap_cr
        ))
        conn.commit()
        conn.close()

    def resolve_to_nse(self, symbol: str) -> str:
        """
        Resolve any symbol (NSE or BSE) to its NSE equivalent.
        Most Indian stocks are dual-listed, so BSE codes can map to NSE.
        """
        # If it's already an NSE symbol, return as-is
        if validate_nse_symbol(symbol):
            return symbol
        
        # If it's a BSE code, look up the NSE equivalent
        if validate_bse_code(symbol):
            info = self.get_symbol_info(symbol)
            if info and info.get("nse_symbol"):
                return info["nse_symbol"]
            
            # If not in database, try to fetch from Yahoo
            nse_symbol = self._lookup_bse_to_nse(symbol)
            if nse_symbol:
                return nse_symbol
            
            raise ValueError(f"BSE code {symbol} has no NSE equivalent found")
        
        raise ValueError(f"Invalid symbol: {symbol}")

    def _lookup_bse_to_nse(self, bse_code: str) -> Optional[str]:
        """Look up NSE symbol for a BSE code using Yahoo Finance."""
        try:
            import yfinance as yf
            
            # Fetch BSE stock info
            yahoo_symbol = f"{bse_code}.BO"
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            # Yahoo sometimes provides the NSE symbol in the info
            # or we can match by ISIN
            isin = info.get("isin")
            if isin:
                return self._find_nse_by_isin(isin)
            
            # Try matching by company name
            company_name = info.get("longName") or info.get("shortName")
            if company_name:
                return self._find_nse_by_name(company_name)
            
            return None
            
        except Exception:
            return None

    def _find_nse_by_isin(self, isin: str) -> Optional[str]:
        """Find NSE symbol by ISIN in database."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT nse_symbol FROM symbols WHERE isin = ?", (isin,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _find_nse_by_name(self, name: str) -> Optional[str]:
        """Find NSE symbol by company name (fuzzy match)."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT nse_symbol, company_name FROM symbols WHERE company_name IS NOT NULL"
        ).fetchall()
        conn.close()
        
        name_lower = name.lower()
        for nse_sym, comp_name in rows:
            if comp_name and comp_name.lower() in name_lower:
                return nse_sym
        
        return None