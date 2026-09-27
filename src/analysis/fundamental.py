"""
Fundamental analysis engine.
Fetches data from Screener.in (scraped) and yfinance.
All computations are deterministic.
"""
import io
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from typing import Optional, Dict
import re
import time

from src.schemas.stock_analysis import (
    FundamentalMetrics, MarketCapCategory
)
from src.data.cache import DataCache
from src.data.price_fetcher import PriceFetcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FundamentalAnalyzer:
    """Fetches and computes fundamental metrics for Indian stocks."""
    
    SCREENER_BASE_URL = "https://www.screener.in/company"
    
    def __init__(self, cache: Optional[DataCache] = None,
                 price_fetcher: Optional[PriceFetcher] = None):
        self.cache = cache or DataCache()
        self.price_fetcher = price_fetcher or PriceFetcher()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

    def _table_to_df(self, table_element) -> Optional[pd.DataFrame]:
        """Parse a BeautifulSoup table element to DataFrame."""
        if table_element is None:
            return None
        
        try:
            html = str(table_element)
            # Wrap in proper HTML structure for pandas to parse
            full_html = f"<html><body>{html}</body></html>"
            dfs = pd.read_html(io.StringIO(full_html))
            return dfs[0] if dfs else None
        except Exception as e:
            logger.error(f"Error parsing table: {e}")
            return None
    
    def analyze(self, nse_symbol: str) -> Optional[FundamentalMetrics]:
        """
        Perform complete fundamental analysis.
        
        Args:
            nse_symbol: NSE symbol (e.g., "RELIANCE")
        
        Returns:
            FundamentalMetrics object or None if analysis fails
        """
        logger.info(f"Starting fundamental analysis for {nse_symbol}")
        
        # Check cache (fundamentals cached for 1 week)
        cache_key = f"{nse_symbol}_fundamentals"
        cached = self.cache.get(cache_key, max_age_hours=168)
        if cached:
            logger.debug(f"Cache hit for {nse_symbol} fundamentals")
            return FundamentalMetrics(**cached)
        
        # Get data from multiple sources
        screener_data = self._fetch_from_screener(nse_symbol)
        yahoo_data = self._fetch_from_yahoo(nse_symbol)
        
        # Merge data (Screener.in takes priority for India-specific metrics)
        merged = self._merge_data(screener_data, yahoo_data)
        
        if not merged:
            logger.error(f"Failed to fetch fundamental data for {nse_symbol}")
            return None
        
        # Compute derived metrics
        metrics = self._build_metrics(nse_symbol, merged)
        
        # Cache the result
        if metrics:
            self.cache.set(
                cache_key, 
                metrics.model_dump(), 
                ttl_hours=168, 
                data_type="json"
            )
        
        return metrics
    
    def _fetch_from_screener(self, symbol: str) -> Optional[Dict]:
        """Fetch fundamental data from Screener.in."""
        cache_key = f"{symbol}_screener_raw"
        cached = self.cache.get(cache_key, max_age_hours=168)
        if cached:
            return cached
        
        url = f"{self.SCREENER_BASE_URL}/{symbol}/consolidated/"
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                # Try standalone financials
                url = f"{self.SCREENER_BASE_URL}/{symbol}/"
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"Screener.in returned {response.status_code} for {symbol}")
                    return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {}
            
            # Extract company name
            name_elem = soup.find('h1')
            if name_elem:
                data['company_name'] = name_elem.text.strip()
            
            # Extract market cap
            data.update(self._parse_market_info(soup))
            
            # Extract ratios (ROCE, ROE, etc.)
            data.update(self._parse_ratios(soup))
            
            # Extract shareholding pattern
            data.update(self._parse_shareholding(soup))
            
            # Extract financial data
            data['financials'] = self._parse_financials(soup)
            
            # Cache raw data
            self.cache.set(cache_key, data, ttl_hours=168, data_type="json")
            
            return data
            
        except Exception as e:
            logger.error(f"Screener.in fetch error for {symbol}: {e}")
            return None
    
    def _fetch_from_yahoo(self, symbol: str) -> Dict:
        """Fetch basic fundamentals from Yahoo Finance."""
        yahoo_symbol = f"{symbol}.NS"
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            data = {
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'book_value': info.get('bookValue'),
                'earnings_per_share': info.get('trailingEps'),
                'revenue_per_share': info.get('revenuePerShare'),
            }
            
            # Clean None values and convert percentages
            for key, value in data.items():
                if value is not None:
                    if 'margin' in key or 'return' in key or 'growth' in key or 'yield' in key:
                        # Yahoo returns some as decimals, some as percentages
                        if abs(value) < 1:  # Decimal format
                            data[key] = value * 100
                    elif key == 'debt_to_equity' and value > 10:
                        # Yahoo sometimes returns D/E * 100
                        data[key] = value / 100
            
            return {k: v for k, v in data.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Yahoo Finance fetch error for {symbol}: {e}")
            return {}
    
    def _parse_market_info(self, soup: BeautifulSoup) -> Dict:
        """Parse market cap and basic info from Screener.in."""
        data = {}
        
        # Market cap - usually in the company info section
        for li in soup.find_all('li', class_='flex'):
            text = li.text.strip().lower()
            if 'market cap' in text:
                # Extract the value (usually in crores)
                value_text = text.split(':')[-1].strip()
                value = self._parse_indian_number(value_text)
                if value:
                    data['market_cap_cr'] = value
            
            elif 'current price' in text:
                value_text = text.split(':')[-1].strip().replace('₹', '').replace(',', '')
                try:
                    data['current_price'] = float(value_text)
                except ValueError:
                    pass
            
            elif 'book value' in text:
                value_text = text.split(':')[-1].strip().replace('₹', '').replace(',', '')
                try:
                    data['book_value'] = float(value_text)
                except ValueError:
                    pass
            
            elif 'stock p/e' in text or 'pe ratio' in text.lower():
                value_text = text.split(':')[-1].strip()
                try:
                    data['pe_ratio'] = float(value_text)
                except ValueError:
                    pass
            
            elif 'roce' in text:
                value_text = text.split(':')[-1].strip().replace('%', '')
                try:
                    data['roce'] = float(value_text)
                except ValueError:
                    pass
            
            elif 'roe' in text:
                value_text = text.split(':')[-1].strip().replace('%', '')
                try:
                    data['roe'] = float(value_text)
                except ValueError:
                    pass
        
        return data
    
    def _parse_ratios(self, soup: BeautifulSoup) -> Dict:
        """Parse financial ratios from the ratios section."""
        data = {}
        
        # Find the ratios section
        ratios_section = soup.find('section', {'id': 'ratios'})
        if not ratios_section:
            return data
        
        table = ratios_section.find('table')
        if not table:
            return data
        
        df = self._table_to_df(table)
        if df is None or df.empty:
            return data
        
        try:
            # Get latest values (last column)
            if len(df.columns) > 1:
                latest_col = df.columns[-1]
                
                for idx, row in df.iterrows():
                    label = str(row.iloc[0]).lower().strip()
                    value = row[latest_col]
                    
                    if pd.isna(value):
                        continue
                    
                    # Clean value - remove % and commas
                    if isinstance(value, str):
                        value = value.replace('%', '').replace(',', '').strip()
                    
                    # Map labels to our field names
                    if 'roce' in label:
                        data['roce'] = self._safe_float(value)
                    elif 'roe' in label:
                        data['roe'] = self._safe_float(value)
                    elif 'debtor days' in label:
                        data['debtor_days'] = self._safe_float(value)
                    elif 'inventory days' in label:
                        data['inventory_days'] = self._safe_float(value)
                    elif 'days payable' in label:
                        data['days_payable'] = self._safe_float(value)
                    elif 'cash conversion cycle' in label:
                        data['cash_conversion_cycle'] = self._safe_float(value)
                    elif 'working capital' in label:
                        data['working_capital_days'] = self._safe_float(value)
                    elif 'roa' in label:
                        data['roa'] = self._safe_float(value)
        
        except Exception as e:
            logger.error(f"Error parsing ratios table: {e}")
        
        return data

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or pd.isna(value):
            return None
        try:
            if isinstance(value, str):
                value = value.replace('%', '').replace(',', '').strip()
                if not value or value == '-':
                    return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_shareholding(self, soup: BeautifulSoup) -> Dict:
        """Parse shareholding pattern from Screener.in."""
        data = {}
        
        # Look for shareholding section
        shareholding_section = None
        for section in soup.find_all('section'):
            section_id = str(section.get('id', '')).lower()
            if 'shareholding' in section_id or 'shareholders' in section_id:
                shareholding_section = section
                break
        
        if not shareholding_section:
            return data
        
        tables = shareholding_section.find_all('table')
        if not tables:
            return data
        
        for table in tables:
            df = self._table_to_df(table)
            if df is None or df.empty:
                continue
            
            try:
                for idx, row in df.iterrows():
                    # Get the label - it may contain button text
                    label = str(row.iloc[0]).lower()
                    
                    # Get the latest value (last column)
                    value = row.iloc[-1] if len(row) > 1 else None
                    
                    if 'promoters' in label or 'promoter' in label:
                        data['promoter_holding'] = self._safe_float(value)
                    elif 'fii' in label or 'foreign' in label:
                        data['fii_holding'] = self._safe_float(value)
                    elif 'dii' in label or 'domestic' in label:
                        data['dii_holding'] = self._safe_float(value)
                    elif 'government' in label:
                        data['government_holding'] = self._safe_float(value)
                    elif 'public' in label:
                        data['public_holding'] = self._safe_float(value)
            
            except Exception as e:
                logger.error(f"Error parsing shareholding table: {e}")
        
        return data
    
    def _parse_financials(self, soup: BeautifulSoup) -> Dict:
        """Parse P&L and Balance Sheet data."""
        data = {}
        
        # P&L section
        pl_section = soup.find('section', {'id': 'profit-loss'})
        if pl_section:
            table = pl_section.find('table')
            if table:
                df = self._table_to_df(table)
                if df is not None and not df.empty and len(df.columns) > 1:
                    latest_col = df.columns[-1]
                    prev_col = df.columns[-2] if len(df.columns) > 2 else None
                    
                    for idx, row in df.iterrows():
                        label = str(row.iloc[0]).lower()
                        value = row[latest_col]
                        
                        if 'sales' in label and 'growth' not in label:
                            data['revenue'] = self._safe_float(value)
                            if prev_col is not None:
                                data['revenue_prev'] = self._safe_float(row[prev_col])
                        elif 'net profit' in label:
                            data['net_profit'] = self._safe_float(value)
                        elif 'operating profit' in label:
                            data['operating_profit'] = self._safe_float(value)
                        elif 'opm' in label:
                            data['operating_margin'] = self._safe_float(value)
                        elif 'eps' in label:
                            data['eps'] = self._safe_float(value)
                        elif 'dividend payout' in label:
                            data['payout_ratio'] = self._safe_float(value)
        
        # Balance Sheet section
        bs_section = soup.find('section', {'id': 'balance-sheet'})
        if bs_section:
            table = bs_section.find('table')
            if table:
                df = self._table_to_df(table)
                if df is not None and not df.empty and len(df.columns) > 1:
                    latest_col = df.columns[-1]
                    
                    for idx, row in df.iterrows():
                        label = str(row.iloc[0]).lower()
                        value = row[latest_col]
                        
                        if 'borrowings' in label:
                            data['total_debt'] = self._safe_float(value)
                        elif 'reserves' in label:
                            data['reserves'] = self._safe_float(value)
                        elif 'equity capital' in label:
                            data['equity_capital'] = self._safe_float(value)
        
        return data
    
    def _parse_indian_number(self, value_str: str) -> Optional[float]:
        """Parse Indian number format (e.g., '1,23,456' or '1.23 Cr')."""
        if not value_str:
            return None
        
        # Remove currency symbol and commas
        clean = value_str.replace('₹', '').replace(',', '').strip()
        
        # Handle crore/lakh suffixes
        if 'cr' in clean.lower() or 'crore' in clean.lower():
            num_part = clean.lower().split('cr')[0].split('crore')[0].strip()
            try:
                return float(num_part)
            except ValueError:
                return None
        
        if 'l' in clean.lower() or 'lakh' in clean.lower():
            num_part = clean.lower().split('l')[0].split('lakh')[0].strip()
            try:
                return float(num_part) / 100  # Convert lakh to crore
            except ValueError:
                return None
        
        # Plain number
        try:
            return float(clean)
        except ValueError:
            return None
    
    def _merge_data(self, screener: Optional[Dict], yahoo: Dict) -> Dict:
        """Merge data from multiple sources. Screener.in takes priority."""
        if not screener and not yahoo:
            return None
        
        merged = {}
        
        # Start with Yahoo data
        merged.update(yahoo)
        
        # Override with Screener.in data (more reliable for India)
        if screener:
            # Only override if Screener has the value
            for key in ['pe_ratio', 'pb_ratio', 'roce', 'roe', 
                       'market_cap_cr', 'debt_to_equity',
                       'promoter_holding', 'pledged_pct', 'fii_holding', 'dii_holding']:
                if key in screener and screener[key] is not None:
                    merged[key] = screener[key]
            
            # Add Screener-only fields
            for key in ['company_name', 'net_margin', 'sales_growth', 
                       'profit_growth', 'payout_ratio', 'cash_conversion_cycle',
                       'working_capital_days', 'total_debt', 'reserves']:
                if key in screener and screener[key] is not None:
                    merged[key] = screener[key]
        
        return merged
    
    def _build_metrics(self, symbol: str, data: Dict) -> Optional[FundamentalMetrics]:
        """Build FundamentalMetrics from merged data."""
        try:
            # Determine market cap category
            market_cap_cr = data.get('market_cap_cr')
            if market_cap_cr is None and 'market_cap' in data:
                # Convert from absolute to crores
                market_cap_cr = data['market_cap'] / 1e7
            
            cap_category = None
            if market_cap_cr:
                if market_cap_cr > 100000:
                    cap_category = MarketCapCategory.LARGE_CAP
                elif market_cap_cr > 25000:
                    cap_category = MarketCapCategory.MID_CAP
                else:
                    cap_category = MarketCapCategory.SMALL_CAP
            
            metrics = FundamentalMetrics(
                symbol=symbol,
                pe_ratio=data.get('pe_ratio'),
                pb_ratio=data.get('pb_ratio'),
                ps_ratio=data.get('ps_ratio'),
                ev_ebitda=data.get('ev_ebitda'),
                
                # Growth
                revenue_growth_1y=data.get('sales_growth') or data.get('revenue_growth'),
                profit_growth_1y=data.get('profit_growth') or data.get('earnings_growth'),
                
                # Profitability
                gross_margin=data.get('gross_margin'),
                operating_margin=data.get('operating_margin'),
                net_margin=data.get('net_margin') or data.get('profit_margin'),
                roe=data.get('roe') or data.get('return_on_equity'),
                roce=data.get('roce'),
                
                # Leverage
                debt_to_equity=data.get('debt_to_equity'),
                current_ratio=data.get('current_ratio'),
                
                # India-specific
                promoter_holding=data.get('promoter_holding'),
                pledged_pct=data.get('pledged_pct'),
                fii_holding=data.get('fii_holding'),
                dii_holding=data.get('dii_holding'),
                market_cap_cr=market_cap_cr,
                market_cap_category=cap_category,
                
                # Dividends
                dividend_yield=data.get('dividend_yield'),
                payout_ratio=data.get('payout_ratio'),
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error building metrics for {symbol}: {e}")
            return None