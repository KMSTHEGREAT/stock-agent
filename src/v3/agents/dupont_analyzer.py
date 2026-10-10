"""
DuPont Analysis Agent — CFA Level 1 methodology.

Decomposes ROE into 5 components:
ROE = (NI/EBT) × (EBT/EBIT) × (EBIT/Rev) × (Rev/Assets) × (Assets/Equity)

This reveals what's actually driving returns:
- Tax Burden (NI/EBT): Tax efficiency
- Interest Burden (EBT/EBIT): Interest cost efficiency  
- Operating Margin (EBIT/Rev): Core profitability
- Asset Turnover (Rev/Assets): Asset efficiency
- Financial Leverage (Assets/Equity): Debt usage

Quality assessment: Margin-driven ROE > Turnover-driven > Leverage-driven
"""
import yfinance as yf
import logging
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DuPontResult:
    """Complete DuPont decomposition output."""
    symbol: str
    roe: float                    # Return on Equity (%)
    roa: float                    # Return on Assets (%)
    
    # 5-component decomposition
    tax_burden: float            # NI/EBT (higher = lower tax rate)
    interest_burden: float       # EBT/EBIT (higher = lower interest cost)
    operating_margin: float      # EBIT/Revenue (%)
    asset_turnover: float        # Revenue/Assets (x)
    financial_leverage: float    # Assets/Equity (x)
    
    # 3-component simplified
    net_profit_margin: float     # NI/Revenue (%)
    
    # Quality assessment
    primary_driver: str          # "margin", "turnover", "leverage"
    roe_quality_score: float     # 0-100
    quality_interpretation: str
    
    # Raw inputs for audit
    revenue: float               # In crores
    ebit: float                  # In crores
    ebt: float                   # In crores  
    net_income: float            # In crores
    total_assets: float          # In crores
    total_equity: float          # In crores
    
    @property
    def full_interpretation(self) -> str:
        """Detailed interpretation of the decomposition."""
        lines = [
            f"ROE: {self.roe:.1f}%",
            f"",
            f"Drivers (contribution to ROE):",
            f"  1. Operating Margin: {self.operating_margin:.1f}% — {'strong' if self.operating_margin > 15 else 'moderate' if self.operating_margin > 8 else 'weak'}",
            f"  2. Asset Turnover: {self.asset_turnover:.2f}x — {'efficient' if self.asset_turnover > 1.0 else 'moderate' if self.asset_turnover > 0.5 else 'low'}",
            f"  3. Financial Leverage: {self.financial_leverage:.2f}x — {'conservative' if self.financial_leverage < 2 else 'moderate' if self.financial_leverage < 3 else 'aggressive'}",
            f"",
            f"Primary Driver: {self.primary_driver.upper()}",
            f"Quality Score: {self.roe_quality_score:.0f}/100",
            f"Assessment: {self.quality_interpretation}",
        ]
        return "\n".join(lines)


class DuPontAnalyzer:
    """Performs 5-component DuPont analysis."""
    
    def analyze(self, symbol: str) -> Optional[DuPontResult]:
        """
        Perform DuPont analysis for a stock.
        
        Args:
            symbol: NSE symbol (e.g., "TCS")
        """
        logger.info(f"[DuPont Agent] Analyzing {symbol}")
        
        # Fetch financial data
        data = self._fetch_data(symbol)
        if not data:
            return None
        
        # Decompose
        result = self._decompose(symbol, data)
        
        if result:
            logger.info(f"[DuPont Agent] {symbol}: ROE={result.roe:.1f}%, "
                       f"driver={result.primary_driver}, quality={result.roe_quality_score:.0f}")
        
        return result
    
    def _fetch_data(self, symbol: str) -> Optional[Dict]:
        """Fetch financial statement data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Get income statement
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            
            if financials is None or financials.empty:
                logger.warning(f"[DuPont Agent] No financials for {symbol}")
                return None
            
            # Extract income statement items (latest year)
            income_data = {}
            for metric in ["Total Revenue", "Operating Income", 
                          "Pretax Income", "Net Income"]:
                if metric in financials.index:
                    income_data[metric] = financials.loc[metric].iloc[0]
            
            # Extract balance sheet items
            balance_data = {}
            if balance_sheet is not None and not balance_sheet.empty:
                for metric in ["Total Assets", "Stockholders Equity"]:
                    if metric in balance_sheet.index:
                        balance_data[metric] = balance_sheet.loc[metric].iloc[0]
            
            # Validate we have minimum required data
            required = ["Total Revenue", "Net Income"]
            for req in required:
                if req not in income_data or income_data[req] is None:
                    logger.warning(f"[DuPont Agent] Missing {req} for {symbol}")
                    return None
            
            if "Stockholders Equity" not in balance_data:
                logger.warning(f"[DuPont Agent] Missing equity for {symbol}")
                return None
            
            # Fill missing EBIT with revenue * 0.15 (assumption)
            if "Operating Income" not in income_data or income_data.get("Operating Income") is None:
                income_data["Operating Income"] = income_data["Total Revenue"] * 0.15
            
            # Fill missing EBT with EBIT * 0.95 (minor interest expense)
            if "Pretax Income" not in income_data or income_data.get("Pretax Income") is None:
                income_data["Pretax Income"] = income_data["Operating Income"] * 0.95
            
            # Estimate total assets if missing
            if "Total Assets" not in balance_data or balance_data.get("Total Assets") is None:
                # Estimate: Revenue / 0.8 (assuming 0.8 asset turnover)
                balance_data["Total Assets"] = income_data["Total Revenue"] / 0.8
            
            return {
                "revenue": income_data["Total Revenue"],
                "ebit": income_data["Operating Income"],
                "ebt": income_data["Pretax Income"],
                "net_income": income_data["Net Income"],
                "total_assets": balance_data["Total Assets"],
                "total_equity": balance_data["Stockholders Equity"],
            }
            
        except Exception as e:
            logger.error(f"[DuPont Agent] Data fetch error: {e}")
            return None
    
    def _detect_bank(self, symbol: str, data: Dict) -> bool:
        """Detect if company is a bank/financial institution."""
        # Known banks
        bank_symbols = {
            "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
            "INDUSINDBK", "BANKBARODA", "PNB", "CANBK", "UNIONBANK",
            "IDFCFIRSTB", "FEDERALBNK", "AUBANK", "BANDHANBNK",
            "RBLBANK", "YESBANK", "CSBBANK", "KARURVYSYA",
            "HDFCLIFE", "SBILIFE", "ICICIPRULI", "LICI",
            "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN", "CHOLAFIN",
            "MUTHOOTFIN", "MANAPPURAM", "PFC", "RECLTD", "IRFC",
        }
        
        if symbol.upper() in bank_symbols:
            return True
        
        # Heuristic: if asset turnover < 0.1 and leverage > 5
        revenue = data.get("revenue", 0)
        total_assets = data.get("total_assets", 1)
        total_equity = data.get("total_equity", 1)
        
        asset_turnover = revenue / total_assets if total_assets > 0 else 0
        leverage = total_assets / total_equity if total_equity > 0 else 0
        
        if asset_turnover < 0.1 and leverage > 5:
            return True
        
        return False
    
    def _bank_decomposition(self, symbol: str, data: Dict) -> Optional[DuPontResult]:
        """
        Bank-specific profitability analysis.
        For banks, standard DuPont is misleading because:
        - 'Revenue' = interest income + fees
        - 'Assets' = loans (huge)
        - 'Leverage' = deposits/equity (naturally high, not risky)
        
        Instead use bank-specific metrics:
        - ROE = Net Income / Equity
        - ROA = Net Income / Total Assets  
        - Net Interest Margin (NIM) proxy
        - Cost-to-Income ratio
        """
        net_income = data.get("net_income", 0)
        total_equity = data.get("total_equity", 0)
        total_assets = data.get("total_assets", 0)
        revenue = data.get("revenue", 0)
        
        if total_equity <= 0 or total_assets <= 0:
            return None
        
        roe = net_income / total_equity
        roa = net_income / total_assets
        
        # Bank-specific quality assessment
        # For banks, ROE of 12-18% is good, leverage is expected
        quality = 50.0
        
        if roe > 0.15:
            quality += 20
        elif roe > 0.12:
            quality += 10
        elif roe < 0.08:
            quality -= 15
        
        # ROA for banks: 1-1.5% is good
        if roa > 0.012:
            quality += 15
        elif roa > 0.01:
            quality += 8
        elif roa < 0.008:
            quality -= 10
        
        quality = max(0, min(100, quality))
        
        # For banks, we don't decompose the same way
        # Just report ROE, ROA, and overall quality
        leverage = total_assets / total_equity
        asset_turnover = revenue / total_assets
        
        interpretation = (
            f"Bank/NBFC: ROE {roe*100:.1f}%, ROA {roa*100:.2f}%. "
            f"Leverage of {leverage:.1f}x is normal for financial institutions. "
        )
        
        if quality > 70:
            interpretation += "Strong profitability for a financial company."
        elif quality > 50:
            interpretation += "Moderate profitability — monitor NIM and asset quality."
        else:
            interpretation += "Weak profitability — check NIM compression and NPAs."
        
        return DuPontResult(
            symbol=symbol,
            roe=round(roe * 100, 2),
            roa=round(roa * 100, 2),
            tax_burden=0.75,  # Standard for Indian banks
            interest_burden=1.0,  # Not meaningful for banks
            operating_margin=round((net_income / revenue) * 100, 2) if revenue > 0 else 0,
            asset_turnover=round(asset_turnover, 4),
            financial_leverage=round(leverage, 2),
            net_profit_margin=round((net_income / revenue) * 100, 2) if revenue > 0 else 0,
            primary_driver="bank_model",  # Special indicator
            roe_quality_score=round(quality, 1),
            quality_interpretation=interpretation,
            revenue=round(revenue / 1e7, 1),
            ebit=round(data.get("ebit", 0) / 1e7, 1),
            ebt=round(data.get("ebt", 0) / 1e7, 1),
            net_income=round(net_income / 1e7, 1),
            total_assets=round(total_assets / 1e7, 1),
            total_equity=round(total_equity / 1e7, 1),
        )
    
    def _decompose(self, symbol: str, data: Dict) -> Optional[DuPontResult]:
        """Perform 5-component DuPont decomposition."""
        try:
            # Detect if this is a bank/financial company
            # Banks have: low asset turnover (0.03-0.05), high leverage (6-12x)
            # and EBT can exceed EBIT (interest income > interest expense)
            is_bank = self._detect_bank(symbol, data)
            
            if is_bank:
                return self._bank_decomposition(symbol, data)
            
            # Standard decomposition for non-banks...
            revenue = data["revenue"]
            ebit = data["ebit"]
            ebt = data["ebt"]
            net_income = data["net_income"]
            total_assets = data["total_assets"]
            total_equity = data["total_equity"]
            
            # Validate denominators
            if any(x <= 0 for x in [revenue, total_assets, total_equity]):
                logger.warning(f"[DuPont Agent] Invalid financial data for {symbol}")
                return None
            
            # Avoid division by zero for EBT and EBIT
            if ebt == 0:
                ebt = net_income / 0.75  # Assume 25% tax
            if ebit == 0:
                ebit = revenue * 0.10   # Assume 10% operating margin
            
            # 5-component decomposition
            # ROE = (NI/EBT) × (EBT/EBIT) × (EBIT/Rev) × (Rev/Assets) × (Assets/Equity)
            
            tax_burden = net_income / ebt if ebt != 0 else 0.75
            interest_burden = ebt / ebit if ebit != 0 else 0.95
            operating_margin = ebit / revenue
            asset_turnover = revenue / total_assets
            financial_leverage = total_assets / total_equity
            
            # Calculate ROE from components
            roe = tax_burden * interest_burden * operating_margin * asset_turnover * financial_leverage
            
            # Also calculate directly for validation
            direct_roe = net_income / total_equity
            
            # Use direct ROE if components don't reconcile (data quality issue)
            if abs(roe - direct_roe) > 0.05:  # 5% tolerance
                roe = direct_roe
            
            # 3-component simplified
            net_profit_margin = net_income / revenue
            
            # ROA
            roa = net_income / total_assets
            
            # Determine primary driver
            # Normalize each component's contribution
            components = {
                "margin": operating_margin * 100,  # Convert to %
                "turnover": asset_turnover,
                "leverage": financial_leverage,
            }
            
            # Weight by typical ranges:
            # - Good operating margin: > 15%
            # - Good asset turnover: > 1.0x
            # - High leverage: > 3.0x
            # Normalize to 0-100 scale for comparison
            normalized = {
                "margin": min(100, operating_margin * 100 / 20 * 100),  # 20% = perfect
                "turnover": min(100, asset_turnover / 2.0 * 100),  # 2.0x = perfect
                "leverage": min(100, financial_leverage / 4.0 * 100),  # 4.0x = max
            }
            
            primary_driver = max(normalized, key=normalized.get)
            
            # Quality score
            quality = self._quality_score(
                operating_margin, asset_turnover, financial_leverage, 
                primary_driver, net_profit_margin
            )
            
            # Interpretation
            interpretation = self._interpret_quality(
                primary_driver, quality, operating_margin, 
                asset_turnover, financial_leverage
            )
            
            return DuPontResult(
                symbol=symbol,
                roe=round(roe * 100, 2),
                roa=round(roa * 100, 2),
                
                tax_burden=round(tax_burden, 3),
                interest_burden=round(interest_burden, 3),
                operating_margin=round(operating_margin * 100, 2),
                asset_turnover=round(asset_turnover, 2),
                financial_leverage=round(financial_leverage, 2),
                
                net_profit_margin=round(net_profit_margin * 100, 2),
                
                primary_driver=primary_driver,
                roe_quality_score=round(quality, 1),
                quality_interpretation=interpretation,
                
                revenue=round(revenue / 1e7, 1),   # Crores
                ebit=round(ebit / 1e7, 1),
                ebt=round(ebt / 1e7, 1),
                net_income=round(net_income / 1e7, 1),
                total_assets=round(total_assets / 1e7, 1),
                total_equity=round(total_equity / 1e7, 1),
            )
            
        except (ZeroDivisionError, ValueError, KeyError) as e:
            logger.error(f"[DuPont Agent] Decomposition error: {e}")
            return None
    
    def _quality_score(self, op_margin: float, asset_turnover: float,
                      leverage: float, primary_driver: str,
                      net_margin: float) -> float:
        """
        Calculate ROE quality score (0-100).
        
        High quality: margin-driven ROE with moderate leverage
        Low quality: leverage-driven ROE with thin margins
        """
        score = 50.0  # Base
        
        # Primary driver adjustment
        if primary_driver == "margin":
            score += 25  # Best quality
        elif primary_driver == "turnover":
            score += 10  # Good quality
        elif primary_driver == "leverage":
            score -= 20  # Riskiest
        
        # Margin quality
        if op_margin > 0.20:
            score += 10
        elif op_margin > 0.15:
            score += 5
        elif op_margin < 0.05:
            score -= 10
        
        # Net margin check
        if net_margin < 0.03:  # Less than 3% net margin
            score -= 10
        elif net_margin > 0.15:  # Above 15% net margin
            score += 10
        
        # Leverage risk
        if leverage > 4.0:
            score -= 15  # Very high leverage
        elif leverage > 3.0:
            score -= 8   # High leverage
        elif leverage < 1.5:
            score += 5   # Conservative
        
        # Turnover bonus
        if asset_turnover > 1.5:
            score += 5   # Very efficient asset use
        
        return max(0, min(100, score))
    
    def _interpret_quality(self, driver: str, quality: float,
                          op_margin: float, turnover: float,
                          leverage: float) -> str:
        """Generate human-readable quality interpretation."""
        if driver == "margin" and quality > 70:
            return ("High-quality ROE driven by strong operating margins. "
                   "This indicates genuine pricing power and cost efficiency.")
        elif driver == "margin" and quality > 50:
            return ("Moderate-quality ROE driven by margins. "
                   "Acceptable but could improve efficiency.")
        elif driver == "turnover" and quality > 60:
            return ("Good-quality ROE driven by asset efficiency. "
                   "The company generates strong revenue from its asset base.")
        elif driver == "leverage":
            if leverage > 4:
                return (f"Low-quality ROE driven by high financial leverage ({leverage:.1f}x). "
                       f"Returns are amplified by debt — risky if margins compress or rates rise.")
            else:
                return (f"ROE partially driven by leverage ({leverage:.1f}x). "
                       f"Acceptable if debt is manageable and interest coverage is adequate.")
        else:
            return (f"Mixed ROE drivers. Margin {op_margin*100:.1f}%, "
                   f"turnover {turnover:.2f}x, leverage {leverage:.1f}x. "
                   f"Quality is moderate — verify sustainability.")