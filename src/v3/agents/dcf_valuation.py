"""
DCF Valuation Agent — CFA Level 2.
Fixed for Indian stocks where Yahoo Finance info dict may be incomplete.
"""
import yfinance as yf
import numpy as np
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DCFResult:
    symbol: str
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: float
    current_price: float
    upside_downside_pct: float
    wacc: float
    terminal_growth_rate: float
    revenue_growth_5y: float
    projected_fcff: List[float]
    sensitivity_matrix: Optional[Dict] = None
    valuation_status: str = "fairly_valued"
    data_quality: str = "good"  # good/partial/estimated

    @property
    def interpretation(self) -> str:
        if self.valuation_status == "undervalued":
            return (f"Undervalued by {abs(self.upside_downside_pct):.0f}%. "
                   f"Intrinsic ₹{self.intrinsic_value_per_share:,.0f} vs "
                   f"market ₹{self.current_price:,.0f}.")
        elif self.valuation_status == "overvalued":
            return (f"Overvalued by {abs(self.upside_downside_pct):.0f}%. "
                   f"Intrinsic ₹{self.intrinsic_value_per_share:,.0f} vs "
                   f"market ₹{self.current_price:,.0f}.")
        return (f"Fairly valued. Intrinsic ₹{self.intrinsic_value_per_share:,.0f} vs "
               f"market ₹{self.current_price:,.0f}.")


class DCFValuationAgent:
    RISK_FREE_RATE = 0.07
    MARKET_RETURN = 0.12
    CORPORATE_TAX_RATE = 0.251
    TERMINAL_GROWTH = 0.025
    MIN_WACC = 0.08
    MAX_WACC = 0.20

    def analyze(self, symbol: str) -> Optional[DCFResult]:
        logger.info(f"[DCF Agent] Valuing {symbol}")

        data = self._fetch_financial_data(symbol)
        if not data:
            return None

        fcff_projections = self._project_fcff(data)
        if not fcff_projections:
            return None

        wacc = self._calculate_wacc(data)
        terminal_value = self._calculate_terminal_value(fcff_projections, wacc, data)
        ev = self._calculate_enterprise_value(fcff_projections, terminal_value, wacc)
        equity_value = self._calculate_equity_value(ev, data)

        shares = data.get("shares_outstanding", 0)
        if shares == 0:
            return None

        intrinsic_per_share = equity_value / shares
        current_price = data.get("current_price", 0)
        if current_price <= 0:
            return None

        upside_pct = ((intrinsic_per_share - current_price) / current_price) * 100

        if upside_pct > 15:
            status = "undervalued"
        elif upside_pct < -15:
            status = "overvalued"
        else:
            status = "fairly_valued"

        return DCFResult(
            symbol=symbol,
            enterprise_value=round(ev / 1e7, 1),
            equity_value=round(equity_value / 1e7, 1),
            intrinsic_value_per_share=round(intrinsic_per_share, 2),
            current_price=round(current_price, 2),
            upside_downside_pct=round(upside_pct, 1),
            wacc=round(wacc * 100, 2),
            terminal_growth_rate=round(self.TERMINAL_GROWTH * 100, 2),
            revenue_growth_5y=round(data.get("growth_5y", 0.08) * 100, 2),
            projected_fcff=[round(f / 1e7, 1) for f in fcff_projections],
            valuation_status=status,
            data_quality=data.get("data_quality", "good"),
        )

    def _fetch_financial_data(self, symbol: str) -> Optional[Dict]:
        """Fetch financial data with multiple fallbacks for Indian stocks."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info

            # Get shares and price (critical)
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

            if not shares or not price:
                # Try from fast_info
                try:
                    price = ticker.fast_info.last_price
                except:
                    pass

                if not shares:
                    shares = info.get("marketCap", 0) / price if price and info.get("marketCap") else 0

            if not shares or not price:
                logger.warning(f"[DCF Agent] No shares/price for {symbol}")
                return None

            # Get revenue — try multiple sources
            revenue = info.get("totalRevenue")
            if not revenue:
                # Try from financials
                try:
                    financials = ticker.financials
                    if financials is not None and not financials.empty:
                        if "Total Revenue" in financials.index:
                            revenue = financials.loc["Total Revenue"].iloc[0]
                except:
                    pass

            if not revenue:
                logger.warning(f"[DCF Agent] No revenue for {symbol}")
                return None

            # Get EBIT — multiple fallbacks
            ebit = (info.get("operatingIncome") or 
                   info.get("ebit") or
                   info.get("operatingProfit"))

            if not ebit:
                # Try from financials
                try:
                    financials = ticker.financials
                    if financials is not None and not financials.empty:
                        if "Operating Income" in financials.index:
                            ebit = financials.loc["Operating Income"].iloc[0]
                        elif "EBIT" in financials.index:
                            ebit = financials.loc["EBIT"].iloc[0]
                except:
                    pass

            if not ebit:
                # Estimate from EBITDA
                ebitda = info.get("ebitda")
                if ebitda:
                    # Rough estimate: EBIT ≈ EBITDA * 0.85 (assuming ~15% D&A)
                    ebit = ebitda * 0.85
                    data_quality = "estimated"

            if not ebit:
                # Last resort: estimate from revenue
                # Indian average operating margin ~15%
                ebit = revenue * 0.15
                data_quality = "estimated"
                logger.info(f"[DCF Agent] EBIT estimated from revenue for {symbol}")
            else:
                data_quality = "good"

            # Get debt and cash
            total_debt = info.get("totalDebt", 0)
            if not total_debt:
                try:
                    balance = ticker.balance_sheet
                    if balance is not None and not balance.empty:
                        if "Total Debt" in balance.index:
                            total_debt = balance.loc["Total Debt"].iloc[0]
                except:
                    pass
                if not total_debt:
                    total_debt = 0

            cash = info.get("cash", 0) or info.get("totalCash", 0) or 0

            # Beta
            beta = info.get("beta") or 1.0
            if beta is None or beta == 0:
                beta = 1.0

            # Historical growth
            growth = self._calculate_historical_growth(ticker, info, revenue)

            return {
                "revenue": revenue,
                "operating_income": ebit,
                "ebitda": info.get("ebitda", ebit * 1.18),
                "total_debt": total_debt,
                "cash": cash,
                "shares_outstanding": shares,
                "current_price": price,
                "beta": beta,
                "growth_5y": growth,
                "data_quality": data_quality,
            }

        except Exception as e:
            logger.error(f"[DCF Agent] Data fetch error: {e}")
            return None

    def _calculate_historical_growth(self, ticker, info, current_revenue) -> float:
        """Calculate historical growth with fallbacks."""
        try:
            financials = ticker.financials
            if financials is not None and not financials.empty:
                if "Total Revenue" in financials.index:
                    revenues = financials.loc["Total Revenue"]
                    if len(revenues) >= 3:
                        latest = revenues.iloc[0]
                        oldest = revenues.iloc[-2]
                        if oldest > 0 and latest > 0:
                            years = len(revenues) - 1
                            cagr = (latest / oldest) ** (1/years) - 1
                            return max(0.0, min(0.20, cagr))

            # Fallback to info
            rev_growth = info.get("revenueGrowth", 0.08)
            if rev_growth and 0 < rev_growth < 0.5:
                return max(0.03, min(0.20, rev_growth))

            return 0.08
        except:
            return 0.08

    def _project_fcff(self, data: Dict) -> Optional[List[float]]:
        ebit = data.get("operating_income", 0)
        if ebit <= 0:
            return None

        tax = self.CORPORATE_TAX_RATE
        growth = data.get("growth_5y", 0.08)
        reinvestment = min(0.50, 0.20 + growth * 1.2)

        nopat = ebit * (1 - tax)
        projections = []
        current = nopat

        for year in range(5):
            fade = 1.0 - (year * 0.15)
            year_growth = growth * fade + self.TERMINAL_GROWTH * (1 - fade)
            fcff = current * (1 + year_growth) * (1 - reinvestment)
            projections.append(fcff)
            current = current * (1 + year_growth)

        return projections

    def _calculate_wacc(self, data: Dict) -> float:
        rf = self.RISK_FREE_RATE
        beta = data.get("beta", 1.0)
        rm = self.MARKET_RETURN
        cost_equity = rf + beta * (rm - rf)
        cost_debt = rf + 0.025

        equity_value = data.get("current_price", 0) * data.get("shares_outstanding", 0)
        debt_value = data.get("total_debt", 0)
        total = equity_value + debt_value

        if total <= 0:
            return cost_equity

        w_e = equity_value / total
        w_d = debt_value / total
        wacc = w_e * cost_equity + w_d * cost_debt * (1 - self.CORPORATE_TAX_RATE)

        return max(self.MIN_WACC, min(self.MAX_WACC, wacc))

    def _calculate_terminal_value(self, projections, wacc, data) -> float:
        terminal_g = self.TERMINAL_GROWTH

        if wacc <= terminal_g + 0.01:
            ebitda = data.get("ebitda", 0)
            return ebitda * 12 if ebitda > 0 else 0

        final_fcff = projections[-1]
        fcff_n1 = final_fcff * (1 + terminal_g)
        return fcff_n1 / (wacc - terminal_g)

    def _calculate_enterprise_value(self, projections, terminal_value, wacc) -> float:
        pv_fcff = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(projections))
        pv_tv = terminal_value / (1 + wacc) ** len(projections)
        return pv_fcff + pv_tv

    def _calculate_equity_value(self, ev, data) -> float:
        return ev - data.get("total_debt", 0) + data.get("cash", 0)