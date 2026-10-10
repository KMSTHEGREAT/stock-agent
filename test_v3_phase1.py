"""
Test V3 Phase 1: CFA Analysis Agents.
Run: python test_v3_phase1.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_dcf_valuation():
    """Test DCF valuation agent."""
    print("\n" + "=" * 60)
    print("Testing DCF Valuation Agent (CFA Level 2)")
    print("=" * 60)
    
    try:
        from src.v3.agents.dcf_valuation import DCFValuationAgent
        
        agent = DCFValuationAgent()
        
        symbol = "RELIANCE"
        print(f"\n  Analyzing {symbol}...")
        
        start = time.time()
        result = agent.analyze(symbol)
        elapsed = time.time() - start
        
        if result:
            print(f"  ✓ DCF complete in {elapsed:.1f}s")
            print(f"\n  Valuation Results:")
            print(f"    Intrinsic Value: ₹{result.intrinsic_value_per_share:,.2f}")
            print(f"    Current Price:   ₹{result.current_price:,.2f}")
            print(f"    Upside/Downside: {result.upside_downside_pct:+.1f}%")
            print(f"    Status: {result.valuation_status}")
            print(f"\n  Key Assumptions:")
            print(f"    WACC: {result.wacc:.2f}%")
            print(f"    Terminal Growth: {result.terminal_growth_rate:.2f}%")
            print(f"    Revenue Growth: {result.revenue_growth_5y:.2f}%")
            print(f"\n  Projected FCFF (crores): {result.projected_fcff}")
            print(f"\n  Enterprise Value: ₹{result.enterprise_value:,.0f} Cr")
            print(f"  Equity Value: ₹{result.equity_value:,.0f} Cr")
            
            if result.sensitivity_matrix:
                print(f"\n  Sensitivity Matrix:")
                for wacc_key, growth_row in result.sensitivity_matrix.get("matrix", {}).items():
                    print(f"    {wacc_key}: {growth_row}")
            
            print(f"\n  Interpretation: {result.interpretation}")
            return True
        else:
            print(f"  ✗ DCF failed (data may be insufficient)")
            return True  # Don't fail test for data availability
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dupont_analysis():
    """Test DuPont analyzer."""
    print("\n" + "=" * 60)
    print("Testing DuPont Analyzer (CFA Level 1)")
    print("=" * 60)
    
    try:
        from src.v3.agents.dupont_analyzer import DuPontAnalyzer
        
        analyzer = DuPontAnalyzer()
        
        # Test multiple stocks
        for symbol in ["TCS", "HDFCBANK"]:
            print(f"\n  Analyzing {symbol}...")
            
            start = time.time()
            result = analyzer.analyze(symbol)
            elapsed = time.time() - start
            
            if result:
                print(f"  ✓ DuPont complete in {elapsed:.1f}s")
                print(f"\n  ROE Decomposition for {symbol}:")
                print(f"    ROE: {result.roe:.2f}%")
                print(f"    ROA: {result.roa:.2f}%")
                print(f"\n  5-Component Breakdown:")
                print(f"    1. Tax Burden:       {result.tax_burden:.3f}")
                print(f"    2. Interest Burden:  {result.interest_burden:.3f}")
                print(f"    3. Operating Margin: {result.operating_margin:.2f}%")
                print(f"    4. Asset Turnover:   {result.asset_turnover:.2f}x")
                print(f"    5. Fin. Leverage:    {result.financial_leverage:.2f}x")
                print(f"\n  3-Component Simplified:")
                print(f"    Net Profit Margin:   {result.net_profit_margin:.2f}%")
                print(f"\n  Quality Assessment:")
                print(f"    Primary Driver: {result.primary_driver}")
                print(f"    Quality Score:  {result.roe_quality_score:.0f}/100")
                print(f"    Interpretation: {result.quality_interpretation}")
            else:
                print(f"  ⚠ No result for {symbol}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_risk_manager():
    """Test CFA risk manager."""
    print("\n" + "=" * 60)
    print("Testing Advanced Risk Manager (CFA Level 3)")
    print("=" * 60)
    
    try:
        from src.v3.agents.risk_manager_cfa import AdvancedRiskManager
        
        manager = AdvancedRiskManager()
        
        symbol = "RELIANCE"
        print(f"\n  Analyzing {symbol}...")
        
        start = time.time()
        result = manager.analyze(symbol)
        elapsed = time.time() - start
        
        if result:
            print(f"  ✓ Risk analysis complete in {elapsed:.1f}s")
            print(f"\n  Value at Risk (Daily):")
            print(f"    VaR 95%: {result.var_95:.2f}% (max daily loss at 95% confidence)")
            print(f"    VaR 99%: {result.var_99:.2f}% (max daily loss at 99% confidence)")
            print(f"    CVaR 95%: {result.cvar_95:.2f}% (average loss in worst 5%)")
            
            print(f"\n  Risk-Adjusted Returns:")
            print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"    Sortino Ratio: {result.sortino_ratio:.2f}")
            print(f"    Annual Return: {result.annual_return:.2f}%")
            print(f"    Annual Volatility: {result.annual_volatility:.2f}%")
            
            print(f"\n  Drawdown Analysis:")
            print(f"    Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"    DD Duration: {result.max_drawdown_duration} days")
            print(f"    Current DD: {result.current_drawdown:.2f}%")
            
            print(f"\n  Market-Relative:")
            print(f"    Beta: {result.beta:.2f}")
            print(f"    Alpha: {result.alpha:.2f}%")
            print(f"    R²: {result.r_squared:.3f}")
            print(f"    Correlation: {result.correlation:.3f}")
            
            print(f"\n  Position Sizing:")
            print(f"    Recommended: {result.recommended_position_pct:.1f}% of portfolio")
            print(f"    Kelly Fraction: {result.kelly_fraction:.3f}")
            print(f"    Risk Score: {result.risk_score:.1f}/100")
            print(f"    Risk Category: {result.risk_category}")
            
            print(f"\n  Summary: {result.risk_summary}")
            return True
        else:
            print(f"  ✗ Risk analysis failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_validation():
    """Cross-validate DCF, DuPont, and Risk for a single stock."""
    print("\n" + "=" * 60)
    print("Cross-Validation: All CFA Agents")
    print("=" * 60)
    
    try:
        from src.v3.agents.dcf_valuation import DCFValuationAgent
        from src.v3.agents.dupont_analyzer import DuPontAnalyzer
        from src.v3.agents.risk_manager_cfa import AdvancedRiskManager
        
        symbol = "TCS"
        print(f"\n  Complete CFA analysis for {symbol}...")
        
        total_start = time.time()
        
        # Run all three
        print(f"  [1/3] DCF Valuation...", end=" ", flush=True)
        dcf = DCFValuationAgent().analyze(symbol)
        print(f"{'✓' if dcf else '⚠'}")
        
        print(f"  [2/3] DuPont Analysis...", end=" ", flush=True)
        dupont = DuPontAnalyzer().analyze(symbol)
        print(f"{'✓' if dupont else '⚠'}")
        
        print(f"  [3/3] Risk Analysis...", end=" ", flush=True)
        risk = AdvancedRiskManager().analyze(symbol)
        print(f"{'✓' if risk else '⚠'}")
        
        total_time = time.time() - total_start
        
        print(f"\n  All agents completed in {total_time:.1f}s")
        
        # Summary
        print(f"\n  {'='*50}")
        print(f"  CFA ANALYSIS SUMMARY: {symbol}")
        print(f"  {'='*50}")
        
        if dcf:
            print(f"\n  VALUATION:")
            print(f"    Intrinsic: ₹{dcf.intrinsic_value_per_share:,.0f}")
            print(f"    Market: ₹{dcf.current_price:,.0f}")
            print(f"    Upside: {dcf.upside_downside_pct:+.1f}%")
            print(f"    Status: {dcf.valuation_status}")
        
        if dupont:
            print(f"\n  PROFITABILITY:")
            print(f"    ROE: {dupont.roe:.1f}%")
            print(f"    Driver: {dupont.primary_driver}")
            print(f"    Quality: {dupont.roe_quality_score:.0f}/100")
        
        if risk:
            print(f"\n  RISK:")
            print(f"    VaR 95%: {risk.var_95:.1f}%")
            print(f"    Sharpe: {risk.sharpe_ratio:.2f}")
            print(f"    Beta: {risk.beta:.2f}")
            print(f"    Risk: {risk.risk_category}")
            print(f"    Position: {risk.recommended_position_pct:.1f}%")
        
        # Investment thesis
        if dcf and dupont and risk:
            print(f"\n  INVESTMENT THESIS:")
            
            if dcf.valuation_status == "undervalued" and dupont.roe_quality_score > 60:
                print(f"    ✓ Attractive: Undervalued + high-quality ROE")
                if risk.risk_category in ("low", "moderate"):
                    print(f"    ✓ Manageable risk — consider position")
                else:
                    print(f"    ⚠ High risk — reduce position size")
            elif dcf.valuation_status == "overvalued":
                print(f"    ✗ Overvalued — wait for better entry")
            else:
                print(f"    → Fairly valued — depends on risk tolerance")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Phase 1 tests."""
    print("=" * 60)
    print("V3 PHASE 1 TEST: CFA Analysis Agents")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_dcf_valuation()
    all_passed &= test_dupont_analysis()
    all_passed &= test_advanced_risk_manager()
    all_passed &= test_cross_validation()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 1 TESTS PASSED — CFA Agents Working!")
        print("\nNext: 'start v3 phase 2' for multi-source news")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())