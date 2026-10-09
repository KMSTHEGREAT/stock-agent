"""
Test V2 Phase 4: Complete workflow.
Run: python test_v2_phase4.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_risk_manager():
    """Test risk manager agent."""
    print("\n" + "=" * 50)
    print("Testing Risk Manager")
    print("=" * 50)
    
    try:
        from src.v2.agents.risk_manager import risk_manager_node
        
        # Create mock state
        state = {
            "symbol": "RELIANCE",
            "technical_output": None,
            "fundamental_output": None,
            "debate_result": None,
            "news_output": None,
        }
        
        result = risk_manager_node(state)
        risk = result.get("risk_assessment")
        
        if risk:
            print(f"  ✓ Risk assessment complete")
            print(f"  Risk Score: {risk.risk_score}/100")
            print(f"  Max Position: {risk.max_position_size_pct:.1f}%")
            print(f"  Volatility: {risk.volatility_risk}")
            print(f"  Liquidity: {risk.liquidity_risk}")
            print(f"  Stop Loss: {risk.suggested_stop_loss}")
            print(f"  Warnings:")
            for w in risk.risk_warnings[:3]:
                print(f"    {w}")
            return True
        else:
            print(f"  ✗ No risk assessment")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_manager():
    """Test portfolio manager agent."""
    print("\n" + "=" * 50)
    print("Testing Portfolio Manager")
    print("=" * 50)
    
    try:
        from src.v2.agents.portfolio_manager import portfolio_manager_node
        from src.v2.graph.state import ResearchThesis, DebateResult, RiskAssessment
        
        # Create mock state with all inputs
        state = {
            "symbol": "TCS",
            "company_name": "Tata Consultancy Services",
            "bull_thesis": ResearchThesis(
                researcher_type="bull",
                symbol="TCS",
                thesis="Strong fundamentals with excellent ROCE and fair valuation.",
                conviction_level=0.7,
            ),
            "bear_thesis": ResearchThesis(
                researcher_type="bear",
                symbol="TCS",
                thesis="IT sector headwinds from global slowdown.",
                conviction_level=0.5,
            ),
            "debate_result": DebateResult(
                bull_thesis="Strong fundamentals",
                bear_thesis="Sector headwinds",
                winner="bull",
                reasoning="Bull argument supported by stronger data",
                sentiment_score=65,
                confidence=0.7,
            ),
            "risk_assessment": RiskAssessment(
                risk_score=35,
                max_position_size_pct=5.0,
                suggested_stop_loss=3100.0,
            ),
            "technical_output": None,
            "fundamental_output": None,
        }
        
        result = portfolio_manager_node(state)
        rec = result.get("final_recommendation")
        
        if rec:
            print(f"  ✓ Portfolio decision made")
            print(f"  Rating: {rec.rating.value.upper()}")
            print(f"  Score: {rec.overall_score}/100")
            print(f"  Conviction: {rec.conviction:.0%}")
            print(f"  Summary: {rec.debate_summary[:150]}...")
            return True
        else:
            print(f"  ✗ No recommendation")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_workflow():
    """Test the complete V2 workflow end-to-end."""
    print("\n" + "=" * 50)
    print("Testing Complete V2 Workflow")
    print("=" * 50)
    
    try:
        from src.v2.graph.workflow import analyze_stock
        
        symbol = "RELIANCE"
        print(f"\n  Running complete V2 analysis for {symbol}...")
        print(f"  (This may take 2-5 minutes)")
        
        start = time.time()
        result = analyze_stock(symbol)
        elapsed = time.time() - start
        
        rec = result.get("final_recommendation")
        
        if rec:
            print(f"\n  ✓ Complete workflow finished in {elapsed:.1f}s")
            print(f"\n  FINAL RECOMMENDATION:")
            print(f"  Rating: {rec.rating.value.upper()}")
            print(f"  Score: {rec.overall_score}/100")
            print(f"  Conviction: {rec.conviction:.0%}")
            
            if rec.entry_price:
                print(f"  Entry: ₹{rec.entry_price:,.2f}")
            if rec.target_price:
                print(f"  Target: ₹{rec.target_price:,.2f}")
            if rec.stop_loss:
                print(f"  Stop Loss: ₹{rec.stop_loss:,.2f}")
            
            # Show debate
            debate = result.get("debate_result")
            if debate:
                print(f"\n  Debate Winner: {debate.winner.upper()}")
                print(f"  Sentiment: {debate.sentiment_score}/100")
            
            # Show bull/bear
            bull = result.get("bull_thesis")
            bear = result.get("bear_thesis")
            if bull:
                print(f"\n  🐂 Bull: {bull.thesis[:100]}...")
            if bear:
                print(f"  🐻 Bear: {bear.thesis[:100]}...")
            
            # Show risk
            risk = result.get("risk_assessment")
            if risk:
                print(f"\n  Risk: {risk.risk_score}/100")
                print(f"  Max Position: {risk.max_position_size_pct:.1f}%")
            
            # Errors
            if result.get("errors"):
                print(f"\n  ⚠ Errors: {result['errors']}")
            
            return True
        else:
            print(f"  ✗ No final recommendation")
            print(f"  Errors: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Phase 4 tests."""
    print("=" * 60)
    print("V2 PHASE 4 TEST: Risk Manager, Portfolio Manager & Workflow")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_risk_manager()
    all_passed &= test_portfolio_manager()
    all_passed &= test_complete_workflow()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 4 TESTS PASSED — V2 COMPLETE!")
        print("\nUsage:")
        print("  python main_v2.py analyze RELIANCE")
        print("  python main_v2.py watchlist RELIANCE TCS INFY")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())