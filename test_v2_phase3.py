"""
Test V2 Phase 3: Bull/Bear Researchers & Debate.
Run: python test_v2_phase3.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_bull_researcher():
    """Test bull researcher agent."""
    print("\n" + "=" * 50)
    print("Testing Bull Researcher")
    print("=" * 50)
    
    try:
        from src.v2.agents.bull_researcher import bull_researcher_node
        
        # Create state with mock analyst data
        state = {
            "symbol": "RELIANCE",
            "company_name": "Reliance Industries",
            "technical_output": None,
            "fundamental_output": None,
            "news_output": None,
            "sentiment_output": None,
        }
        
        start = time.time()
        result = bull_researcher_node(state)
        elapsed = time.time() - start
        
        thesis = result.get("bull_thesis")
        
        if thesis:
            print(f"  ✓ Bull thesis generated in {elapsed:.1f}s")
            print(f"  Conviction: {thesis.conviction_level:.0%}")
            print(f"  Thesis: {thesis.thesis[:150]}...")
            print(f"  Evidence:")
            for e in thesis.supporting_evidence[:3]:
                print(f"    + {e[:80]}")
            return True
        else:
            print(f"  ✗ No thesis. Errors: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bear_researcher():
    """Test bear researcher agent."""
    print("\n" + "=" * 50)
    print("Testing Bear Researcher")
    print("=" * 50)
    
    try:
        from src.v2.agents.bear_researcher import bear_researcher_node
        
        state = {
            "symbol": "RELIANCE",
            "company_name": "Reliance Industries",
            "technical_output": None,
            "fundamental_output": None,
            "news_output": None,
            "sentiment_output": None,
        }
        
        start = time.time()
        result = bear_researcher_node(state)
        elapsed = time.time() - start
        
        thesis = result.get("bear_thesis")
        
        if thesis:
            print(f"  ✓ Bear thesis generated in {elapsed:.1f}s")
            print(f"  Conviction: {thesis.conviction_level:.0%}")
            print(f"  Thesis: {thesis.thesis[:150]}...")
            print(f"  Evidence:")
            for e in thesis.supporting_evidence[:3]:
                print(f"    - {e[:80]}")
            return True
        else:
            print(f"  ✗ No thesis. Errors: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_debate_moderator():
    """Test debate moderator agent."""
    print("\n" + "=" * 50)
    print("Testing Debate Moderator")
    print("=" * 50)
    
    try:
        from src.v2.agents.debate_moderator import debate_moderator_node
        from src.v2.graph.state import ResearchThesis
        
        # Create state with bull and bear theses
        state = {
            "symbol": "TCS",
            "company_name": "Tata Consultancy Services",
            "bull_thesis": ResearchThesis(
                researcher_type="bull",
                symbol="TCS",
                thesis="TCS has strong fundamentals with ROCE 63%, low debt, and consistent dividend. IT sector recovery expected with digital transformation demand.",
                supporting_evidence=["ROCE 63% - excellent capital efficiency", 
                                   "Promoter holding 72% - strong commitment",
                                   "P/E 15 - fair valuation"],
                conviction_level=0.7,
            ),
            "bear_thesis": ResearchThesis(
                researcher_type="bear",
                symbol="TCS",
                thesis="IT sector faces headwinds from global slowdown and AI disruption. Wage inflation compressing margins. Client spending cuts likely.",
                supporting_evidence=["Global IT spending slowdown",
                                   "AI disruption threat to services",
                                   "Margin pressure from wage inflation"],
                conviction_level=0.5,
            ),
            "technical_output": None,
            "fundamental_output": None,
            "news_output": None,
        }
        
        start = time.time()
        result = debate_moderator_node(state)
        elapsed = time.time() - start
        
        debate = result.get("debate_result")
        
        if debate:
            print(f"  ✓ Debate moderated in {elapsed:.1f}s")
            print(f"  Winner: {debate.winner}")
            print(f"  Sentiment Score: {debate.sentiment_score}/100")
            print(f"  Confidence: {debate.confidence:.0%}")
            print(f"  Reasoning: {debate.reasoning[:150]}...")
            return True
        else:
            print(f"  ✗ No debate result")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_debate_pipeline():
    """Test bull + bear + debate with real analyst data."""
    print("\n" + "=" * 50)
    print("Testing Full Debate Pipeline (Analysts → Researchers → Moderator)")
    print("=" * 50)
    
    try:
        from src.v2.agents.technical_analyst import technical_analyst_node
        from src.v2.agents.fundamental_analyst import fundamental_analyst_node
        from src.v2.agents.news_analyst import news_analyst_node
        from src.v2.agents.bull_researcher import bull_researcher_node
        from src.v2.agents.bear_researcher import bear_researcher_node
        from src.v2.agents.debate_moderator import debate_moderator_node
        
        symbol = "RELIANCE"
        state = {"symbol": symbol, "company_name": "Reliance Industries"}
        
        total_start = time.time()
        
        # Step 1: Run analysts
        print(f"  [1/3] Running analysts for {symbol}...")
        
        print(f"    Technical...", end=" ", flush=True)
        state.update(technical_analyst_node(state))
        print("✓")
        
        print(f"    Fundamental...", end=" ", flush=True)
        state.update(fundamental_analyst_node(state))
        print("✓")
        
        print(f"    News...", end=" ", flush=True)
        state.update(news_analyst_node(state))
        print("✓")
        
        # Step 2: Run researchers
        print(f"\n  [2/3] Running researchers...")
        
        print(f"    Bull researcher...", end=" ", flush=True)
        state.update(bull_researcher_node(state))
        bull = state.get("bull_thesis")
        print(f"✓ conviction={bull.conviction_level:.0%}" if bull else "✗")
        
        print(f"    Bear researcher...", end=" ", flush=True)
        state.update(bear_researcher_node(state))
        bear = state.get("bear_thesis")
        print(f"✓ conviction={bear.conviction_level:.0%}" if bear else "✗")
        
        # Step 3: Run debate moderator
        print(f"\n  [3/3] Moderating debate...")
        
        print(f"    Debate moderator...", end=" ", flush=True)
        state.update(debate_moderator_node(state))
        debate = state.get("debate_result")
        print(f"✓ winner={debate.winner}" if debate else "✗")
        
        total_time = time.time() - total_start
        
        # Display results
        print(f"\n  {'='*50}")
        print(f"  DEBATE RESULTS FOR {symbol}")
        print(f"  {'='*50}")
        
        if debate:
            print(f"\n  Winner: {debate.winner.upper()}")
            print(f"  Sentiment: {debate.sentiment_score}/100")
            print(f"  Confidence: {debate.confidence:.0%}")
            print(f"\n  Reasoning: {debate.reasoning}")
            
            if bull:
                print(f"\n  🐂 BULL THESIS:")
                print(f"     {bull.thesis[:200]}")
                print(f"     Conviction: {bull.conviction_level:.0%}")
            
            if bear:
                print(f"\n  🐻 BEAR THESIS:")
                print(f"     {bear.thesis[:200]}")
                print(f"     Conviction: {bear.conviction_level:.0%}")
        
        print(f"\n  Total time: {total_time:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Phase 3 tests."""
    print("=" * 60)
    print("V2 PHASE 3 TEST: Bull/Bear Researchers & Debate")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_bull_researcher()
    all_passed &= test_bear_researcher()
    all_passed &= test_debate_moderator()
    all_passed &= test_full_debate_pipeline()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 3 TESTS PASSED - Ready for Phase 4!")
        print("\nNext: 'start v2 phase 4' to build risk manager & portfolio manager")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())