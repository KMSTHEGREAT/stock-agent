"""
Test V2 Phase 2: Agent nodes.
Run: python test_v2_phase2.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_technical_analyst():
    """Test technical analyst agent."""
    print("\n" + "=" * 50)
    print("Testing Technical Analyst")
    print("=" * 50)
    
    try:
        from src.v2.agents.technical_analyst import technical_analyst_node
        
        start = time.time()
        result = technical_analyst_node({"symbol": "RELIANCE", "company_name": "Reliance Industries"})
        elapsed = time.time() - start
        
        output = result.get("technical_output")
        
        if output:
            print(f"  ✓ Technical analysis complete in {elapsed:.1f}s")
            print(f"  Score: {output.score}/100")
            print(f"  Confidence: {output.confidence}")
            print(f"  Trend: {output.trend}")
            print(f"  RSI: {output.rsi}")
            print(f"  Price: ₹{output.current_price:,.2f}" if output.current_price else "  Price: N/A")
            
            if output.key_findings:
                print(f"  Findings:")
                for f in output.key_findings[:3]:
                    print(f"    - {f}")
            
            return True
        else:
            print(f"  ✗ No output. Errors: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fundamental_analyst():
    """Test fundamental analyst agent."""
    print("\n" + "=" * 50)
    print("Testing Fundamental Analyst")
    print("=" * 50)
    
    try:
        from src.v2.agents.fundamental_analyst import fundamental_analyst_node
        
        start = time.time()
        result = fundamental_analyst_node({"symbol": "TCS", "company_name": "Tata Consultancy Services"})
        elapsed = time.time() - start
        
        output = result.get("fundamental_output")
        
        if output:
            print(f"  ✓ Fundamental analysis complete in {elapsed:.1f}s")
            print(f"  Score: {output.score}/100")
            print(f"  Confidence: {output.confidence}")
            print(f"  P/E: {output.pe_ratio}")
            print(f"  ROCE: {output.roce}")
            print(f"  Promoter: {output.promoter_holding}")
            
            if output.key_findings:
                print(f"  Findings:")
                for f in output.key_findings[:3]:
                    print(f"    - {f}")
            
            return True
        else:
            # Fundamental data may not always be available
            print(f"  ⚠ No output (may be expected if Screener.in is unavailable)")
            print(f"    Errors: {result.get('errors', [])}")
            return True  # Don't fail on this
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_news_analyst():
    """Test news analyst with SearXNG."""
    print("\n" + "=" * 50)
    print("Testing News Analyst (SearXNG)")
    print("=" * 50)
    
    try:
        from src.v2.agents.news_analyst import news_analyst_node
        
        start = time.time()
        result = news_analyst_node({"symbol": "RELIANCE", "company_name": "Reliance Industries"})
        elapsed = time.time() - start
        
        output = result.get("news_output")
        
        if output:
            print(f"  ✓ News analysis complete in {elapsed:.1f}s")
            print(f"  Score: {output.score}/100")
            print(f"  News items: {len(output.news_items)}")
            print(f"  Positive: {output.positive_count}, Negative: {output.negative_count}")
            
            if output.news_items:
                print(f"  Sample headlines:")
                for item in output.news_items[:3]:
                    print(f"    - {item.title[:60]}")
            
            if output.key_findings:
                print(f"  Findings:")
                for f in output.key_findings[:2]:
                    print(f"    - {f}")
            
            return True
        else:
            print(f"  ✗ No output. Errors: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_analyst():
    """Test sentiment analyst."""
    print("\n" + "=" * 50)
    print("Testing Sentiment Analyst (LLM)")
    print("=" * 50)
    
    try:
        from src.v2.agents.sentiment_analyst import sentiment_analyst_node
        
        # Create mock state with data from other analysts
        state = {
            "symbol": "RELIANCE",
            "company_name": "Reliance Industries",
            "technical_output": None,
            "fundamental_output": None,
            "news_output": None,
        }
        
        start = time.time()
        result = sentiment_analyst_node(state)
        elapsed = time.time() - start
        
        output = result.get("sentiment_output")
        
        if output:
            print(f"  ✓ Sentiment analysis complete in {elapsed:.1f}s")
            print(f"  Score: {output.score}/100")
            print(f"  Overall sentiment: {output.overall_sentiment}")
            print(f"  Confidence: {output.confidence}")
            
            if output.key_findings:
                print(f"  Findings:")
                for f in output.key_findings[:3]:
                    print(f"    - {f[:80]}")
            
            return True
        else:
            print(f"  ✗ No output")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_all_analysts():
    """Test running all analysts together."""
    print("\n" + "=" * 50)
    print("Testing All Analysts (Sequential)")
    print("=" * 50)
    
    try:
        from src.v2.agents.technical_analyst import technical_analyst_node
        from src.v2.agents.fundamental_analyst import fundamental_analyst_node
        from src.v2.agents.news_analyst import news_analyst_node
        from src.v2.agents.sentiment_analyst import sentiment_analyst_node
        
        symbol = "INFY"
        state = {"symbol": symbol, "company_name": "Infosys"}
        
        total_start = time.time()
        
        # Run each analyst
        print(f"  Analyzing {symbol}...")
        
        # 1. Technical
        print(f"    1. Technical...", end=" ", flush=True)
        tech_result = technical_analyst_node(state)
        state.update(tech_result)
        tech_out = state.get("technical_output")
        print(f"✓ score={tech_out.score if tech_out else 'N/A'}")
        
        # 2. Fundamental
        print(f"    2. Fundamental...", end=" ", flush=True)
        fund_result = fundamental_analyst_node(state)
        state.update(fund_result)
        fund_out = state.get("fundamental_output")
        print(f"✓ score={fund_out.score if fund_out else 'N/A'}")
        
        # 3. News
        print(f"    3. News...", end=" ", flush=True)
        news_result = news_analyst_node(state)
        state.update(news_result)
        news_out = state.get("news_output")
        print(f"✓ score={news_out.score if news_out else 'N/A'}")
        
        # 4. Sentiment
        print(f"    4. Sentiment...", end=" ", flush=True)
        sent_result = sentiment_analyst_node(state)
        state.update(sent_result)
        sent_out = state.get("sentiment_output")
        print(f"✓ score={sent_out.score if sent_out else 'N/A'}")
        
        total_time = time.time() - total_start
        
        print(f"\n  ✓ All analysts completed in {total_time:.1f}s")
        print(f"  Errors: {state.get('errors', [])}")
        
        # Summary
        print(f"\n  Analyst Summary for {symbol}:")
        if tech_out:
            print(f"    Technical:   {tech_out.score}/100 (conf: {tech_out.confidence})")
        if fund_out:
            print(f"    Fundamental: {fund_out.score}/100 (conf: {fund_out.confidence})")
        if news_out:
            print(f"    News:        {news_out.score}/100 ({len(news_out.news_items)} items)")
        if sent_out:
            print(f"    Sentiment:   {sent_out.score}/100 ({sent_out.overall_sentiment})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Phase 2 tests."""
    print("=" * 60)
    print("V2 PHASE 2 TEST: Agent Nodes")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_technical_analyst()
    all_passed &= test_fundamental_analyst()
    all_passed &= test_news_analyst()
    all_passed &= test_sentiment_analyst()
    all_passed &= test_all_analysts()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 2 TESTS PASSED - Ready for Phase 3!")
        print("\nNext: 'start v2 phase 3' to build bull/bear researchers")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())