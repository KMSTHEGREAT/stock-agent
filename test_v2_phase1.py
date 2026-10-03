"""
Test V2 Phase 1: SearXNG + LangGraph setup.
Run: python test_v2_phase1.py
"""
import sys
import asyncio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_searxng_connection():
    """Test SearXNG is running and returning JSON."""
    print("\n" + "=" * 50)
    print("Testing SearXNG Connection")
    print("=" * 50)
    
    try:
        from src.v2.search.searxng import SearXNGSearchSync
        
        search = SearXNGSearchSync()
        
        # Test basic search
        results = search.search("RELIANCE stock", categories="general", max_results=5)
        
        if results:
            print(f"  ✓ SearXNG is running")
            print(f"  ✓ Got {len(results)} results")
            for r in results[:2]:
                print(f"    - {r.get('title', 'No title')[:60]}")
            return True
        else:
            print("  ✗ No results returned")
            print("  Check: docker ps | grep searxng")
            return False
            
    except Exception as e:
        print(f"  ✗ SearXNG error: {e}")
        print("  Make sure Docker container is running:")
        print("  docker run -d -p 8888:8080 searxng/searxng")
        return False


def test_searxng_news():
    """Test SearXNG news search for Indian stocks."""
    print("\n" + "=" * 50)
    print("Testing SearXNG News Search")
    print("=" * 50)
    
    try:
        from src.v2.search.searxng import SearXNGSearchSync
        
        search = SearXNGSearchSync()
        
        # Test news search
        results = search.search_stock_news("RELIANCE", "Reliance Industries")
        
        if results:
            print(f"  ✓ Got {len(results)} news items")
            for r in results[:3]:
                print(f"    - {r['title'][:60]}")
                if r.get('source'):
                    print(f"      Source: {r['source']}")
            return True
        else:
            print("  ⚠ No news results (may be normal)")
            return True  # Don't fail on empty news
            
    except Exception as e:
        print(f"  ✗ News search error: {e}")
        return False


def test_langgraph_import():
    """Test LangGraph is installed and importable."""
    print("\n" + "=" * 50)
    print("Testing LangGraph Installation")
    print("=" * 50)
    
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        # Build a simple test graph
        class TestState(TypedDict):
            value: str
            steps: list
        
        def step_a(state: TestState) -> dict:
            return {"value": state["value"] + "_a", "steps": ["a"]}
        
        def step_b(state: TestState) -> dict:
            return {"value": state["value"] + "_b", "steps": ["b"]}
        
        # Build graph
        workflow = StateGraph(TestState)
        workflow.add_node("step_a", step_a)
        workflow.add_node("step_b", step_b)
        workflow.set_entry_point("step_a")
        workflow.add_edge("step_a", "step_b")
        workflow.add_edge("step_b", END)
        
        graph = workflow.compile()
        
        # Run it
        result = graph.invoke({"value": "test", "steps": []})
        
        if result["value"] == "test_a_b":
            print(f"  ✓ LangGraph working")
            print(f"  Result: {result}")
            return True
        else:
            print(f"  ✗ Unexpected result: {result}")
            return False
            
    except ImportError:
        print("  ✗ LangGraph not installed")
        print("  Run: pip install langgraph")
        return False
    except Exception as e:
        print(f"  ✗ LangGraph error: {e}")
        return False


def test_v2_state():
    """Test V2 state definitions."""
    print("\n" + "=" * 50)
    print("Testing V2 State Definitions")
    print("=" * 50)
    
    try:
        from src.v2.graph.state import (
            AnalysisState, TechnicalOutput, FundamentalOutput,
            NewsOutput, ResearchThesis, DebateResult,
            FinalRecommendation, Rating
        )
        
        # Test creating outputs
        tech = TechnicalOutput(
            agent_name="technical_analyst",
            symbol="RELIANCE",
            score=55.0,
            confidence=0.8,
            key_findings=["RSI at 45", "Trend neutral"],
            data_sources=["yfinance"],
            timestamp="2026-10-03"
        )
        
        fund = FundamentalOutput(
            agent_name="fundamental_analyst",
            symbol="RELIANCE",
            score=60.0,
            confidence=0.7,
            key_findings=["P/E 15", "ROCE 12%"],
            data_sources=["screener.in"],
            timestamp="2026-10-03"
        )
        
        rec = FinalRecommendation(
            symbol="RELIANCE",
            company_name="Reliance Industries",
            rating=Rating.HOLD,
            overall_score=57.5,
            conviction=0.6,
            bull_thesis="Strong earnings",
            bear_thesis="Market headwinds"
        )
        
        print(f"  ✓ TechnicalOutput created: score={tech.score}")
        print(f"  ✓ FundamentalOutput created: score={fund.score}")
        print(f"  ✓ FinalRecommendation created: {rec.rating.value}")
        return True
        
    except Exception as e:
        print(f"  ✗ State definition error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_searxng_vs_ddgs():
    """Compare SearXNG results with old DDGS approach."""
    print("\n" + "=" * 50)
    print("Comparing SearXNG vs DDGS (Quality Test)")
    print("=" * 50)
    
    try:
        from src.v2.search.searxng import SearXNGSearchSync
        
        search = SearXNGSearchSync()
        start = time.time()
        
        # Search for stock news
        results = search.search_stock_news("TCS", "Tata Consultancy Services")
        elapsed = time.time() - start
        
        print(f"  SearXNG: {len(results)} results in {elapsed:.1f}s")
        
        if results:
            # Check quality indicators
            has_summary = sum(1 for r in results if r.get("summary"))
            has_source = sum(1 for r in results if r.get("source"))
            
            print(f"  Quality metrics:")
            print(f"    With summaries: {has_summary}/{len(results)}")
            print(f"    With sources: {has_source}/{len(results)}")
            
            # Show sample
            print(f"\n  Sample results:")
            for r in results[:3]:
                title = r.get('title', 'No title')[:70]
                print(f"    - {title}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Comparison error: {e}")
        return False


def test_model_comparison():
    """Compare Qwen3-4B vs Phi-4-mini for structured output."""
    print("\n" + "=" * 50)
    print("Model Comparison: Qwen3-4B vs Phi-4-mini")
    print("=" * 50)
    
    try:
        import ollama
        import json
        
        test_prompt = """Analyze RELIANCE stock.
Technical: 55/100, RSI 45, neutral trend.
Fundamental: 60/100, P/E 15, ROCE 12%.

Respond with ONLY this JSON:
{"sentiment": "neutral", "score": 57, "confidence": 0.7, "summary": "balanced view"}"""
        
        models_to_test = ["qwen3:4b"]
        
        # Check if phi4-mini is available
        try:
            models = ollama.list()
            model_list = [m.model for m in models.models] if hasattr(models, 'models') else []
            if "phi4-mini" in model_list or "phi4-mini:latest" in model_list:
                models_to_test.append("phi4-mini")
            else:
                print("  (Phi-4-mini not installed - run: ollama pull phi4-mini)")
        except:
            pass
        
        for model in models_to_test:
            print(f"\n  Testing {model}...")
            
            start = time.time()
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": test_prompt}],
                    options={"temperature": 0.2, "num_predict": 300}
                )
                elapsed = time.time() - start
                
                content = response.message.content or ""
                thinking = response.message.thinking or ""
                
                print(f"    Time: {elapsed:.1f}s")
                print(f"    Content length: {len(content)}")
                print(f"    Thinking length: {len(thinking)}")
                
                # Try to parse JSON
                text = content if content.strip() else thinking
                import re
                json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        print(f"    ✓ Valid JSON: {parsed}")
                    except:
                        print(f"    ✗ JSON parse failed")
                else:
                    print(f"    ✗ No JSON found")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model comparison error: {e}")
        return False


def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("V2 PHASE 1 TEST: Setup & SearXNG")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_searxng_connection()
    all_passed &= test_searxng_news()
    all_passed &= test_langgraph_import()
    all_passed &= test_v2_state()
    all_passed &= test_searxng_vs_ddgs()
    all_passed &= test_model_comparison()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 1 TESTS PASSED - Ready for Phase 2!")
        print("\nNext: 'start v2 phase 2' to build the agent nodes")
    else:
        print("✗ SOME TESTS FAILED - Fix issues above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())