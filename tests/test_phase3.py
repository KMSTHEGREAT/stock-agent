"""
Test Phase 3: LLM synthesis and scoring engine.
Run: python test_phase3.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_llm_connection():
    """Test basic Ollama connection."""
    print("\n" + "=" * 50)
    print("Testing LLM Connection")
    print("=" * 50)
    
    try:
        import ollama
        import re
        
        response = ollama.chat(
            model='qwen3:4b',
            messages=[
                {'role': 'user', 'content': 'Respond with exactly: "Connection OK"'}
            ],
            options={'num_predict': 100, 'temperature': 0.1}
        )
        
        content = response['message']['content'].strip()
        
        # Remove thinking tags if present
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        print(f"  Response: {content}")
        
        if 'ok' in content.lower() or content:
            print("  ✓ Ollama connection working")
            return True
        else:
            print("  ✗ Empty response")
            return False
            
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


def test_structured_output():
    """Test structured JSON output from Ollama."""
    print("\n" + "=" * 50)
    print("Testing Structured Output")
    print("=" * 50)
    
    try:
        import ollama
        import json
        import re
        
        schema = {
            "type": "object",
            "properties": {
                "stock_name": {"type": "string"},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number"}
            },
            "required": ["stock_name", "sentiment"]
        }
        
        response = ollama.chat(
            model='qwen3:4b',
            messages=[
                {'role': 'user', 'content': 'Analyze Reliance Industries stock. Current price ₹1226. Return JSON with stock_name, sentiment, and confidence.'}
            ],
            format=schema,
            options={'temperature': 0.3, 'num_predict': 200}
        )
        
        content = response['message']['content'].strip()
        
        # Remove thinking tags if present
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        result = json.loads(content)
        
        print(f"  Parsed JSON: {json.dumps(result, indent=2)}")
        
        if 'sentiment' in result:
            print("  ✓ Structured output working")
            return True
        else:
            print("  ✗ Missing required fields")
            return False
            
    except json.JSONDecodeError:
        print("  ✗ Invalid JSON returned")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_llm_synthesis():
    """Test full LLM synthesis with real stock data."""
    print("\n" + "=" * 50)
    print("Testing LLM Synthesis")
    print("=" * 50)
    
    try:
        from src.analysis.technical import TechnicalAnalyzer
        from src.analysis.fundamental import FundamentalAnalyzer
        from src.analysis.technical_scorer import TechnicalScorer
        from src.analysis.fundamental_scorer import FundamentalScorer
        from src.llm.synthesizer import LLMSynthesizer
        from src.schemas.stock_analysis import FundamentalMetrics
        
        tech_analyzer = TechnicalAnalyzer()
        fund_analyzer = FundamentalAnalyzer()
        tech_scorer = TechnicalScorer()
        fund_scorer = FundamentalScorer()
        synthesizer = LLMSynthesizer()
        
        symbol = "TCS"
        print(f"\nFull analysis pipeline for {symbol}...")
        
        # 1. Get metrics
        print("  1. Computing technical metrics...")
        tech_metrics = tech_analyzer.analyze(symbol, period="1y")
        
        print("  2. Computing fundamental metrics...")
        fund_metrics = fund_analyzer.analyze(symbol)
        
        if tech_metrics is None:
            print("  ✗ Technical analysis failed")
            return False
        
        # 2. Score them
        print("  3. Scoring...")
        tech_scores = tech_scorer.score(tech_metrics)
        
        # Handle fund_metrics being None or empty
        if fund_metrics and isinstance(fund_metrics, FundamentalMetrics):
            fund_scores = fund_scorer.score(fund_metrics)
            company_name = symbol  # Use symbol as name
        else:
            # Create empty metrics
            fund_metrics = FundamentalMetrics(symbol=symbol)
            fund_scores = {"total": 50}
            company_name = symbol
        
        # 3. LLM synthesis
        print("  4. LLM synthesis (Qwen 3 4B)...")
        llm_result = synthesizer.synthesize_analysis(
            symbol=symbol,
            company_name=company_name,
            technical_metrics=tech_metrics,
            technical_scores=tech_scores,
            fundamental_metrics=fund_metrics,
            fundamental_scores=fund_scores,
            news_context="",
            market_context=None
        )
        
        if llm_result:
            print("\n  ✓ LLM Analysis Result:")
            print(f"  Summary: {llm_result['analysis_summary'][:200]}")
            print(f"  Confidence: {llm_result.get('confidence_level', 0):.1%}")
            
            if llm_result.get('key_risks'):
                print(f"  Key Risks:")
                for risk in llm_result['key_risks'][:3]:
                    print(f"    - {risk[:80]}")
            
            if llm_result.get('key_catalysts'):
                print(f"  Key Catalysts:")
                for catalyst in llm_result['key_catalysts'][:3]:
                    print(f"    - {catalyst[:80]}")
            return True
        else:
            print("  ✗ LLM synthesis failed")
            return False
            
    except Exception as e:
        print(f"✗ LLM synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scoring_engine():
    """Test the combined scoring engine."""
    print("\n" + "=" * 50)
    print("Testing Scoring Engine")
    print("=" * 50)
    
    try:
        from src.analysis.technical import TechnicalAnalyzer
        from src.analysis.fundamental import FundamentalAnalyzer
        from src.scoring.engine import ScoringEngine
        
        tech_analyzer = TechnicalAnalyzer()
        fund_analyzer = FundamentalAnalyzer()
        scoring = ScoringEngine()
        
        symbol = "RELIANCE"
        print(f"\nAnalyzing {symbol}...")
        
        # Get metrics
        tech_metrics = tech_analyzer.analyze(symbol, period="1y")
        fund_metrics = fund_analyzer.analyze(symbol)
        
        if tech_metrics is None:
            print("✗ Technical analysis failed")
            return False
        
        # Compute score (without news and macro for now)
        scores = scoring.compute_score(
            technical_metrics=tech_metrics,
            fundamental_metrics=fund_metrics or {},
            news_score=50,  # Neutral
            macro_score=50  # Neutral
        )
        
        print(f"\n✓ Scores computed:")
        print(f"  Technical:     {scores['technical_score']}/100")
        print(f"  Fundamental:   {scores['fundamental_score']}/100")
        print(f"  News:          {scores['news_score']}/100 (neutral)")
        print(f"  Macro:         {scores['macro_score']}/100 (neutral)")
        print(f"  ─────────────────────────────")
        print(f"  OVERALL:       {scores['overall_score']}/100")
        print(f"  Recommendation: {scores['recommendation'].value.upper()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Scoring engine error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recommendation_generator():
    """Test recommendation generation."""
    print("\n" + "=" * 50)
    print("Testing Recommendation Generator")
    print("=" * 50)
    
    try:
        from src.analysis.technical import TechnicalAnalyzer
        from src.recommendations.generator import RecommendationGenerator
        from src.schemas.stock_analysis import Recommendation
        
        tech_analyzer = TechnicalAnalyzer()
        rec_gen = RecommendationGenerator()
        
        symbol = "INFY"
        print(f"\nGenerating recommendation for {symbol}...")
        
        metrics = tech_analyzer.analyze(symbol, period="1y")
        if metrics is None:
            print("✗ Analysis failed")
            return False
        
        # Test with different recommendations
        for rec, score in [
            (Recommendation.BUY, 75),
            (Recommendation.HOLD, 55),
            (Recommendation.SELL, 35)
        ]:
            result = rec_gen.generate(
                symbol=symbol,
                company_name=symbol,
                technical_metrics=metrics,
                recommendation=rec,
                overall_score=score
            )
            
            print(f"\n  {rec.value.upper()} (Score: {score}):")
            print(f"    Current Price: ₹{result['current_price']}")
            if result.get('entry_price'):
                print(f"    Entry: ₹{result['entry_price']}")
            if result.get('stop_loss'):
                print(f"    Stop Loss: ₹{result['stop_loss']}")
            if result.get('target_price'):
                print(f"    Target: ₹{result['target_price']}")
            if result.get('risk_reward_ratio'):
                print(f"    Risk/Reward: 1:{result['risk_reward_ratio']}")
            print(f"    Strategy: {result['entry_strategy']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Recommendation generator error: {e}")
        return False


def main():
    """Run Phase 3 tests."""
    print("=" * 50)
    print("PHASE 3 TEST: LLM Synthesis & Scoring")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_llm_connection()
    all_passed &= test_structured_output()
    all_passed &= test_scoring_engine()
    all_passed &= test_llm_synthesis()
    all_passed &= test_recommendation_generator()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ PHASE 3 TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())