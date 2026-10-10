"""
Test V3 Phase 3: Self-Improvement Engine.
Run: python test_v3_phase3.py
"""
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))


def test_prediction_tracking():
    """Test recording predictions and outcomes."""
    print("\n" + "=" * 50)
    print("Testing Prediction Tracking")
    print("=" * 50)
    
    try:
        from src.v3.selfimprove.engine import SelfImprovementEngine
        
        engine = SelfImprovementEngine()
        
        # Record several predictions
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
        ratings = ["HOLD", "OVERWEIGHT", "HOLD", "HOLD", "SELL"]
        scores = [53.4, 63.2, 58.0, 52.6, 35.0]
        
        print("  Recording predictions...")
        for symbol, rating, score in zip(symbols, ratings, scores):
            pred_id = engine.record_prediction(
                symbol=symbol,
                rating=rating,
                score=score,
                target=random.uniform(1000, 1500),
                conviction=random.uniform(0.4, 0.8),
            )
            print(f"    {symbol}: {rating} ({score}) → ID {pred_id}")
        
        # Simulate outcomes (some correct, some wrong)
        print("\n  Recording outcomes...")
        outcomes = [
            {"symbol": "RELIANCE", "return_1w": 0.015},   # +1.5% (predicted HOLD=neutral, actual=neutral → correct)
            {"symbol": "TCS", "return_1w": 0.035},        # +3.5% (predicted OVERWEIGHT=bullish, actual=bullish → correct)
            {"symbol": "INFY", "return_1w": -0.025},       # -2.5% (predicted HOLD=neutral, actual=bearish → neutral error)
            {"symbol": "HDFCBANK", "return_1w": -0.005},   # -0.5% (predicted HOLD, actual=neutral → correct)
            {"symbol": "ITC", "return_1w": 0.01},          # +1% (predicted SELL=bearish, actual=neutral → over_pessimistic)
        ]
        
        engine.batch_update_outcomes(outcomes)
        print("    Outcomes recorded")
        
        # Check metrics
        metrics = engine.get_performance_metrics()
        
        print(f"\n  Performance Metrics:")
        print(f"    Total evaluated: {metrics['total_evaluated']}")
        print(f"    Correct: {metrics['total_correct']}")
        print(f"    Accuracy: {metrics['overall_accuracy_pct']}%")
        print(f"    Error breakdown: {metrics['error_breakdown']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_suggestion_generation():
    """Test generating improvement suggestions."""
    print("\n" + "=" * 50)
    print("Testing Improvement Suggestions")
    print("=" * 50)
    
    try:
        from src.v3.selfimprove.engine import SelfImprovementEngine
        
        engine = SelfImprovementEngine()
        
        # Generate suggestions from current data
        suggestions = engine.analyze_and_suggest()
        
        print(f"  Generated {len(suggestions)} suggestions")
        
        if suggestions:
            for i, s in enumerate(suggestions[:3], 1):
                print(f"\n  Suggestion {i}:")
                print(f"    Type: {s.improvement_type}")
                print(f"    Target: {s.target}")
                print(f"    Description: {s.description[:100]}")
                print(f"    Reason: {s.reason}")
                print(f"    Confidence: {s.confidence:.0%}")
        else:
            print("  (Not enough data yet — need 10+ evaluated predictions)")
            print("  This is expected for a fresh system")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_self_healing():
    """Test self-healing on errors."""
    print("\n" + "=" * 50)
    print("Testing Self-Healing")
    print("=" * 50)
    
    try:
        from src.v3.selfimprove.engine import SelfImprovementEngine
        
        engine = SelfImprovementEngine()
        
        # Test with different error types
        test_errors = [
            (TimeoutError("Connection timeout to Ollama"), "Network timeout"),
            (ValueError("Invalid JSON: {incomplete"), "JSON parse error"),
            (KeyError("'operatingIncome'"), "Missing data field"),
            (Exception("Unknown error occurred"), "Unknown error"),
        ]
        
        for error, description in test_errors:
            print(f"\n  Testing: {description}")
            
            result = engine.handle_error(
                error, 
                context={"symbol": "TEST", "agent": "test_agent"}
            )
            
            print(f"    Healed: {result.get('healed', False)}")
            if result.get('diagnosis'):
                diag = result['diagnosis']
                print(f"    Category: {diag.get('category')}")
                print(f"    Cause: {diag.get('cause', 'N/A')}")
            if result.get('fix_applied'):
                print(f"    Fix: {result['fix_applied'].get('description', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_guardrails():
    """Test guardrail system."""
    print("\n" + "=" * 50)
    print("Testing Guardrails")
    print("=" * 50)
    
    try:
        from src.v3.selfimprove.engine import SelfImprovementEngine
        
        engine = SelfImprovementEngine()
        
        # Add some guardrails
        engine.add_guardrail(
            rule="Never recommend position > 10% for single stock",
            trigger="position_size > 10",
            action="block"
        )
        
        engine.add_guardrail(
            rule="Require promoter pledging < 20% before BUY rating",
            trigger="pledged_pct > 20 AND rating == BUY",
            action="warn"
        )
        
        # Get active guardrails
        guardrails = engine.get_active_guardrails()
        
        print(f"  Active guardrails: {len(guardrails)}")
        for g in guardrails:
            print(f"    [{g['action']:>5}] {g['rule']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_improvement_report():
    """Test comprehensive improvement report."""
    print("\n" + "=" * 50)
    print("Testing Improvement Report")
    print("=" * 50)
    
    try:
        from src.v3.selfimprove.engine import SelfImprovementEngine
        
        engine = SelfImprovementEngine()
        
        report = engine.get_improvement_report()
        
        print(f"  Report Summary:")
        print(f"    Total predictions evaluated: {report['performance']['total_evaluated']}")
        print(f"    Overall accuracy: {report['performance']['overall_accuracy_pct']}%")
        print(f"    Active suggestions: {report['active_suggestions']}")
        print(f"    Improvements applied: {report['improvements_applied']}")
        print(f"    Recent incidents: {report['recent_incidents']}")
        print(f"    Active guardrails: {report['active_guardrails']}")
        
        if report.get('suggestions'):
            print(f"\n  Top suggestions:")
            for s in report['suggestions'][:2]:
                print(f"    {s['improvement_type']} → {s['target']}")
                print(f"      {s['description'][:80]}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("V3 PHASE 3 TEST: Self-Improvement Engine")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_prediction_tracking()
    all_passed &= test_suggestion_generation()
    all_passed &= test_self_healing()
    all_passed &= test_guardrails()
    all_passed &= test_improvement_report()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 3 TESTS PASSED")
        print("\nNext: 'start v3 phase 4' for modern UI")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())