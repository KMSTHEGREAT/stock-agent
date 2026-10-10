"""
Test V3 Phase 2: Multi-source news aggregator.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_news_fetch():
    """Test fetching news for a stock."""
    print("\n" + "=" * 50)
    print("Testing Multi-Source News Fetch")
    print("=" * 50)
    
    try:
        from src.v3.news.multi_source import MultiSourceNewsAggregator
        
        aggregator = MultiSourceNewsAggregator()
        
        symbol = "RELIANCE"
        print(f"\n  Fetching news for {symbol}...")
        
        start = time.time()
        news = aggregator.get_stock_news(symbol, "Reliance Industries")
        elapsed = time.time() - start
        
        print(f"  ✓ Fetched {len(news)} items in {elapsed:.1f}s")
        
        if news:
            # Show items
            print(f"\n  Top news items:")
            for item in news[:5]:
                print(f"    [{item.source}] {item.title[:60]}")
                print(f"      Reliability: {item.reliability:.2f}")
                if item.sentiment_hint:
                    print(f"      Sentiment hint: {item.sentiment_hint}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_source_diversity():
    """Test source diversity analysis."""
    print("\n" + "=" * 50)
    print("Testing Source Diversity Analysis")
    print("=" * 50)
    
    try:
        from src.v3.news.multi_source import MultiSourceNewsAggregator
        
        aggregator = MultiSourceNewsAggregator()
        
        # Get market news
        print("  Fetching market news...")
        news = aggregator.get_market_news(max_items=30)
        
        print(f"  ✓ Got {len(news)} market news items")
        
        # Analyze diversity
        diversity = aggregator.analyze_source_diversity(news)
        
        print(f"\n  Source Diversity Analysis:")
        print(f"    Diversity Score: {diversity['diversity_score']}/100")
        print(f"    Unique Sources: {diversity['unique_sources']}")
        print(f"    Average Reliability: {diversity['avg_reliability']:.2f}")
        
        if diversity.get('source_counts'):
            print(f"\n    Source breakdown:")
            for source, count in sorted(diversity['source_counts'].items()):
                print(f"      {source}: {count} items")
        
        if diversity.get('warning'):
            print(f"\n    ⚠ Warning: {diversity['warning']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_sentiment_keywords():
    """Test keyword-based sentiment detection."""
    print("\n" + "=" * 50)
    print("Testing Keyword Sentiment Detection")
    print("=" * 50)
    
    try:
        from src.v3.news.multi_source import MultiSourceNewsAggregator
        
        aggregator = MultiSourceNewsAggregator()
        
        # Get stock news
        news = aggregator.get_stock_news("TCS", "Tata Consultancy Services")
        
        if news:
            # Run keyword sentiment
            sentiment = aggregator.detect_sentiment_keywords(news)
            
            print(f"  Keyword Sentiment Analysis:")
            print(f"    Score: {sentiment['score']}/100")
            print(f"    Label: {sentiment['label']}")
            print(f"    Counts: {sentiment['counts']}")
            
            # Show sentiment by item
            print(f"\n  Sentiment by item:")
            for item in news[:5]:
                print(f"    [{item.sentiment_hint or 'none':>8}] {item.title[:50]}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("V3 PHASE 2 TEST: Multi-Source News")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_news_fetch()
    all_passed &= test_source_diversity()
    all_passed &= test_sentiment_keywords()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 2 TESTS PASSED")
        print("\nNext: 'start v3 phase 3' for self-improvement engine")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())