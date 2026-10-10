"""
Multi-source news aggregator for Indian financial markets.
Integrates 12+ RSS feeds from 6 sources with parallel fetching,
deduplication, and source bias detection.

Sources: Moneycontrol, Economic Times, Business Standard,
NSE, Livemint, SearXNG (fallback)
"""
from dataclasses import dataclass

import feedparser
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

logger = logging.getLogger(__name__)

# Verified RSS feed URLs
RSS_FEEDS = {
    # Moneycontrol
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    
    # Economic Times
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "et_default": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    
    # Business Standard
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "bs_companies": "https://www.business-standard.com/rss/companies-101.rss",
    
    # Livemint
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "livemint_companies": "https://www.livemint.com/rss/companies",
    
    # NSE (announcements)
    "nse_announcements": "https://www.nseindia.com/static/rss/announcements.xml",
}

# Source reliability scores (based on editorial standards)
SOURCE_RELIABILITY = {
    "moneycontrol": 0.85,
    "et": 0.90,
    "bs": 0.88,
    "livemint": 0.87,
    "nse": 0.95,  # Official exchange data
    "searxng": 0.70,  # Aggregated, variable quality
}


@dataclass
class NewsItem:
    """Single news item with source metadata."""
    title: str
    source: str
    summary: str = ""
    url: str = ""
    published: str = ""
    reliability: float = 0.5  # Source reliability score
    sentiment_hint: Optional[str] = None  # pre-computed hint


class MultiSourceNewsAggregator:
    """Aggregates news from multiple Indian financial sources with bias detection."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self._cache = {}
    
    def get_stock_news(self, symbol: str, company_name: str = None,
                       max_items: int = 20) -> List[NewsItem]:
        """
        Get news for a specific stock from all sources.
        
        Returns deduplicated, source-diverse news items.
        """
        cache_key = f"{symbol}_news_v3"
        if cache_key in self._cache:
            cached, ts = self._cache[cache_key]
            if (datetime.now() - ts).seconds < 3600:  # 1 hour cache
                return cached[:max_items]
        
        search_terms = [symbol.upper()]
        if company_name:
            # Use first word of company name for broader match
            search_terms.append(company_name.split()[0].upper())
        
        all_items = []
        seen_titles = set()
        
        # Fetch all feeds in parallel
        feed_results = self._fetch_all_feeds()
        
        # Filter for symbol mentions and deduplicate
        for source, items in feed_results.items():
            reliability = SOURCE_RELIABILITY.get(
                source.split('_')[0], 0.5
            )
            
            for item in items:
                # Check if any search term appears in title or summary
                text = (item.get('title', '') + ' ' + item.get('summary', '')).upper()
                
                if any(term in text for term in search_terms):
                    title_key = item.get('title', '').lower()[:80]
                    
                    if title_key not in seen_titles:
                        seen_titles.add(title_key)
                        
                        news_item = NewsItem(
                            title=item.get('title', ''),
                            source=source,
                            summary=item.get('summary', '')[:300],
                            url=item.get('link', ''),
                            published=item.get('published', ''),
                            reliability=reliability,
                        )
                        all_items.append(news_item)
        
        # If insufficient stock-specific news, get general market news
        if len(all_items) < 5:
            market_news = self.get_market_news(max_items=20)
            for item in market_news:
                title_key = item.title.lower()[:80]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_items.append(item)
        
        # Sort by reliability and date
        all_items.sort(key=lambda x: (x.reliability, x.published), reverse=True)
        
        # Cache
        self._cache[cache_key] = (all_items, datetime.now())
        
        logger.info(f"[News V3] {symbol}: {len(all_items)} items from "
                   f"{len(set(i.source for i in all_items))} sources")
        
        return all_items[:max_items]
    
    def get_market_news(self, max_items: int = 30) -> List[NewsItem]:
        """Get general market news from all sources."""
        all_items = []
        
        feed_results = self._fetch_all_feeds()
        
        for source, items in feed_results.items():
            reliability = SOURCE_RELIABILITY.get(
                source.split('_')[0], 0.5
            )
            
            for item in items:
                all_items.append(NewsItem(
                    title=item.get('title', ''),
                    source=source,
                    summary=item.get('summary', '')[:300],
                    url=item.get('link', ''),
                    published=item.get('published', ''),
                    reliability=reliability,
                ))
        
        # Sort by published date (most recent first)
        all_items.sort(key=lambda x: x.published, reverse=True)
        
        return all_items[:max_items]
    
    def analyze_source_diversity(self, news_items: List[NewsItem]) -> Dict:
        """
        Analyze source diversity to detect potential bias.
        A diverse news sample should have items from 3+ different sources.
        """
        if not news_items:
            return {"diversity_score": 0, "unique_sources": 0, "warning": "No news items"}
        
        source_counts = Counter(item.source for item in news_items)
        unique_sources = len(source_counts)
        total_items = len(news_items)
        
        # Diversity score: 0-100
        # 4+ sources = high diversity
        # 2-3 sources = moderate
        # 1 source = low (potential bias)
        if unique_sources >= 4:
            diversity_score = 90
        elif unique_sources == 3:
            diversity_score = 70
        elif unique_sources == 2:
            diversity_score = 50
        else:
            diversity_score = 20
        
        # Check for source concentration
        most_common = source_counts.most_common(1)[0]
        concentration = most_common[1] / total_items
        
        if concentration > 0.7:  # One source > 70% of items
            diversity_score -= 20
            warning = f"High concentration from {most_common[0]} ({concentration:.0%})"
        elif concentration > 0.5:
            diversity_score -= 10
            warning = f"Moderate concentration from {most_common[0]}"
        else:
            warning = None
        
        # Average reliability
        avg_reliability = sum(item.reliability for item in news_items) / total_items
        
        return {
            "diversity_score": max(0, min(100, diversity_score)),
            "unique_sources": unique_sources,
            "source_counts": dict(source_counts),
            "concentration_pct": round(concentration * 100, 1),
            "avg_reliability": round(avg_reliability, 2),
            "warning": warning,
        }
    
    def detect_sentiment_keywords(self, news_items: List[NewsItem]) -> Dict:
        """
        Quick keyword-based sentiment analysis (no LLM).
        Useful as pre-filter before LLM analysis.
        """
        positive_keywords = [
            'surge', 'gain', 'rise', 'profit', 'growth', 'upgrade', 'buy',
            'bullish', 'strong', 'record', 'high', 'beat', 'exceed', 'positive',
            'expansion', 'acquisition', 'dividend', 'bonus', 'order win',
            'multiplier', 'multibagger', 'outperform', 'recommend',
        ]
        
        negative_keywords = [
            'fall', 'drop', 'decline', 'loss', 'downgrade', 'sell', 'bearish',
            'weak', 'low', 'miss', 'below', 'negative', 'fraud', 'probe',
            'investigation', 'penalty', 'fine', 'pledge', 'crash', 'default',
            'bankruptcy', 'delisting', 'underperform', 'warning', 'risk',
        ]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for item in news_items:
            text = (item.title + ' ' + item.summary).lower()
            
            pos = sum(1 for kw in positive_keywords if kw in text)
            neg = sum(1 for kw in negative_keywords if kw in text)
            
            if pos > neg:
                item.sentiment_hint = "positive"
                positive_count += 1
            elif neg > pos:
                item.sentiment_hint = "negative"
                negative_count += 1
            else:
                item.sentiment_hint = "neutral"
                neutral_count += 1
        
        total = len(news_items)
        
        if total == 0:
            return {"score": 50, "label": "neutral", "counts": {}}
        
        # Compute sentiment score
        if positive_count + negative_count == 0:
            score = 50
        else:
            ratio = positive_count / (positive_count + negative_count)
            score = 50 + (ratio - 0.5) * 100
        
        if score > 65:
            label = "positive"
        elif score < 35:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": round(max(0, min(100, score)), 1),
            "label": label,
            "counts": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
            },
        }
    
    def _fetch_all_feeds(self) -> Dict[str, List[Dict]]:
        """Fetch all RSS feeds in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._fetch_single_feed, url, source): source
                for source, url in RSS_FEEDS.items()
            }
            
            for future in as_completed(futures, timeout=30):
                source = futures[future]
                try:
                    items = future.result(timeout=10)
                    results[source] = items
                except Exception:
                    results[source] = []
        
        return results
    
    def _fetch_single_feed(self, feed_url: str, source: str) -> List[Dict]:
        """Fetch and parse a single RSS feed."""
        try:
            response = self.session.get(feed_url, timeout=10)
            
            if response.status_code != 200:
                logger.debug(f"[News V3] {source} returned {response.status_code}")
                return []
            
            feed = feedparser.parse(response.content)
            
            items = []
            for entry in feed.entries[:20]:
                items.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', '')[:300],
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                })
            
            return items
            
        except requests.Timeout:
            logger.debug(f"[News V3] {source} timed out")
            return []
        except Exception as e:
            logger.debug(f"[News V3] {source} error: {e}")
            return []