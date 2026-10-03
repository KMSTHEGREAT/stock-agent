"""
News aggregator for Indian stock market.
Combines RSS feeds and DuckDuckGo search.
"""
import feedparser
import requests
from typing import List, Dict, Optional
from datetime import datetime

from v1.data.cache import DataCache
from v1.utils.logger import setup_logger

logger = setup_logger(__name__)


class IndianNewsAggregator:
    """Aggregates news from Indian financial news sources."""
    
    RSS_FEEDS = {
        "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
        "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
        "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
        "livemint_markets": "https://www.livemint.com/rss/markets",
    }
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
    
    def get_stock_news(self, symbol: str, company_name: str = None, 
                       max_items: int = 10) -> List[Dict]:
        """Get recent news for a specific stock."""
        cache_key = f"{symbol}_news"
        cached = self.cache.get(cache_key, max_age_hours=6)
        if cached:
            return cached
        
        news_items = []
        
        # Search terms
        search_terms = [symbol]
        if company_name:
            search_terms.append(company_name)
        
        # 1. Search RSS feeds
        for term in search_terms:
            rss_news = self._search_rss_feeds(term)
            news_items.extend(rss_news)
        
        # 2. DuckDuckGo search
        try:
            ddg_news = self._search_duckduckgo(symbol)
            news_items.extend(ddg_news)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        # Deduplicate by title
        seen_titles = set()
        unique_news = []
        for item in news_items:
            title_key = item['title'].lower()[:80]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(item)
        
        # Sort and limit
        unique_news = sorted(
            unique_news, 
            key=lambda x: x.get('published', ''), 
            reverse=True
        )[:max_items]
        
        # Cache
        self.cache.set(cache_key, unique_news, ttl_hours=6, data_type="json")
        
        logger.info(f"Fetched {len(unique_news)} news items for {symbol}")
        return unique_news
    
    def get_market_news(self, max_items: int = 20) -> List[Dict]:
        """Get general market news."""
        cache_key = "market_news"
        cached = self.cache.get(cache_key, max_age_hours=3)
        if cached:
            return cached
        
        news_items = []
        for feed_name, feed_url in self.RSS_FEEDS.items():
            items = self._fetch_rss_feed(feed_url, source=feed_name)
            news_items.extend(items)
        
        news_items = sorted(
            news_items, 
            key=lambda x: x.get('published', ''), 
            reverse=True
        )[:max_items]
        
        self.cache.set(cache_key, news_items, ttl_hours=3, data_type="json")
        return news_items
    
    def _search_rss_feeds(self, search_term: str) -> List[Dict]:
        """Search RSS feeds for mentions of search term."""
        results = []
        
        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                items = self._fetch_rss_feed(feed_url, source=feed_name)
                
                for item in items:
                    if (search_term.lower() in item['title'].lower() or
                        search_term.lower() in item.get('summary', '').lower()):
                        results.append(item)
            except Exception as e:
                logger.debug(f"Error searching feed {feed_name}: {e}")
        
        return results
    
    def _fetch_rss_feed(self, feed_url: str, source: str) -> List[Dict]:
        """Fetch and parse an RSS feed."""
        try:
            response = self.session.get(feed_url, timeout=10)
            
            if response.status_code != 200:
                return []
            
            feed = feedparser.parse(response.content)
            
            items = []
            for entry in feed.entries[:30]:
                items.append({
                    'title': entry.get('title', ''),
                    'source': source,
                    'summary': entry.get('summary', '')[:300],
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                })
            
            return items
            
        except requests.Timeout:
            logger.warning(f"RSS feed {source} timed out")
            return []
        except Exception as e:
            logger.debug(f"Error fetching {source}: {e}")
            return []
    
    def _search_duckduckgo(self, symbol: str) -> List[Dict]:
        """Search DuckDuckGo for recent news."""
        try:
            from ddgs import DDGS
            
            results = []
            with DDGS() as ddgs:
                query = f"{symbol} NSE stock India"
                news_results = ddgs.news(query, region="in-en", max_results=10)
                
                for r in news_results:
                    results.append({
                        'title': r.get('title', ''),
                        'source': r.get('source', 'DuckDuckGo'),
                        'summary': r.get('body', '')[:300],
                        'link': r.get('url', ''),
                        'published': r.get('date', ''),
                    })
            
            return results
            
        except ImportError:
            logger.warning("duckduckgo_search not installed")
            return []
        except Exception as e:
            logger.debug(f"DuckDuckGo error: {e}")
            return []