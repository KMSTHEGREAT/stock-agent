"""
SearXNG integration for unlimited, self-hosted web search.
Replaces DDGS with better results and no rate limits.
"""
import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from src.v1.data.cache import DataCache
from src.v1.utils.logger import setup_logger

logger = setup_logger(__name__)


class SearXNGSearch:
    """Self-hosted metasearch via SearXNG Docker instance."""
    
    def __init__(self, base_url: str = "http://localhost:8888",
                 cache: Optional[DataCache] = None):
        self.base_url = base_url
        self.cache = cache or DataCache()
        self.client = httpx.AsyncClient(timeout=15.0)
    
    async def search(self, query: str, categories: str = "general",
                     time_range: str = "", max_results: int = 10) -> List[Dict]:
        """
        Search via SearXNG.
        
        Args:
            query: Search query
            categories: "general", "news", "images", "videos"
            time_range: "", "day", "week", "month", "year"
            max_results: Maximum results to return
        
        Returns:
            List of result dictionaries
        """
        cache_key = f"searxng_{query}_{categories}_{time_range}"
        cached = self.cache.get(cache_key, max_age_hours=6)
        if cached:
            return cached[:max_results]
        
        params = {
            "q": query,
            "format": "json",
            "categories": categories,
        }
        
        if time_range:
            params["time_range"] = time_range
        
        try:
            response = await self.client.get(
                f"{self.base_url}/search",
                params=params
            )
            
            if response.status_code != 200:
                logger.warning(f"SearXNG returned {response.status_code}")
                return []
            
            data = response.json()
            results = data.get("results", [])[:max_results]
            
            # Cache
            self.cache.set(cache_key, results, ttl_hours=6, data_type="json")
            
            return results
            
        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
            return []
    
    async def search_news(self, query: str, days: int = 7) -> List[Dict]:
        """Search for news with time filter."""
        time_range = "day" if days <= 1 else "week" if days <= 7 else "month"
        
        results = await self.search(
            query=query,
            categories="news",
            time_range=time_range,
            max_results=15
        )
        
        # Filter and format results
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": r.get("publishedDate", "") or r.get("engine", ""),
                "summary": r.get("content", "")[:300],
                "published": r.get("publishedDate", ""),
            })
        
        return formatted
    
    async def search_stock_news(self, symbol: str, 
                               company_name: str = None) -> List[Dict]:
        """Search for stock-specific news."""
        queries = [
            f"{symbol} NSE stock news India",
            f"{symbol} earnings results",
        ]
        
        if company_name:
            queries.append(f"{company_name} stock India")
        
        all_results = []
        seen_urls = set()
        
        # Run searches in parallel
        tasks = [self.search_news(q) for q in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for results in results_list:
            if isinstance(results, Exception):
                continue
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        
        # Sort by date (most recent first)
        all_results.sort(key=lambda x: x.get("published", ""), reverse=True)
        
        return all_results[:20]
    
    async def search_financial_data(self, query: str) -> List[Dict]:
        """Search for financial data, reports, announcements."""
        results = await self.search(
            query=f"{query} site:screener.in OR site:nseindia.com OR site:bseindia.com",
            categories="general",
            max_results=10
        )
        return results
    
    async def health_check(self) -> bool:
        """Check if SearXNG is running."""
        try:
            response = await self.client.get(
                f"{self.base_url}/healthz",
                timeout=5.0
            )
            return response.status_code == 200
        except:
            # Try a search as fallback
            try:
                response = await self.client.get(
                    f"{self.base_url}/search",
                    params={"q": "test", "format": "json"},
                    timeout=5.0
                )
                return response.status_code == 200
            except:
                return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ============ SYNCHRONOUS WRAPPER ============
# For use in non-async code (like LangGraph nodes)

class SearXNGSearchSync:
    """Synchronous wrapper for SearXNGSearch."""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
        import httpx
        self.client = httpx.Client(timeout=15.0)
    
    def search(self, query: str, categories: str = "general",
               time_range: str = "", max_results: int = 10) -> List[Dict]:
        """Synchronous search."""
        params = {
            "q": query,
            "format": "json",
            "categories": categories,
        }
        
        if time_range:
            params["time_range"] = time_range
        
        try:
            response = self.client.get(
                f"{self.base_url}/search",
                params=params
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            return data.get("results", [])[:max_results]
            
        except Exception as e:
            logger.error(f"SearXNG sync search error: {e}")
            return []
    
    def search_news(self, query: str, days: int = 7) -> List[Dict]:
        """Synchronous news search."""
        time_range = "day" if days <= 1 else "week" if days <= 7 else "month"
        
        results = self.search(
            query=query,
            categories="news",
            time_range=time_range,
            max_results=15
        )
        
        return [{
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source": r.get("engine", ""),
            "summary": r.get("content", "")[:300],
            "published": r.get("publishedDate", ""),
        } for r in results]
    
    def search_stock_news(self, symbol: str,
                          company_name: str = None) -> List[Dict]:
        """Synchronous stock news search."""
        queries = [
            f"{symbol} NSE stock news India",
            f"{symbol} earnings results latest",
        ]
        
        if company_name:
            queries.append(f"{company_name} stock India")
        
        all_results = []
        seen_urls = set()
        
        for query in queries:
            results = self.search_news(query)
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        
        return all_results[:20]