"""
SearXNG integration for unlimited, self-hosted web search.
Self-contained — no V1 dependencies.
"""
import httpx
import json
import logging
import sqlite3
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Simple cache using the existing database
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "stock_agent.db"


class SimpleCache:
    """Minimal cache that doesn't depend on V1 code."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS v2_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TEXT,
                expires_at TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def get(self, key: str, max_age_hours: int = 6) -> Optional[list]:
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT value, expires_at FROM v2_cache WHERE key = ?",
                (key,)
            ).fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Check expiry
            expires = datetime.fromisoformat(row[1])
            if datetime.now(timezone.utc) > expires:
                return None
            
            return json.loads(row[0])
        except:
            return None
    
    def set(self, key: str, value: list, ttl_hours: int = 6):
        try:
            expires = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO v2_cache (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value), datetime.now(timezone.utc).isoformat(), expires.isoformat())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Cache write error: {e}")


class SearXNGSearch:
    """Self-hosted metasearch via SearXNG."""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
        self.cache = SimpleCache()
        self.client = httpx.Client(timeout=15.0)
    
    def search(self, query: str, categories: str = "general",
               time_range: str = "", max_results: int = 10) -> List[Dict]:
        """
        Search via SearXNG.
        
        Args:
            query: Search query
            categories: "general", "news", "images", "videos"  
            time_range: "", "day", "week", "month", "year"
            max_results: Maximum results
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
            response = self.client.get(
                f"{self.base_url}/search",
                params=params
            )
            
            if response.status_code != 200:
                logger.warning(f"SearXNG returned {response.status_code}")
                return []
            
            data = response.json()
            results = data.get("results", [])[:max_results]
            
            # Cache
            self.cache.set(cache_key, results, ttl_hours=6)
            
            return results
            
        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
            return []
    
    def search_news(self, query: str, days: int = 7) -> List[Dict]:
        """Search for news."""
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
        """Search for stock-specific news from multiple queries."""
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
    
    def search_financial_data(self, query: str) -> List[Dict]:
        """Search for financial data from specific sites."""
        return self.search(
            query=f"{query} site:screener.in OR site:nseindia.com OR site:bseindia.com",
            categories="general",
            max_results=10
        )
    
    def health_check(self) -> bool:
        """Check if SearXNG is running."""
        try:
            response = self.client.get(
                f"{self.base_url}/search",
                params={"q": "test", "format": "json"},
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()