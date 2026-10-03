"""
SQLite-based cache for API responses.
Reduces external API calls and speeds up repeated analyses.
"""
import sqlite3
import json
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from pathlib import Path

from config.settings import CACHE_DIR, DATABASE
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCache:
    """SQLite-backed cache with TTL support."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(DATABASE["path"])
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize cache table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                data_type TEXT,
                created_at TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Index for faster expiry checks
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expiry 
            ON cache(expires_at)
        """)
        conn.commit()
        conn.close()
    
    def get(self, key: str, max_age_hours: Optional[int] = None) -> Optional[Any]:
        """
        Retrieve cached data if not expired.
        
        Args:
            key: Cache key (e.g., "RELIANCE_price_1y")
            max_age_hours: Maximum age in hours (overrides stored TTL)
        
        Returns:
            Cached data if valid, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT value, data_type, created_at, expires_at FROM cache WHERE key = ?",
                (key,)
            ).fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Check expiry
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now(timezone.utc) > expires_at:
                # Cache expired, remove it
                self._remove(key)
                return None
            
            # Override TTL if specified
            if max_age_hours is not None:
                created_at = datetime.fromisoformat(row["created_at"])
                if datetime.now(timezone.utc) > created_at + timedelta(hours=max_age_hours):
                    self._remove(key)
                    return None
            
            # Deserialize
            if row["data_type"] == "json":
                return json.loads(row["value"])
            else:
                return pickle.loads(row["value"])
                
        except Exception as e:
            logger.error(f"Cache retrieval error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_hours: int = 24, 
            data_type: str = "pickle") -> bool:
        """
        Store data in cache.
        
        Args:
            key: Cache key
            value: Data to cache
            ttl_hours: Time to live in hours
            data_type: "json" or "pickle"
        """
        try:
            # Serialize
            if data_type == "json":
                serialized = json.dumps(value).encode()
            else:
                serialized = pickle.dumps(value)
            
            expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
            
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO cache (key, value, data_type, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, (key, serialized, data_type, 
                  datetime.now(timezone.utc).isoformat(),
                  expires_at.isoformat()))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache storage error for {key}: {e}")
            return False
    
    def _remove(self, key: str):
        """Remove a specific cache entry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    
    def clear_expired(self):
        """Remove all expired cache entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "DELETE FROM cache WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(),)
        )
        removed = cursor.rowcount
        conn.commit()
        conn.close()
        if removed > 0:
            logger.info(f"Cleared {removed} expired cache entries")
    
    def clear_all(self):
        """Clear entire cache."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        logger.info("Cleared all cache entries")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        expired = conn.execute(
            "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(),)
        ).fetchone()[0]
        conn.close()
        
        return {
            "total_entries": total,
            "expired_entries": expired,
            "valid_entries": total - expired
        }