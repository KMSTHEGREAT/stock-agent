"""
Configuration settings for the stock analysis agent.
All tunable parameters should be defined here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "stock_agent.db"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Database
DATABASE = {
    "path": str(DB_PATH),
    "cache_ttl_hours": 24,  # Price data cache
    "fundamentals_ttl_hours": 168,  # Fundamentals cache (1 week)
    "news_ttl_hours": 6,  # News cache
}

# LLM Configuration
LLM = {
    "model": "phi4-mini",  # Changed from qwen3:4b
    "base_url": "http://localhost:11434",
    "temperature": 0.3,
    "max_tokens": 1000,
    "num_ctx": 4096,
    "timeout": 180,
}

# API Keys (add to .env file)
API_KEYS = {
    "finnhub": os.getenv("FINNHUB_API_KEY", ""),
    # Add other API keys as needed
}

# Market Configuration (Indian Market)
MARKET = {
    "timezone": "Asia/Kolkata",
    "currency": "INR",
    "currency_symbol": "₹",
    "exchange": "NSE",
    "market_hours": {
        "open": "09:15",
        "close": "15:30",
    },
    "indices": {
        "nifty_50": "^NSEI",
        "sensex": "^BSESN",
        "nifty_bank": "^NSEBANK",
        "nifty_it": "^CNXIT",
    },
    # NIFTY sectoral indices for sector classification
    "sectoral_indices": [
        "NIFTY IT", "NIFTY Bank", "NIFTY Pharma", "NIFTY FMCG",
        "NIFTY Auto", "NIFTY Metal", "NIFTY Energy", "NIFTY Realty",
        "NIFTY Infrastructure", "NIFTY PSU Bank", "NIFTY Media",
        "NIFTY Healthcare", "NIFTY Consumer Durables", "NIFTY Oil & Gas",
    ],
}

# SearXNG
SEARXNG = {
    "base_url": "http://localhost:8888",
    "timeout": 15,
}


# Scoring Weweights (India-adjusted)
SCORING = {
    "weights": {
        "technical": 0.30,
        "fundamental": 0.35,
        "news_sentiment": 0.20,
        "macro_context": 0.15,
    },
    "technical_sub_weights": {
        "trend": 0.15,
        "momentum": 0.10,
        "volatility": 0.05,
    },
    "fundamental_sub_weights": {
        "valuation": 0.10,
        "growth": 0.08,
        "profitability": 0.08,
        "leverage": 0.04,
        "promoter_governance": 0.05,
    },
    "recommendation_thresholds": {
        "strong_buy": 80,
        "buy": 70,
        "hold": 50,
        "sell": 40,
        "strong_sell": 30,
    },
}

# Technical Analysis Parameters
TECHNICAL = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,
    "sma_periods": [20, 50, 200],
    "volume_ma_period": 20,
}

# Risk Management
RISK = {
    "max_position_size_pct": 5,  # Max 5% of portfolio per stock
    "stop_loss_atr_multiplier": 1.5,
    "target_atr_multiplier_conservative": 2.0,
    "target_atr_multiplier_aggressive": 3.0,
    "circuit_limit_warning": 0.18,  # 18% daily move warning
    "min_liquidity_cr": 10,  # Minimum 10 crore market cap (in INR crores)
}

# Logging
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "stock_agent.log"),
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5,
}

# Disclaimer
DISCLAIMER = """
⚠️ DISCLAIMER: This analysis is for personal research and educational 
purposes only. It is NOT investment advice. The analysis may contain 
errors. Consult a SEBI Registered Investment Advisor before making 
investment decisions. Past performance does not guarantee future results.
"""