Indian Stock Analysis Agent

A personal stock analysis tool for Indian markets (NSE/BSE) that combinesdeterministic quantitative scoring with LLM-powered qualitative analysis.
Features

    Technical Analysis: RSI, MACD, Bollinger Bands, ATR, Moving Averages
    Fundamental Analysis: P/E, ROCE, Promoter Holding, Pledged Shares
    News Sentiment: RSS feeds + DuckDuckGo + LLM scoring
    LLM Analysis: Qwen 3 4B for qualitative insights (runs locally)
    Scoring: 0-100 composite score with India-specific weights
    Recommendations: Entry/Exit/Stop-loss prices in INR
    Market Context: Nifty/Sensex/VIX regime detection

Architecture

┌─────────────────────────────────────────────────┐
│                  User Interface                  │
│         (Streamlit Web UI / CLI)                 │
├─────────────────────────────────────────────────┤
│              Analysis Pipeline                   │
├─────────┬─────────┬─────────┬───────────────────┤
│Technical│Fundament│  News   │  Market Context    │
│Analysis │Analysis │ Analysis│  (Nifty/VIX)       │
├─────────┴─────────┴─────────┴───────────────────┤
│              Scoring Engine                      │
│  Tech 30% | Fund 35% | News 20% | Macro 15%    │
├─────────────────────────────────────────────────┤
│           LLM Synthesis (Qwen 3 4B)              │
│     Qualitative analysis & risk flags            │
├─────────────────────────────────────────────────┤
│              Data Layer                          │
│  yfinance | Screener.in | RSS | DuckDuckGo     │
├─────────────────────────────────────────────────┤
│           SQLite (Cache + History)               │
└─────────────────────────────────────────────────┘
text
 
  
 
 

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama with Qwen 3 4B model
- 6GB VRAM (for LLM), 16GB RAM

### Installation

```bash
# 1. Install Ollama and model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# 2. Setup Python environment
python -m venv venv
source venv/bin activate
pip install -r requirements.txt

# 3. Run tests
python test_setup.py

# 4. Start the web UI
streamlit run app.py
 
 
Usage
bash
 
  
 
 
# Web UI (recommended)
streamlit run app.py

# CLI
python main.py analyze RELIANCE
python main.py analyze TCS --detailed
python main.py watchlist RELIANCE TCS INFY
python main.py market
python main.py history

# Watchlist management
python main.py add RELIANCE
python main.py list-watchlist
python main.py remove RELIANCE
 
 
Scoring System
Component
	
Weight
	
Source
Technical	30%	RSI, MACD, Bollinger, SMA/EMA
Fundamental	35%	P/E, ROCE, Promoter Holding, Growth
News Sentiment	20%	LLM + keyword analysis
Market Context	15%	Nifty/Sensex trend, VIX
 
 
India-Specific Adjustments

     Promoter Holding: >50% is positive (skin in the game)
     Pledged Shares: Major red flag, heavily penalized
     ROCE: More relevant than ROE for Indian companies
     FII/DII Flows: Tracked as market context
     Circuit Limits: 18%+ daily moves flagged

Data Sources (All Free)
Source
	
Usage
	
Rate Limit
yfinance (.NS)	Price history, basic fundamentals	Unlimited (unofficial)
Screener.in	Detailed fundamentals, shareholding	Manual/scraping
RSS feeds	News from Moneycontrol, ET, etc.	Free
DuckDuckGo	Additional news search	Free
NSE indices	Nifty/Sensex for market context	Free
 
 
Disclaimer

⚠️ This tool is for personal research and educational purposes only.
It is NOT investment advice. Consult a SEBI Registered Investment Advisor
before making investment decisions. Past performance does not guarantee
future results.
Tech Stack

     Language: Python 3.11+
     LLM: Qwen 3 4B (via Ollama, runs locally on 6GB VRAM)
     Framework: Streamlit (UI), Click (CLI)
     Data: yfinance, BeautifulSoup, feedparser
     Analysis: pandas-ta, pandas, numpy
     Database: SQLite
     Validation: Pydantic

Project Structure
text
 
  
 
 
stock-agent/
├── src/
│   ├── data/          # Data fetching and caching
│   ├── analysis/      # Technical & fundamental analysis
│   ├── llm/           # LLM synthesis (Qwen 3 4B)
│   ├── news/          # News aggregation & sentiment
│   ├── scoring/       # Scoring engine
│   ├── recommendations/ # Entry/exit price generation
│   ├── schemas/       # Pydantic models
│   └── utils/         # Logging, validation
├── config/            # Settings
├── tests/             # Test files
├── app.py             # Streamlit UI
├── main.py            # CLI
└── README.md
 
 
text
 
  
 
 

---

## Final Setup Instructions

```bash
# 1. Make scripts executable
chmod +x main.py

# 2. Install streamlit if not already
pip install streamlit plotly

# 3. Run final test
python -c "
import sys
sys.path.insert(0, '.')
from src.pipeline import StockAnalysisPipeline

pipeline = StockAnalysisPipeline()
result = pipeline.analyze_stock('RELIANCE')

if result:
    print('✓ SYSTEM WORKING')
    print(f'  {result[\"symbol\"]}: {result[\"overall_score\"]}/100')
    print(f'  Recommendation: {result[\"recommendation\"].upper()}')
    print(f'  Time: {result[\"processing_time_seconds\"]:.1f}s')
else:
    print('✗ SYSTEM FAILED')
"

# 4. Start using
streamlit run app.py