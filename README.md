# Indian Stock Analysis Agent

A personal stock analysis tool for Indian markets (NSE/BSE) that combines deterministic quantitative scoring with LLM-powered qualitative analysis.

## Features

* **Technical Analysis**: RSI, MACD, Bollinger Bands, ATR, Moving Averages
* **Fundamental Analysis**: P/E, ROCE, Promoter Holding, Pledged Shares
* **News Sentiment**: RSS feeds + DuckDuckGo + LLM scoring
* **LLM Analysis**: Qwen 3 4B for qualitative insights (runs locally)
* **Scoring**: 0-100 composite score with India-specific weights
* **Recommendations**: Entry/Exit/Stop-loss prices in INR
* **Market Context**: Nifty/Sensex/VIX regime detection

---

## Architecture

```text
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
```

---

## Quick Start

### Prerequisites
* Python 3.11+
* Ollama with Qwen 3 4B model installed
* Minimum Hardware: 6GB VRAM (for LLM), 16GB RAM

### Installation

```bash
# 1. Install Ollama and pull the model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly

# 3. Make CLI tool executable
chmod +x main.py
```

### Verification & Testing

Run the final validation pipeline test to ensure components are wired correctly:

```bash
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
```

---

## Usage

### Web UI (Recommended)
```bash
streamlit run app.py
```

### CLI Operations
```bash
# Analyze stocks
python main.py analyze RELIANCE
python main.py analyze TCS --detailed

# Watchlist management
python main.py watchlist RELIANCE TCS INFY
python main.py add RELIANCE
python main.py list-watchlist
python main.py remove RELIANCE

# Market status and analytics history
python main.py market
python main.py history
```

---

## Analysis Matrix

### Scoring System

| Component | Weight | Target Metrics / Source |
| :--- | :---: | :--- |
| **Technical** | 30% | RSI, MACD, Bollinger Bands, SMA/EMA |
| **Fundamental** | 35% | P/E, ROCE, Promoter Holding, Revenue/Profit Growth |
| **News Sentiment** | 20% | LLM synthesis + keyword matching analysis |
| **Market Context** | 15% | Nifty/Sensex overall momentum trend, India VIX |

### India-Specific Rule Modifiers
* **Promoter Holding**: Levels >50% are scored positively (verifies skin in the game).
* **Pledged Shares**: Monitored as a major structural red flag and heavily penalized.
* **ROCE**: Weighted higher than ROE due to its historical reliability for Indian corporate capital structures.
* **FII/DII Flows**: Evaluated actively within the macro market context layers.
* **Circuit Limits**: Intraday moves nearing or exceeding 18% trigger sudden volatility flags.

---

## Data Sources (100% Free Tier)

| Source | Usage Context | Rate Limit Status |
| :--- | :--- | :--- |
| **yfinance (`.NS`)** | Structural price history, standard baseline fundamentals | Unlimited (via public API endpoints) |
| **Screener.in** | Granular Indian fundamentals, historical shareholding data | Manual extraction / web scraping |
| **RSS Feeds** | Real-time news extraction from Moneycontrol, Economic Times, etc. | Unrestricted / Free |
| **DuckDuckGo** | Dynamic search querying for localized auxiliary news events | Unrestricted / Free |
| **NSE Indices** | Nifty 50, BSE Sensex benchmarks, and volatility tracking | Unrestricted / Free |

---

## Tech Stack

* **Language**: Python 3.11+
* **Local LLM Engine**: Qwen 3 4B via Ollama framework
* **Application Interfaces**: Streamlit (Dashboard), Click (Terminal Commands)
* **Data Scrapers**: yfinance, BeautifulSoup4, feedparser
* **Processing Engines**: pandas, pandas-ta, numpy
* **Data Architecture**: SQLite (Caching, History Logs), Pydantic (Data Serialization)

---

## Project Structure

```text
stock-agent/
├── config/              # Application and API configuration settings
├── src/
│   ├── data/            # Data orchestration, ingestion & cache layers
│   ├── analysis/        # Technical analysis indicators & fundamental processing
│   ├── llm/             # Prompt engineering and local Qwen text synthesis
│   ├── news/            # RSS feed fetching & DuckDuckGo query scraping
│   ├── scoring/         # Core weighted processing metrics
│   ├── recommendations/ # Dynamic target, stop-loss and limit entry logic
│   ├── schemas/         # Strict structural verification via Pydantic
│   └── utils/           # Structural validation, execution logs, file parsers
├── tests/               # Unit diagnostics and pipeline system checks
├── app.py               # Streamlit interactive UI layout
├── main.py              # Click command-line utility entrypoint
└── README.md
```

---

## Disclaimer

⚠️ **This software application is developed exclusively for educational and personal research purposes.** It is not structured to provide professional investment advice. Always execute independent verification and consult with a **SEBI Registered Investment Advisor** prior to deploying capital into active market environments. Past performance tracking features do not offer guarantees regarding future strategic allocations.
