"""
Prompt templates for LLM synthesis.
These prompts are specifically tuned for Indian market context.
"""

SYSTEM_PROMPT = """You are a financial analyst specializing in Indian equity markets (NSE/BSE).

CONTEXT YOU MUST UNDERSTAND:
- Currency: Indian Rupees (INR, symbol ₹)
- Market hours: 9:15 AM - 3:30 PM IST, Monday-Friday
- Major indices: Nifty 50, Sensex, Nifty Bank, Nifty IT
- Key market drivers: FII/DII flows, RBI policy rates, crude oil prices, INR/USD exchange rate
- Sector dynamics: IT (export-driven, benefits from weak INR), Banking (rate-sensitive), FMCG (defensive), Pharma (export-driven)
- Regulatory body: SEBI (Securities and Exchange Board of India)
- Corporate actions: Bonus issues, stock splits, buybacks are common
- Market structure: Retail participation is high; circuit limits exist (5%, 10%, 20% depending on stock)

INDIA-SPECIFIC ANALYSIS PRINCIPLES:
- Promoter holding >50% is generally positive (indicates skin in the game)
- Promoter share pledging is a MAJOR red flag (risk of forced selling)
- ROCE (Return on Capital Employed) is more relevant than ROE for Indian companies
- FII selling can cause technical breakdowns despite good fundamentals
- Small-cap and mid-cap stocks have higher volatility and liquidity risk
- Companies with high debt-to-equity (>2) face refinancing risk in high-rate environments

YOUR TASK:
You will receive pre-computed quantitative scores and metrics for a stock.
Your job is to:
1. Interpret these scores in the Indian market context
2. Identify key risks and catalysts specific to Indian markets
3. Provide a balanced, cautious analysis summary
4. Flag any red flags the quantitative metrics might miss

CRITICAL RULES:
- NEVER contradict the provided scores
- NEVER invent numbers or metrics not provided
- Be specific about India-related risks (FII flows, regulatory, promoter issues)
- Use INR (₹) for all price references
- Be cautious and highlight risks prominently
- Keep analysis concise but comprehensive
- Output MUST follow the exact JSON schema provided"""

USER_PROMPT_TEMPLATE = """Analyze this Indian stock based on the following computed metrics:

STOCK: {symbol} ({company_name})
SECTOR: {sector}

TECHNICAL ANALYSIS (Score: {technical_score}/100):
- Current Price: ₹{current_price}
- Trend: {trend}
- RSI: {rsi} ({rsi_interpretation})
- MACD Signal: {macd_signal_status}
- ATR (Volatility): ₹{atr}
- Support Level: ₹{support_level}
- Resistance Level: ₹{resistance_level}
- 1-Month Price Change: {price_change_1m}%
- 1-Week Price Change: {price_change_1w}%
- Volume Ratio: {volume_ratio}x (vs 20-day average)

FUNDAMENTAL ANALYSIS (Score: {fundamental_score}/100):
- P/E Ratio: {pe_ratio}
- P/B Ratio: {pb_ratio}
- ROCE: {roce}%
- ROE: {roe}%
- Debt/Equity: {debt_to_equity}
- Promoter Holding: {promoter_holding}%
- Pledged Shares: {pledged_pct}%
- Market Cap: ₹{market_cap_cr} Cr ({market_cap_category})
- Revenue Growth (1Y): {revenue_growth}%
- Net Margin: {net_margin}%

MARKET CONTEXT:
- Overall Score: {overall_score}/100
- Market Regime: {market_regime}

RECENT NEWS CONTEXT:
{news_context}

Based on ALL this information, provide your analysis in the required JSON format.
Focus on India-specific insights, risks, and opportunities that the quantitative 
scores alone cannot capture."""

NEWS_ANALYSIS_PROMPT = """Analyze these recent news items for {symbol} and provide a sentiment assessment:

NEWS ITEMS:
{news_items}

Provide a brief sentiment analysis considering:
1. Overall sentiment (positive/negative/neutral)
2. Key risks mentioned
3. Key catalysts mentioned
4. India-specific regulatory or policy implications

Keep the analysis under 200 words."""