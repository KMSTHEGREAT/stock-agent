"""
Bear Researcher Agent for V2.
Argues AGAINST the stock — identifies risks and concerns.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import ResearchThesis
from src.v2.llm import call_llm, extract_json

logger = logging.getLogger(__name__)


def bear_researcher_node(state: dict) -> dict:
    """
    LangGraph node: Bear Researcher.
    
    Takes all analyst data and builds a bearish risk assessment.
    The LLM must use the provided data, not invent metrics.
    """
    symbol = state["symbol"]
    company_name = state.get("company_name", symbol)
    
    logger.info(f"[Bear Researcher] Building bear case for {symbol}")
    
    try:
        # Gather context
        tech = state.get("technical_output")
        fund = state.get("fundamental_output")
        news = state.get("news_output")
        sentiment = state.get("sentiment_output")
        
        # Build prompt
        prompt = _build_bear_prompt(symbol, company_name, tech, fund, news, sentiment)
        
        # Call LLM
        result = extract_json(call_llm(
            prompt=prompt,
            system=_BEAR_SYSTEM_PROMPT,
            temperature=0.4,
            num_predict=500
        ))
        
        if result:
            thesis = ResearchThesis(
                researcher_type="bear",
                symbol=symbol,
                thesis=_safe_str(result.get("thesis", ""),
                                "Bear thesis generation failed"),
                supporting_evidence=_safe_list(result.get("evidence", [])),
                counter_arguments_addressed=_safe_list(
                    result.get("counter_arguments", [])),
                conviction_level=_safe_float(
                    result.get("conviction", 0.5), 0.5),
            )
            
            logger.info(f"[Bear Researcher] {symbol}: conviction={thesis.conviction_level}")
            
            return {"bear_thesis": thesis}
        
        # Fallback
        return {"bear_thesis": _fallback_bear_thesis(symbol, tech, fund, news)}
        
    except Exception as e:
        logger.error(f"[Bear Researcher] Error: {e}")
        return {
            "bear_thesis": _fallback_bear_thesis(symbol, None, None, None),
            "errors": [f"Bear researcher error: {str(e)}"],
        }


_BEAR_SYSTEM_PROMPT = """You are a BEAR researcher — your job is to identify the risks and concerns with this investment.

RULES:
- Use ONLY the data provided below — do NOT invent numbers or metrics
- If data shows something positive, acknowledge it but explain the risks
- Focus on downside risks, potential pitfalls, and unfavorable risk/reward
- Consider Indian market risks (FII outflows, regulatory changes, promoter issues)
- Your conviction should reflect how serious the risks actually are

Respond with ONLY this JSON:
{
  "thesis": "2-3 sentence risk assessment for avoiding this stock",
  "evidence": ["specific risk 1", "specific risk 2", "specific risk 3"],
  "counter_arguments": ["how you counter the main bullish point", "another counter"],
  "conviction": 0.6
}"""


def _build_bear_prompt(symbol, company_name, tech, fund, news, sentiment) -> str:
    """Build a data-rich prompt for the bear researcher."""
    
    context_parts = []
    context_parts.append(f"STOCK: {symbol} ({company_name})")
    context_parts.append(f"DATE: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Technical data — focus on bearish signals
    if tech:
        context_parts.append(f"\nTECHNICAL ANALYSIS (Score: {tech.score:.0f}/100):")
        if tech.current_price:
            context_parts.append(f"- Current Price: ₹{tech.current_price:,.2f}")
        if tech.trend:
            context_parts.append(f"- Trend: {tech.trend}")
        if tech.rsi:
            if tech.rsi > 70:
                context_parts.append(f"- RSI: {tech.rsi:.1f} (OVERBOUGHT — reversal risk)")
            else:
                context_parts.append(f"- RSI: {tech.rsi:.1f}")
        if tech.price_change_1m is not None:
            context_parts.append(f"- 1-Month Change: {tech.price_change_1m:+.1f}%")
        if tech.resistance_level:
            context_parts.append(f"- Resistance: ₹{tech.resistance_level:,.2f}")
        
        # Include all findings (bear will find the negative ones)
        for finding in tech.key_findings[:3]:
            context_parts.append(f"- {finding}")
    
    # Fundamental data — focus on risk indicators
    if fund:
        context_parts.append(f"\nFUNDAMENTAL ANALYSIS (Score: {fund.score:.0f}/100):")
        if fund.pe_ratio:
            if fund.pe_ratio > 30:
                context_parts.append(f"- P/E Ratio: {fund.pe_ratio:.1f} (HIGH valuation)")
            else:
                context_parts.append(f"- P/E Ratio: {fund.pe_ratio:.1f}")
        if fund.debt_to_equity and fund.debt_to_equity > 1:
            context_parts.append(f"- Debt/Equity: {fund.debt_to_equity:.2f} (HIGH leverage)")
        if fund.pledged_pct and fund.pledged_pct > 5:
            context_parts.append(f"- Pledged Shares: {fund.pledged_pct:.1f}% (RISK)")
        if fund.revenue_growth is not None and fund.revenue_growth < 5:
            context_parts.append(f"- Revenue Growth: {fund.revenue_growth:.1f}% (WEAK)")
        
        for finding in fund.key_findings[:3]:
            context_parts.append(f"- {finding}")
    
    # News data
    if news:
        context_parts.append(f"\nNEWS ANALYSIS (Score: {news.score:.0f}/100):")
        context_parts.append(f"- Total Items: {len(news.news_items)}")
        context_parts.append(f"- Negative: {news.negative_count}, Positive: {news.positive_count}")
        
        if news.news_items:
            context_parts.append("- Recent headlines:")
            for item in news.news_items[:3]:
                context_parts.append(f"  * {item.title[:80]}")
    
    # Sentiment
    if sentiment:
        context_parts.append(f"\nSENTIMENT (Score: {sentiment.score:.0f}/100):")
        context_parts.append(f"- Overall: {sentiment.overall_sentiment}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""{context}

TASK: Build the BEAR case against {symbol}.

Focus on:
1. Negative signals and risks in the data
2. Potential downside catalysts
3. Why bullish arguments may not hold
4. India-specific risks (regulatory, FII flows, sector headwinds)

Respond with ONLY valid JSON."""
    
    return prompt


def _fallback_bear_thesis(symbol, tech, fund, news) -> ResearchThesis:
    """Rule-based bear thesis fallback."""
    evidence = []
    
    if tech and tech.score < 40:
        evidence.append(f"Technical score {tech.score:.0f}/100 — weak momentum")
    if tech and tech.trend == "down":
        evidence.append("Downtrend — bearish price action")
    if fund and fund.pe_ratio and fund.pe_ratio > 30:
        evidence.append(f"P/E {fund.pe_ratio:.1f} — expensive valuation")
    if fund and fund.pledged_pct and fund.pledged_pct > 10:
        evidence.append(f"Promoter pledging {fund.pledged_pct:.0f}% — governance risk")
    if fund and fund.debt_to_equity and fund.debt_to_equity > 1.5:
        evidence.append(f"Debt/Equity {fund.debt_to_equity:.2f} — high leverage")
    if news and news.negative_count > 0:
        evidence.append(f"{news.negative_count} negative news items")
    
    if not evidence:
        evidence.append("No significant bearish signals detected")
    
    return ResearchThesis(
        researcher_type="bear",
        symbol=symbol,
        thesis=f"Risk assessment: {symbol} shows some concerns that warrant caution.",
        supporting_evidence=evidence[:4],
        conviction_level=0.4,
    )


def _safe_str(value, default: str) -> str:
    if value is None:
        return default
    return str(value)[:500]


def _safe_list(value) -> list:
    if isinstance(value, list):
        return [str(v)[:150] for v in value[:5]]
    return []


def _safe_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        return max(0.0, min(1.0, float(value)))
    except:
        return default