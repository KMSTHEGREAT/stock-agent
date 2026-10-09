"""
Bull Researcher Agent for V2.
Argues FOR the stock — builds the strongest possible bull case
using data from all analysts.
"""
import logging
from datetime import datetime, timezone

from src.v2.graph.state import ResearchThesis
from src.v2.llm import call_llm, extract_json

logger = logging.getLogger(__name__)


def bull_researcher_node(state: dict) -> dict:
    """
    LangGraph node: Bull Researcher.
    
    Takes all analyst data and builds a bullish investment thesis.
    The LLM must use the provided data, not invent metrics.
    """
    symbol = state["symbol"]
    company_name = state.get("company_name", symbol)
    
    logger.info(f"[Bull Researcher] Building bull case for {symbol}")
    
    try:
        # Gather context from analysts
        tech = state.get("technical_output")
        fund = state.get("fundamental_output")
        news = state.get("news_output")
        sentiment = state.get("sentiment_output")
        
        # Build the prompt with actual data
        prompt = _build_bull_prompt(symbol, company_name, tech, fund, news, sentiment)
        
        # Call LLM
        result = extract_json(call_llm(
            prompt=prompt,
            system=_BULL_SYSTEM_PROMPT,
            temperature=0.4,  # Slightly creative but grounded
            num_predict=500
        ))
        
        if result:
            thesis = ResearchThesis(
                researcher_type="bull",
                symbol=symbol,
                thesis=_safe_str(result.get("thesis", ""), 
                                "Bull thesis generation failed"),
                supporting_evidence=_safe_list(result.get("evidence", [])),
                counter_arguments_addressed=_safe_list(
                    result.get("counter_arguments", [])),
                conviction_level=_safe_float(
                    result.get("conviction", 0.5), 0.5),
            )
            
            logger.info(f"[Bull Researcher] {symbol}: conviction={thesis.conviction_level}")
            
            return {"bull_thesis": thesis}
        
        # Fallback: simple rule-based bull thesis
        return {"bull_thesis": _fallback_bull_thesis(symbol, tech, fund, news)}
        
    except Exception as e:
        logger.error(f"[Bull Researcher] Error: {e}")
        return {
            "bull_thesis": _fallback_bull_thesis(symbol, None, None, None),
            "errors": [f"Bull researcher error: {str(e)}"],
        }


_BULL_SYSTEM_PROMPT = """You are a BULL researcher — your job is to build the strongest possible investment case FOR this stock.

RULES:
- Use ONLY the data provided below — do NOT invent numbers or metrics
- If data shows something negative, acknowledge it but explain why it's not a dealbreaker
- Focus on upside potential, positive catalysts, and favorable risk/reward
- Consider Indian market context (FII flows, sector trends, RBI policy)
- Your conviction should reflect how strong the data actually is (don't overstate)

Respond with ONLY this JSON:
{
  "thesis": "2-3 sentence investment thesis for buying this stock",
  "evidence": ["specific evidence 1", "specific evidence 2", "specific evidence 3"],
  "counter_arguments": ["how you address the main bearish point", "how you address another concern"],
  "conviction": 0.7
}"""


def _build_bull_prompt(symbol, company_name, tech, fund, news, sentiment) -> str:
    """Build a data-rich prompt for the bull researcher."""
    
    context_parts = []
    context_parts.append(f"STOCK: {symbol} ({company_name})")
    context_parts.append(f"DATE: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Technical data
    if tech:
        context_parts.append(f"\nTECHNICAL ANALYSIS (Score: {tech.score:.0f}/100):")
        if tech.current_price:
            context_parts.append(f"- Current Price: ₹{tech.current_price:,.2f}")
        if tech.rsi:
            rsi_status = "oversold" if tech.rsi < 30 else "overbought" if tech.rsi > 70 else "neutral"
            context_parts.append(f"- RSI: {tech.rsi:.1f} ({rsi_status})")
        if tech.trend:
            context_parts.append(f"- Trend: {tech.trend}")
        if tech.support_level:
            context_parts.append(f"- Support: ₹{tech.support_level:,.2f}")
        if tech.price_change_1m is not None:
            context_parts.append(f"- 1-Month Change: {tech.price_change_1m:+.1f}%")
        
        for finding in tech.key_findings[:3]:
            context_parts.append(f"- {finding}")
    
    # Fundamental data
    if fund:
        context_parts.append(f"\nFUNDAMENTAL ANALYSIS (Score: {fund.score:.0f}/100):")
        if fund.pe_ratio:
            context_parts.append(f"- P/E Ratio: {fund.pe_ratio:.1f}")
        if fund.roce:
            context_parts.append(f"- ROCE: {fund.roce:.1f}%")
        if fund.promoter_holding is not None:
            context_parts.append(f"- Promoter Holding: {fund.promoter_holding:.1f}%")
        if fund.pledged_pct is not None and fund.pledged_pct > 0:
            context_parts.append(f"- Pledged: {fund.pledged_pct:.1f}%")
        if fund.revenue_growth is not None:
            context_parts.append(f"- Revenue Growth: {fund.revenue_growth:.1f}%")
        
        for finding in fund.key_findings[:3]:
            context_parts.append(f"- {finding}")
    
    # News data
    if news:
        context_parts.append(f"\nNEWS ANALYSIS (Score: {news.score:.0f}/100):")
        context_parts.append(f"- Total Items: {len(news.news_items)}")
        context_parts.append(f"- Positive: {news.positive_count}, Negative: {news.negative_count}")
        
        if news.news_items:
            context_parts.append("- Recent headlines:")
            for item in news.news_items[:3]:
                context_parts.append(f"  * {item.title[:80]}")
        
        if news.key_findings:
            for finding in news.key_findings[:2]:
                context_parts.append(f"- {finding}")
    
    # Sentiment
    if sentiment:
        context_parts.append(f"\nSENTIMENT (Score: {sentiment.score:.0f}/100):")
        context_parts.append(f"- Overall: {sentiment.overall_sentiment}")
        
        if sentiment.key_findings:
            for finding in sentiment.key_findings[:2]:
                context_parts.append(f"- {finding}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""{context}

TASK: Build the strongest BULL case for {symbol}.

Focus on:
1. Positive signals in the data
2. Potential upside catalysts
3. Why current concerns are manageable
4. India-specific tailwinds if applicable

Respond with ONLY valid JSON."""
    
    return prompt


def _fallback_bull_thesis(symbol, tech, fund, news) -> ResearchThesis:
    """Rule-based bull thesis fallback."""
    evidence = []
    
    if fund and fund.score > 60:
        evidence.append(f"Fundamental score {fund.score:.0f}/100 — decent fundamentals")
    if fund and fund.roce and fund.roce > 15:
        evidence.append(f"ROCE {fund.roce:.1f}% indicates good capital efficiency")
    if fund and fund.promoter_holding and fund.promoter_holding > 45:
        evidence.append(f"Promoter holding {fund.promoter_holding:.0f}% shows commitment")
    if tech and tech.rsi and tech.rsi < 35:
        evidence.append(f"RSI {tech.rsi:.0f} suggests oversold — potential bounce")
    if news and news.positive_count > 0:
        evidence.append(f"{news.positive_count} positive news items")
    
    if not evidence:
        evidence.append("No strong bullish signals detected")
    
    return ResearchThesis(
        researcher_type="bull",
        symbol=symbol,
        thesis=f"Rule-based analysis: {symbol} shows moderate investment potential based on available data.",
        supporting_evidence=evidence[:4],
        conviction_level=0.4,  # Low conviction for fallback
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