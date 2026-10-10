"""
V3 Stock Analysis Agent — Modern Dark UI with LLM Debate.
Combines CFA-grade deterministic analysis with LLM-powered bull/bear debate.
Properly ordered: all functions defined before use.

Run: streamlit run app_v3.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import sqlite3
import time
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Page config MUST be first
st.set_page_config(
    page_title="Stock Agent V3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { background-color: #0e1117; font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    section[data-testid="stSidebar"] { background-color: #1a1f2b; border-right: 1px solid #2d3748; }
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    h1 { color: #fafafa; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #e2e8f0; font-weight: 700; font-size: 1.5rem; margin-top: 1.5rem; }
    h3 { color: #94a3b8; font-weight: 600; font-size: 1.1rem; }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, #1a1f2b 0%, #252b3b 100%); border-radius: 12px; padding: 16px; border: 1px solid #2d3748; }
    div[data-testid="stMetric"] label { color: #64748b; font-size: 0.85rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #fafafa; font-weight: 700; }
    .rating-badge { display: inline-block; padding: 10px 32px; border-radius: 10px; font-weight: 800; font-size: 1.3em; text-transform: uppercase; letter-spacing: 1.5px; margin: 10px 0; }
    .rating-buy { background: linear-gradient(135deg, #059669, #10b981); color: white; }
    .rating-overweight { background: linear-gradient(135deg, #059669, #34d399); color: white; }
    .rating-hold { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; }
    .rating-underweight { background: linear-gradient(135deg, #c2410c, #f97316); color: white; }
    .rating-sell { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; }
    .valuation-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #3b82f6; border-radius: 12px; padding: 24px; margin: 16px 0; }
    .valuation-value { font-size: 2.5rem; font-weight: 800; color: #3b82f6; }
    .news-item { padding: 12px 16px; border-bottom: 1px solid #2d3748; border-radius: 8px; margin: 4px 0; background: #1a1f2b; }
    .news-item:hover { background: #252b3b; }
    .news-title { color: #e2e8f0; font-weight: 500; font-size: 0.95rem; }
    .news-source { color: #64748b; font-size: 0.8rem; }
    .si-metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e293b; }
    .si-label { color: #64748b; }
    .si-value { color: #e2e8f0; font-weight: 600; }
    .debate-card { padding: 20px; border-radius: 12px; background: #1a1f2b; margin: 12px 0; }
    .debate-bull { border-left: 4px solid #10b981; }
    .debate-bear { border-left: 4px solid #ef4444; }
    .stButton > button { background: linear-gradient(135deg, #1a1f2b, #252b3b); color: #e2e8f0; border: 1px solid #2d3748; border-radius: 8px; font-weight: 600; }
    .stButton > button:hover { background: linear-gradient(135deg, #252b3b, #2d3748); border-color: #4b5563; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1f2b; border-radius: 8px 8px 0 0; padding: 10px 20px; color: #64748b; }
    .stTabs [aria-selected="true"] { background-color: #252b3b; color: #00d4aa; }
</style>
""", unsafe_allow_html=True)

# ============ PROJECT IMPORTS ============
from src.v3.agents.dcf_valuation import DCFValuationAgent, DCFResult
from src.v3.agents.dupont_analyzer import DuPontAnalyzer, DuPontResult
from src.v3.agents.risk_manager_cfa import AdvancedRiskManager, AdvancedRiskResult
from src.v3.news.multi_source import MultiSourceNewsAggregator
from src.v3.selfimprove.engine import SelfImprovementEngine

# LLM availability
try:
    from src.v2.llm import call_llm, extract_json
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# V1 imports
try:
    from src.v1.analysis.technical import TechnicalAnalyzer
    from src.v1.analysis.technical_scorer import TechnicalScorer
    from src.v1.analysis.fundamental import FundamentalAnalyzer
    from src.v1.analysis.fundamental_scorer import FundamentalScorer
    HAS_V1 = True
except ImportError:
    HAS_V1 = False


# ============================================================================
# ALL HELPER AND DISPLAY FUNCTIONS
# (Defined BEFORE page routing so Pylance doesn't complain)
# ============================================================================

def format_inr(value, decimals=2):
    if value is None:
        return "—"
    return f"₹{value:,.{decimals}f}"


def get_rating_badge(rating):
    return f'<div class="rating-badge rating-{rating.lower()}">{rating.upper()}</div>'


# ============ ANALYSIS HELPER FUNCTIONS ============

def _build_debate_context(symbol, tech, fund, dcf, dupont, risk, news):
    """Build debate context string from all results."""
    parts = [f"STOCK: {symbol}"]
    
    if tech:
        parts.append(f"\nTECHNICAL (Score: {tech['score']:.0f}/100):")
        parts.append(f"  Price: ₹{tech['current_price']:,.0f}")
        parts.append(f"  RSI: {tech['rsi']}")
        parts.append(f"  Trend: {tech['trend']}")
    
    if fund:
        parts.append(f"\nFUNDAMENTAL (Score: {fund['score']:.0f}/100):")
        if fund.get('pe_ratio'):
            parts.append(f"  P/E: {fund['pe_ratio']:.1f}")
        if fund.get('roce'):
            parts.append(f"  ROCE: {fund['roce']:.1f}%")
    
    if dcf:
        parts.append(f"\nDCF VALUATION:")
        parts.append(f"  Intrinsic: ₹{dcf['intrinsic_value']:,.0f}")
        parts.append(f"  Upside: {dcf['upside_pct']:+.1f}%")
        parts.append(f"  Status: {dcf['status']}")
    
    if dupont:
        parts.append(f"\nPROFITABILITY:")
        parts.append(f"  ROE: {dupont['roe']:.1f}%")
        parts.append(f"  Quality: {dupont['quality_score']:.0f}/100")
    
    if risk:
        parts.append(f"\nRISK: {risk['risk_category']}")
        parts.append(f"  VaR: {risk['var_95']:.1f}%, Sharpe: {risk['sharpe']:.2f}")
    
    if news:
        parts.append(f"\nNEWS: {news['sentiment_label']} ({news['sentiment_score']}/100)")
    
    return "\n".join(parts)


def _compute_final_rating_v3(tech, fund, dcf, dupont, risk, debate):
    """Compute final rating from all components."""
    scores = []
    weights = []
    
    if tech:
        scores.append(tech["score"])
        weights.append(0.20)
    if fund:
        scores.append(fund["score"])
        weights.append(0.25)
    if dcf and dcf.get("upside_pct") is not None:
        dcf_score = 50 + max(-25, min(25, dcf["upside_pct"] / 2))
        scores.append(dcf_score)
        weights.append(0.20)
    if dupont and dupont.get("quality_score") is not None:
        scores.append(dupont["quality_score"])
        weights.append(0.10)
    if risk and risk.get("risk_score") is not None:
        scores.append(100 - risk["risk_score"])
        weights.append(0.10)
    if debate and debate.get("sentiment_score") is not None:
        scores.append(debate["sentiment_score"])
        weights.append(0.15)
    
    if scores and weights:
        total = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        overall = total / weight_sum
    else:
        overall = 50
    
    if overall >= 75:
        rating = "BUY"
    elif overall >= 62:
        rating = "OVERWEIGHT"
    elif overall >= 45:
        rating = "HOLD"
    elif overall >= 32:
        rating = "UNDERWEIGHT"
    else:
        rating = "SELL"
    
    conviction = 0.5
    if debate:
        conviction += 0.1 if debate.get("winner") in ("bull", "bear") else -0.05
    if risk and risk.get("risk_score", 50) > 60:
        conviction -= 0.15
    conviction = max(0.1, min(0.95, conviction))
    
    entry = target = stop = None
    if tech and tech.get("current_price"):
        price = tech["current_price"]
        atr = tech.get("atr") or price * 0.02
        if rating in ("BUY", "OVERWEIGHT"):
            entry = round(price * 0.98, 2)
            target = round(price + atr * 2.5, 2)
            stop = round(price - atr * 1.5, 2)
        elif rating == "HOLD":
            entry = round(price * 0.94, 2)
            target = round(price + atr * 2.0, 2)
            stop = round(price - atr * 1.5, 2)
    
    return {"rating": rating, "score": round(overall, 1), "conviction": round(conviction, 2),
            "entry": entry, "target": target, "stop": stop}


def _run_debate_with_progress(symbol, tech, fund, dcf, dupont, risk, news, st_obj):
    """Run LLM debate with progress updates."""
    try:
        from src.v2.llm import call_llm, extract_json
    except ImportError:
        return None
    
    context = _build_debate_context(symbol, tech, fund, dcf, dupont, risk, news)
    
    # === BULL ===
    st_obj.write("   🐂 Bull researcher thinking... (20-40s)")
    
    bull_prompt = f"""{context}

Build the strongest BULL case for {symbol}. Use ONLY the data above.

Respond with ONLY JSON:
{{"thesis": "2-3 sentences", "evidence": ["point 1", "point 2", "point 3"], "conviction": 0.7}}"""
    
    bull = extract_json(call_llm(bull_prompt, temperature=0.4, num_predict=400))
    st_obj.write(f"   ✓ Bull case ready: {bull.get('thesis', '')[:60]}..." if bull else "   ⚠️ Bull case failed")
    
    # === BEAR ===
    st_obj.write("   🐻 Bear researcher thinking... (20-40s)")
    
    bear_prompt = f"""{context}

Build the strongest BEAR case against {symbol}. Use ONLY the data above.

Respond with ONLY JSON:
{{"thesis": "2-3 sentences", "evidence": ["risk 1", "risk 2", "risk 3"], "conviction": 0.6}}"""
    
    bear = extract_json(call_llm(bear_prompt, temperature=0.4, num_predict=400))
    st_obj.write(f"   ✓ Bear case ready: {bear.get('thesis', '')[:60]}..." if bear else "   ⚠️ Bear case failed")
    
    # === MODERATOR ===
    st_obj.write("   ⚖️ Debate moderator evaluating... (15-30s)")
    
    bull_thesis = bull.get("thesis", "N/A") if bull else "N/A"
    bear_thesis = bear.get("thesis", "N/A") if bear else "N/A"
    
    moderator_prompt = f"""Evaluate this debate for {symbol}:

BULL: {bull_thesis}
BEAR: {bear_thesis}

Respond with ONLY JSON:
{{"winner": "bull or bear or inconclusive", "reasoning": "1-2 sentences", "sentiment_score": 55}}"""
    
    debate = extract_json(call_llm(moderator_prompt, temperature=0.2, num_predict=300))
    st_obj.write("   ✓ Debate moderated")
    
    return {
        "bull_thesis": bull_thesis,
        "bull_evidence": bull.get("evidence", []) if bull else [],
        "bull_conviction": bull.get("conviction", 0.5) if bull else 0.5,
        "bear_thesis": bear_thesis,
        "bear_evidence": bear.get("evidence", []) if bear else [],
        "bear_conviction": bear.get("conviction", 0.5) if bear else 0.5,
        "winner": debate.get("winner", "inconclusive") if debate else "inconclusive",
        "reasoning": debate.get("reasoning", "") if debate else "",
        "sentiment_score": debate.get("sentiment_score", 50) if debate else 50,
    }


# ============ DISPLAY FUNCTIONS ============

def display_dcf(dcf: DCFResult):
    """Display DCF valuation results."""
    st.markdown("### 💰 DCF Valuation")
    
    st.markdown(f"""
    <div class="valuation-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="color:#64748b; font-size:0.9rem;">Intrinsic Value</div>
                <div class="valuation-value">{format_inr(dcf.intrinsic_value_per_share, 0)}</div>
            </div>
            <div style="text-align:right;">
                <div style="color:#64748b; font-size:0.9rem;">Market Price</div>
                <div style="color:#fafafa; font-size:2rem; font-weight:800;">{format_inr(dcf.current_price, 0)}</div>
            </div>
            <div style="text-align:right;">
                <div style="color:#64748b; font-size:0.9rem;">Upside</div>
                <div style="color:{'#10b981' if dcf.upside_downside_pct > 0 else '#ef4444'}; font-size:2rem; font-weight:800;">
                    {dcf.upside_downside_pct:+.1f}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    status_colors = {"undervalued": "rating-buy", "fairly_valued": "rating-hold", "overvalued": "rating-sell"}
    badge_class = status_colors.get(dcf.valuation_status, "rating-hold")
    st.markdown(f'<div class="rating-badge {badge_class}" style="font-size:0.9em;">{dcf.valuation_status.replace("_", " ").upper()}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("WACC", f"{dcf.wacc:.1f}%")
    with col2: st.metric("Terminal Growth", f"{dcf.terminal_growth_rate:.1f}%")
    with col3: st.metric("Revenue Growth", f"{dcf.revenue_growth_5y:.1f}%")
    
    if dcf.projected_fcff:
        fig = go.Figure()
        years = list(range(1, len(dcf.projected_fcff) + 1))
        fig.add_trace(go.Bar(
            x=years, y=dcf.projected_fcff,
            marker_color=['#1a1f2b'] * (len(years) - 1) + ['#00d4aa'],
            text=[f"₹{v:,.0f}Cr" for v in dcf.projected_fcff],
            textposition='outside'
        ))
        fig.update_layout(title="Projected FCFF (Crores)", paper_bgcolor='#0e1117',
                         plot_bgcolor='#1a1f2b', font_color='#e2e8f0', height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.info(dcf.interpretation)


def display_dupont(dupont: DuPontResult):
    """Display DuPont analysis."""
    st.markdown("### 📈 DuPont ROE Decomposition")
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("ROE", f"{dupont.roe:.1f}%")
    with col2: st.metric("ROA", f"{dupont.roa:.1f}%")
    with col3: st.metric("Quality", f"{dupont.roe_quality_score:.0f}/100")
    
    fig = go.Figure(go.Bar(
        x=['Op Margin', 'Asset Turn', 'Leverage', 'Tax Burden', 'Int Burden'],
        y=[dupont.operating_margin, min(100, dupont.asset_turnover * 50),
           min(100, dupont.financial_leverage * 20), dupont.tax_burden * 100, dupont.interest_burden * 100],
        marker_color=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'],
    ))
    fig.update_layout(title="ROE Components", paper_bgcolor='#0e1117',
                     plot_bgcolor='#1a1f2b', font_color='#e2e8f0', height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    driver_colors = {"margin": "#10b981", "turnover": "#3b82f6", "leverage": "#ef4444", "bank_model": "#f59e0b"}
    color = driver_colors.get(dupont.primary_driver, "#64748b")
    st.markdown(
        f"<div style='padding:12px 20px;background:#1a1f2b;border-radius:8px;border-left:4px solid {color};'>"
        f"<b style='color:{color};'>{dupont.primary_driver.replace('_', ' ').title()}</b> — {dupont.quality_interpretation}</div>",
        unsafe_allow_html=True)


def display_risk(risk: AdvancedRiskResult):
    """Display risk metrics."""
    st.markdown("### ⚠️ Risk Analysis")
    
    risk_colors = {"low": "rating-buy", "moderate": "rating-hold",
                   "high": "rating-underweight", "very_high": "rating-sell"}
    badge = risk_colors.get(risk.risk_category, "rating-hold")
    st.markdown(f'<div class="rating-badge {badge}" style="font-size:1em;">{risk.risk_category.replace("_", " ").upper()} RISK</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("VaR (95%)", f"{risk.var_95:.1f}%")
    with col2: st.metric("Sharpe", f"{risk.sharpe_ratio:.2f}")
    with col3: st.metric("Beta", f"{risk.beta:.2f}")
    with col4: st.metric("Position", f"{risk.recommended_position_pct:.1f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Max DD", f"{risk.max_drawdown:.1f}%")
    with col2: st.metric("Volatility", f"{risk.annual_volatility:.1f}%")
    with col3: st.metric("Alpha", f"{risk.alpha:.1f}%")
    with col4: st.metric("Sortino", f"{risk.sortino_ratio:.2f}")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk.risk_score, title={'text': "Risk Score"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#ef4444" if risk.risk_score > 60 else "#f59e0b" if risk.risk_score > 40 else "#10b981"},
               'steps': [{'range': [0, 30], 'color': "#1a3a2a"}, {'range': [30, 50], 'color': "#1a3a1a"},
                         {'range': [50, 70], 'color': "#3a2a1a"}, {'range': [70, 100], 'color': "#3a1a1a"}]}
    ))
    fig.update_layout(paper_bgcolor='#0e1117', font_color='#e2e8f0', height=250)
    st.plotly_chart(fig, use_container_width=True)


def display_debate(debate: dict):
    """Display LLM bull/bear debate."""
    st.markdown("### ⚖️ LLM Debate: Bull vs Bear")
    
    if not debate:
        st.warning("LLM debate unavailable")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""<div class="debate-card debate-bull">
                <div style="color:#10b981; font-weight:700; font-size:1.1rem; margin-bottom:8px;">
                    🐂 BULL CASE (Conviction: {debate.get('bull_conviction', 0.5):.0%})
                </div>
                <div style="color:#e2e8f0; margin-bottom:12px;">{debate.get('bull_thesis', 'N/A')}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">
                    <ul>{''.join(f'<li>{e}</li>' for e in debate.get('bull_evidence', [])[:3])}</ul>
                </div>
            </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            f"""<div class="debate-card debate-bear">
                <div style="color:#ef4444; font-weight:700; font-size:1.1rem; margin-bottom:8px;">
                    🐻 BEAR CASE (Conviction: {debate.get('bear_conviction', 0.5):.0%})
                </div>
                <div style="color:#e2e8f0; margin-bottom:12px;">{debate.get('bear_thesis', 'N/A')}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">
                    <ul>{''.join(f'<li>{e}</li>' for e in debate.get('bear_evidence', [])[:3])}</ul>
                </div>
            </div>""", unsafe_allow_html=True)
    
    winner = debate.get("winner", "inconclusive")
    winner_colors = {"bull": "#10b981", "bear": "#ef4444", "inconclusive": "#f59e0b"}
    color = winner_colors.get(winner, "#64748b")
    sentiment = debate.get("sentiment_score", 50)
    
    st.markdown(
        f"""<div style="padding:16px 24px; background:#1a1f2b; border-radius:10px; border-left:4px solid {color}; margin:16px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:{color}; font-weight:700; font-size:1.1rem;">{winner.upper()} WINS</span>
                    <span style="color:#64748b; margin-left:16px;">Sentiment: {sentiment:.0f}/100</span>
                </div>
            </div>
            <div style="color:#94a3b8; margin-top:8px; font-size:0.9rem;">{debate.get('reasoning', '')}</div>
        </div>""", unsafe_allow_html=True)


def display_news(news_items, symbol):
    """Display news with sentiment."""
    st.markdown(f"### 📰 News for {symbol}")
    
    if not news_items:
        st.info("No recent news found")
        return
    
    news_agg = MultiSourceNewsAggregator()
    sentiment = news_agg.detect_sentiment_keywords(news_items)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Sentiment", sentiment['label'].title(), f"{sentiment['score']}/100")
    with col2: st.metric("Items", len(news_items))
    with col3:
        diversity = news_agg.analyze_source_diversity(news_items)
        st.metric("Source Diversity", f"{diversity['diversity_score']}/100")
    
    for item in news_items[:8]:
        icon = {'positive': '📈', 'negative': '📉', 'neutral': '➡️'}.get(
            item.sentiment_hint or 'neutral', '➡️')
        st.markdown(
            f"""<div class="news-item">{icon} <span class="news-title">{item.title}</span><br>
                <span class="news-source">[{item.source}] Reliability: {item.reliability:.0%}</span></div>""",
            unsafe_allow_html=True)


def display_analysis_summary(session):
    """Compact summary for dashboard."""
    result = session.get('v3_result')
    if not result:
        return
    
    symbol = result.get('symbol', '')
    st.subheader(f"Latest Analysis: {symbol}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    dcf = result.get('dcf')
    dupont = result.get('dupont')
    risk = result.get('risk')
    debate = result.get('debate')
    
    with col1:
        if dcf:
            st.metric("DCF Value", format_inr(dcf.get('intrinsic_value'), 0),
                     f"{dcf.get('upside_pct', 0):+.1f}%")
    
    with col2:
        if dupont:
            st.metric("ROE", f"{dupont.get('roe', 0):.1f}%", f"Q: {dupont.get('quality_score', 0):.0f}")
    
    with col3:
        if risk:
            st.metric("Risk", risk.get('risk_category', '—').title(),
                     f"Pos: {risk.get('position_pct', 0):.1f}%")
    
    with col4:
        if debate:
            st.metric("Debate", debate.get('winner', 'N/A').title(),
                     f"Sent: {debate.get('sentiment_score', 50):.0f}")


def display_v3_results(session):
    """Display unified V3 results."""
    result = session.get('v3_result')
    if not result:
        return
    
    symbol = result.get('symbol', '')
    st.markdown(f"## {symbol} — Complete Analysis")
    
    rating = result.get('rating', 'HOLD')
    score = result.get('overall_score', 50)
    
    st.markdown(get_rating_badge(rating), unsafe_allow_html=True)
    st.metric("Overall Score", f"{score}/100")
    
    st.markdown("### Component Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    tech = result.get('technical')
    fund = result.get('fundamental')
    dcf = result.get('dcf')
    risk = result.get('risk')
    
    with col1:
        if tech: st.metric("Technical", f"{tech['score']:.0f}/100")
    with col2:
        if fund: st.metric("Fundamental", f"{fund['score']:.0f}/100")
    with col3:
        if dcf: st.metric("DCF Upside", f"{dcf['upside_pct']:+.1f}%")
    with col4:
        if risk: st.metric("Risk", risk['risk_category'].title())
    
    if result.get('entry_price'):
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Entry", format_inr(result['entry_price']))
        with col2: st.metric("Target", format_inr(result['target_price']))
        with col3: st.metric("Stop Loss", format_inr(result['stop_loss']))
    
    st.markdown("---")
    
    if dcf:
        st.markdown(f"""
        <div class="valuation-card">
            <div style="color:#64748b;">DCF Intrinsic Value</div>
            <div class="valuation-value">{format_inr(dcf['intrinsic_value'], 0)}</div>
            <div style="color:#64748b;">vs Market: {format_inr(dcf['current_price'], 0)}</div>
        </div>""", unsafe_allow_html=True)
    
    dupont = result.get('dupont')
    if dupont:
        st.markdown(f"""
        <div style="padding:16px;background:#1a1f2b;border-radius:8px;margin:8px 0;">
            <b>ROE: {dupont['roe']:.1f}%</b> (Driver: {dupont['primary_driver']}, Quality: {dupont['quality_score']:.0f}/100)
        </div>""", unsafe_allow_html=True)
    
    debate = result.get('debate')
    if debate:
        display_debate(debate)
    
    news = result.get('news')
    if news:
        st.markdown(f"### 📰 News ({news['item_count']} items, {news['sentiment_label']})")
        for item in news.get('items', [])[:5]:
            icon = {'positive': '📈', 'negative': '📉'}.get(item.get('sentiment'), '➡️')
            st.markdown(
                f"<div class='news-item'>{icon} {item['title']} "
                f"<span class='news-source'>([{item['source']}])</span></div>",
                unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("⚠️ For personal research only. Not investment advice.")


# ============ SESSION STATE ============
if 'v3_result' not in st.session_state:
    st.session_state.v3_result = None
if 'analysis_symbol' not in st.session_state:
    st.session_state.analysis_symbol = None


# ============================================================================
# SIDEBAR AND PAGE ROUTING
# (All functions are now defined above — Pylance is happy)
# ============================================================================

st.sidebar.markdown("# 📊 Stock Agent")
st.sidebar.markdown("### V3 — CFA + LLM Debate")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🔍 Analyze Stock", "📊 CFA Metrics", "🧠 Self-Improvement", "🌐 Market"],
    key="nav"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Components:**
- DCF Valuation ✅
- DuPont Analysis ✅
- Risk Metrics (CFA) ✅
- Multi-Source News ✅
- LLM Debate {'✅' if HAS_LLM else '❌ (Ollama not detected)'}
- Self-Improvement ✅

**Model:** Phi-4-mini via Ollama

**⚠️ Not investment advice**
""")


# ============ DASHBOARD PAGE ============
if page == "🏠 Dashboard":
    st.title("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            engine = SelfImprovementEngine()
            metrics = engine.get_performance_metrics()
            st.metric("Accuracy", f"{metrics['overall_accuracy_pct']}%")
        except:
            st.metric("Accuracy", "—")
    
    with col2:
        st.metric("LLM Status", "✅ Active" if HAS_LLM else "❌ Offline")
    
    with col3:
        st.metric("News Sources", "9 feeds")
    
    with col4:
        try:
            engine = SelfImprovementEngine()
            report = engine.get_improvement_report()
            st.metric("Guardrails", report['active_guardrails'])
        except:
            st.metric("Guardrails", "0")
    
    st.subheader("Quick Analysis")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        quick_symbol = st.text_input("NSE Symbol", placeholder="RELIANCE, TCS, INFY...", key="quick_symbol")
    
    with col2:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    
    if analyze_clicked and quick_symbol:
        symbol = quick_symbol.upper()
        
        with st.status(f"Analyzing {symbol}...", expanded=True) as status:
            st.write("📊 Technical analysis...")
            tech_result = None
            try:
                tech_metrics = TechnicalAnalyzer().analyze(symbol, period="1y")
                if tech_metrics:
                    tech_scores = TechnicalScorer().score(tech_metrics)
                    tech_result = {
                        "score": tech_scores.get("total", 50),
                        "rsi": tech_metrics.rsi,
                        "trend": tech_metrics.trend.value if tech_metrics.trend else "neutral",
                        "current_price": tech_metrics.current_price,
                        "atr": tech_metrics.atr,
                    }
                    st.write(f"  ✅ Tech: {tech_result['score']:.0f}/100, RSI: {tech_result['rsi']:.0f}")
            except Exception as e:
                st.write(f"  ❌ {e}")
            
            st.write("📋 Fundamental analysis...")
            fund_result = None
            try:
                fund_metrics = FundamentalAnalyzer().analyze(symbol)
                if fund_metrics:
                    fund_scores = FundamentalScorer().score(fund_metrics)
                    fund_result = {"score": fund_scores.get("total", 50),
                                  "pe_ratio": fund_metrics.pe_ratio}
                    st.write(f"  ✅ Fund: {fund_result['score']:.0f}/100")
            except Exception as e:
                st.write(f"  ❌ {e}")
            
            st.write("💰 DCF valuation...")
            dcf_result = None
            try:
                dcf = DCFValuationAgent().analyze(symbol)
                if dcf:
                    dcf_result = {"intrinsic_value": dcf.intrinsic_value_per_share,
                                 "current_price": dcf.current_price,
                                 "upside_pct": dcf.upside_downside_pct,
                                 "status": dcf.valuation_status}
                    st.write(f"  ✅ DCF: ₹{dcf.intrinsic_value_per_share:,.0f} ({dcf.upside_downside_pct:+.1f}%)")
            except Exception as e:
                st.write(f"  ❌ {e}")
            
            st.write("📈 DuPont analysis...")
            dupont_result = None
            try:
                dupont = DuPontAnalyzer().analyze(symbol)
                if dupont:
                    dupont_result = {"roe": dupont.roe,
                                    "primary_driver": dupont.primary_driver,
                                    "quality_score": dupont.roe_quality_score}
                    st.write(f"  ✅ ROE: {dupont.roe:.1f}% (Q: {dupont.roe_quality_score:.0f})")
            except Exception as e:
                st.write(f"  ❌ {e}")
            
            st.write("⚠️ Risk analysis...")
            risk_result = None
            try:
                risk = AdvancedRiskManager().analyze(symbol)
                if risk:
                    risk_result = {"risk_category": risk.risk_category,
                                  "risk_score": risk.risk_score,
                                  "var_95": risk.var_95,
                                  "sharpe": risk.sharpe_ratio,
                                  "beta": risk.beta,
                                  "position_pct": risk.recommended_position_pct}
                    st.write(f"  ✅ {risk.risk_category} risk, Sharpe: {risk.sharpe_ratio:.2f}")
            except Exception as e:
                st.write(f"  ❌ {e}")
            
            st.write("📰 News analysis...")
            news_result = None
            try:
                news_agg = MultiSourceNewsAggregator()
                news_items = news_agg.get_stock_news(symbol)
                if news_items:
                    sentiment = news_agg.detect_sentiment_keywords(news_items)
                    news_result = {"sentiment_score": sentiment.get("score", 50),
                                  "sentiment_label": sentiment.get("label", "neutral"),
                                  "item_count": len(news_items),
                                  "items": [{"title": i.title, "source": i.source} for i in news_items[:10]]}
                    st.write(f"  ✅ {len(news_items)} items ({sentiment.get('label')})")
            except Exception as e:
                st.write(f"  ❌ {e}")
            
            if HAS_LLM:
                st.write("⚖️ LLM Debate (60-120s)...")
                debate_result = _run_debate_with_progress(
                    symbol, tech_result, fund_result, dcf_result,
                    dupont_result, risk_result, news_result, st)
            else:
                debate_result = None
            
            st.write("🎯 Computing rating...")
            final = _compute_final_rating_v3(
                tech_result, fund_result, dcf_result,
                dupont_result, risk_result, debate_result)
            
            st.write(f"  ✅ **{final['rating']}** (Score: {final['score']:.1f}/100)")
            status.update(label="✓ Complete!", state="complete")
        
        # Store result
        st.session_state.v3_result = {
            "symbol": symbol,
            "technical": tech_result,
            "fundamental": fund_result,
            "dcf": dcf_result,
            "dupont": dupont_result,
            "risk": risk_result,
            "news": news_result,
            "debate": debate_result,
            "rating": final["rating"],
            "overall_score": final["score"],
            "conviction": final["conviction"],
            "entry_price": final.get("entry"),
            "target_price": final.get("target"),
            "stop_loss": final.get("stop"),
        }
        
        # Show results immediately below
        st.markdown("---")
        display_v3_results(st.session_state)
    
    
    


# ============ ANALYZE PAGE ============
elif page == "🔍 Analyze Stock":
    st.title("🔍 Analyze Stock")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prefill = st.session_state.get("analyze_symbol", "")
        symbol = st.text_input("NSE Symbol", value=prefill, key="analyze_symbol", placeholder="RELIANCE")
    
    with col2:
        run_clicked = st.button("Run Full Analysis", type="primary", use_container_width=True)
    
    with st.expander("⚙️ Options"):
        col1, col2 = st.columns(2)
        with col1:
            include_llm = st.checkbox("Include LLM Debate (slower)", value=HAS_LLM)
            include_news = st.checkbox("Include News Analysis", value=True)
        with col2:
            include_dcf = st.checkbox("Include DCF Valuation", value=True)
            include_risk = st.checkbox("Include Risk Analysis", value=True)
    
    if run_clicked and symbol:
        symbol = symbol.upper()
        
        est_time = 10
        if include_dcf: est_time += 5
        if include_risk: est_time += 8
        if include_news: est_time += 5
        if include_llm: est_time += 90
        
        st.info(f"⏱️ Estimated: ~{est_time} seconds. LLM debate is the slowest part.")
        
        progress_bar = st.progress(0)
        
        tech_result = None
        fund_result = None
        dcf_result = None
        dupont_result = None
        risk_result = None
        news_result = None
        debate_result = None
        
        with st.status("Starting analysis...", expanded=True) as status:
            
            # STEP 1: Technical
            st.write("📊 **Step 1/7: Technical Analysis**")
            progress_bar.progress(5)
            try:
                tech_metrics = TechnicalAnalyzer().analyze(symbol, period="1y")
                if tech_metrics:
                    tech_scores = TechnicalScorer().score(tech_metrics)
                    tech_result = {
                        "score": tech_scores.get("total", 50),
                        "rsi": tech_metrics.rsi,
                        "trend": tech_metrics.trend.value if tech_metrics.trend else "neutral",
                        "current_price": tech_metrics.current_price,
                        "support": tech_metrics.support_level,
                        "resistance": tech_metrics.resistance_level,
                        "atr": tech_metrics.atr,
                    }
                    st.write(f"  ✅ Score: {tech_result['score']:.0f}/100, RSI: {tech_result['rsi']:.0f}, Trend: {tech_result['trend']}")
                else:
                    st.write("  ⚠️ No data")
            except Exception as e:
                st.write(f"  ❌ Failed: {e}")
            
            progress_bar.progress(15)
            
            # STEP 2: Fundamental
            st.write("📋 **Step 2/7: Fundamental Analysis**")
            try:
                fund_metrics = FundamentalAnalyzer().analyze(symbol)
                if fund_metrics:
                    fund_scores = FundamentalScorer().score(fund_metrics)
                    fund_result = {
                        "score": fund_scores.get("total", 50),
                        "pe_ratio": fund_metrics.pe_ratio,
                        "roce": fund_metrics.roce,
                        "promoter_holding": fund_metrics.promoter_holding,
                    }
                    st.write(f"  ✅ Score: {fund_result['score']:.0f}/100, P/E: {fund_result.get('pe_ratio', 'N/A')}")
                else:
                    st.write("  ⚠️ No data")
            except Exception as e:
                st.write(f"  ❌ Failed: {e}")
            
            progress_bar.progress(25)
            
            # STEP 3: DCF
            if include_dcf:
                st.write("💰 **Step 3/7: DCF Valuation**")
                try:
                    dcf = DCFValuationAgent().analyze(symbol)
                    if dcf:
                        dcf_result = {
                            "intrinsic_value": dcf.intrinsic_value_per_share,
                            "current_price": dcf.current_price,
                            "upside_pct": dcf.upside_downside_pct,
                            "status": dcf.valuation_status,
                            "wacc": dcf.wacc,
                        }
                        st.write(f"  ✅ Intrinsic: ₹{dcf.intrinsic_value_per_share:,.0f}, Upside: {dcf.upside_downside_pct:+.1f}%")
                    else:
                        st.write("  ⚠️ Insufficient data")
                except Exception as e:
                    st.write(f"  ❌ Failed: {e}")
            else:
                st.write("⏭️ Step 3: DCF Skipped")
            
            progress_bar.progress(40)
            
            # STEP 4: DuPont
            st.write("📈 **Step 4/7: DuPont Analysis**")
            try:
                dupont = DuPontAnalyzer().analyze(symbol)
                if dupont:
                    dupont_result = {
                        "roe": dupont.roe,
                        "primary_driver": dupont.primary_driver,
                        "quality_score": dupont.roe_quality_score,
                        "operating_margin": dupont.operating_margin,
                        "leverage": dupont.financial_leverage,
                    }
                    st.write(f"  ✅ ROE: {dupont.roe:.1f}%, Driver: {dupont.primary_driver}, Quality: {dupont.roe_quality_score:.0f}/100")
                else:
                    st.write("  ⚠️ No data")
            except Exception as e:
                st.write(f"  ❌ Failed: {e}")
            
            progress_bar.progress(50)
            
            # STEP 5: Risk
            if include_risk:
                st.write("⚠️ **Step 5/7: Risk Analysis**")
                st.write("  (Fetching 2 years of price data...)")
                try:
                    risk = AdvancedRiskManager().analyze(symbol)
                    if risk:
                        risk_result = {
                            "risk_category": risk.risk_category,
                            "risk_score": risk.risk_score,
                            "var_95": risk.var_95,
                            "sharpe": risk.sharpe_ratio,
                            "beta": risk.beta,
                            "max_drawdown": risk.max_drawdown,
                            "position_pct": risk.recommended_position_pct,
                        }
                        st.write(f"  ✅ {risk.risk_category} risk, VaR: {risk.var_95:.1f}%, Sharpe: {risk.sharpe_ratio:.2f}")
                    else:
                        st.write("  ⚠️ No data")
                except Exception as e:
                    st.write(f"  ❌ Failed: {e}")
            else:
                st.write("⏭️ Step 5: Risk Skipped")
            
            progress_bar.progress(60)
            
            # STEP 6: News
            if include_news:
                st.write("📰 **Step 6/7: News Analysis**")
                st.write("  (Fetching from 9 RSS feeds...)")
                try:
                    news_agg = MultiSourceNewsAggregator()
                    news_items = news_agg.get_stock_news(symbol)
                    
                    if news_items:
                        sentiment = news_agg.detect_sentiment_keywords(news_items)
                        news_result = {
                            "sentiment_score": sentiment.get("score", 50),
                            "sentiment_label": sentiment.get("label", "neutral"),
                            "item_count": len(news_items),
                            "items": [{"title": item.title, "source": item.source} for item in news_items[:10]],
                        }
                        st.write(f"  ✅ {len(news_items)} items, Sentiment: {sentiment.get('label')} ({sentiment.get('score')}/100)")
                    else:
                        st.write("  ⚠️ No news found")
                except Exception as e:
                    st.write(f"  ❌ Failed: {e}")
            else:
                st.write("⏭️ Step 6: News Skipped")
            
            progress_bar.progress(70)
            
            # STEP 7: LLM Debate
            if include_llm and HAS_LLM:
                st.write("⚖️ **Step 7/7: LLM Debate (Bull vs Bear)**")
                st.write("  🐂 Bull researcher thinking... (20-40s)")
                
                debate_result = _run_debate_with_progress(
                    symbol, tech_result, fund_result, dcf_result,
                    dupont_result, risk_result, news_result, st
                )
                
                if debate_result:
                    st.write(f"  ✅ Debate: {debate_result.get('winner', 'N/A')} wins, Sentiment: {debate_result.get('sentiment_score', 50)}/100")
                else:
                    st.write("  ⚠️ LLM debate unavailable")
            else:
                st.write("⏭️ Step 7: LLM Debate Skipped")
            
            progress_bar.progress(90)
            
            # FINAL RATING
            st.write("🎯 **Computing Final Rating...**")
            
            final = _compute_final_rating_v3(
                tech_result, fund_result, dcf_result,
                dupont_result, risk_result, debate_result
            )
            
            st.write(f"  ✅ **{final['rating']}** (Score: {final['score']:.1f}/100)")
            
            progress_bar.progress(100)
            status.update(label="✓ Analysis Complete!", state="complete")
        
        progress_bar.empty()
        
        st.session_state.v3_result = {
            "symbol": symbol,
            "technical": tech_result,
            "fundamental": fund_result,
            "dcf": dcf_result,
            "dupont": dupont_result,
            "risk": risk_result,
            "news": news_result,
            "debate": debate_result,
            "rating": final["rating"],
            "overall_score": final["score"],
            "conviction": final["conviction"],
            "entry_price": final.get("entry"),
            "target_price": final.get("target"),
            "stop_loss": final.get("stop"),
        }
        st.session_state.analysis_symbol = symbol
        
        st.rerun()
    
    if st.session_state.get('v3_result'):
        display_v3_results(st.session_state)


# ============ CFA METRICS PAGE ============
elif page == "📊 CFA Metrics":
    st.title("📊 CFA Analysis")
    
    cfa_symbol = st.text_input("Symbol", key="cfa_symbol")
    
    if st.button("Run CFA Analysis", type="primary") and cfa_symbol:
        cfa_symbol = cfa_symbol.upper()
        
        tab1, tab2, tab3 = st.tabs(["💰 DCF", "📈 DuPont", "⚠️ Risk"])
        
        with tab1:
            with st.spinner("DCF..."):
                dcf = DCFValuationAgent().analyze(cfa_symbol)
            if dcf:
                display_dcf(dcf)
            else:
                st.warning("DCF unavailable")
        
        with tab2:
            with st.spinner("DuPont..."):
                dupont = DuPontAnalyzer().analyze(cfa_symbol)
            if dupont:
                display_dupont(dupont)
            else:
                st.warning("DuPont unavailable")
        
        with tab3:
            with st.spinner("Risk..."):
                risk = AdvancedRiskManager().analyze(cfa_symbol)
            if risk:
                display_risk(risk)
            else:
                st.warning("Risk unavailable")


# ============ SELF-IMPROVEMENT PAGE ============
elif page == "🧠 Self-Improvement":
    st.title("🧠 Self-Improvement")
    
    try:
        engine = SelfImprovementEngine()
        report = engine.get_improvement_report()
        perf = report['performance']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Accuracy", f"{perf['overall_accuracy_pct']}%")
        with col2: st.metric("Evaluated", perf['total_evaluated'])
        with col3: st.metric("Correct", perf['total_correct'])
        with col4: st.metric("Guardrails", report['active_guardrails'])
        
        if perf.get('error_breakdown'):
            st.subheader("Errors")
            error_df = pd.DataFrame(list(perf['error_breakdown'].items()), columns=['Error Type', 'Count'])
            st.dataframe(error_df, use_container_width=True, hide_index=True)
        
        if report.get('suggestions'):
            st.subheader("Suggestions")
            for s in report['suggestions']:
                with st.expander(f"{s['improvement_type']} → {s['target']}"):
                    st.write(s['description'])
                    st.write(f"Reason: {s['reason']}")
        
    except Exception as e:
        st.error(f"Error: {e}")


# ============ MARKET PAGE ============
elif page == "🌐 Market":
    st.title("🌐 Market")
    
    try:
        import yfinance as yf
        
        st.subheader("Nifty 50")
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="6mo")
        
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Close'],
                mode='lines', name='Nifty 50',
                line=dict(color='#00d4aa', width=2)
            ))
            fig.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#1a1f2b',
                             font_color='#e2e8f0', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Market News")
        news_agg = MultiSourceNewsAggregator()
        market_news = news_agg.get_market_news(max_items=10)
        
        for item in market_news[:8]:
            st.markdown(
                f"<div class='news-item'>• {item.title} "
                f"<span class='news-source'>([{item.source}])</span></div>",
                unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    pass
