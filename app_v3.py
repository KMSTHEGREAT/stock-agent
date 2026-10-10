"""
V3 Stock Analysis Agent — Modern Dark UI with LLM Debate.
Complete rewrite with real-time progress updates.

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
    
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    section[data-testid="stSidebar"] {
        background-color: #1a1f2b;
        border-right: 1px solid #2d3748;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    h1 { color: #fafafa; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #e2e8f0; font-weight: 700; font-size: 1.5rem; margin-top: 1.5rem; }
    h3 { color: #94a3b8; font-weight: 600; font-size: 1.1rem; }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2b 0%, #252b3b 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d3748;
    }
    
    div[data-testid="stMetric"] label { color: #64748b; font-size: 0.85rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #fafafa; font-weight: 700; }
    
    .rating-badge {
        display: inline-block;
        padding: 10px 32px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.3em;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 10px 0;
    }
    
    .rating-buy { background: linear-gradient(135deg, #059669, #10b981); color: white; }
    .rating-overweight { background: linear-gradient(135deg, #059669, #34d399); color: white; }
    .rating-hold { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; }
    .rating-underweight { background: linear-gradient(135deg, #c2410c, #f97316); color: white; }
    .rating-sell { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; }
    
    .valuation-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .valuation-value { font-size: 2.5rem; font-weight: 800; color: #3b82f6; }
    
    .news-item {
        padding: 12px 16px;
        border-bottom: 1px solid #2d3748;
        border-radius: 8px;
        margin: 4px 0;
        background: #1a1f2b;
    }
    
    .news-item:hover { background: #252b3b; }
    .news-title { color: #e2e8f0; font-weight: 500; font-size: 0.95rem; }
    .news-source { color: #64748b; font-size: 0.8rem; }
    
    .debate-card {
        padding: 20px;
        border-radius: 12px;
        background: #1a1f2b;
        margin: 12px 0;
    }
    
    .debate-bull { border-left: 4px solid #10b981; }
    .debate-bear { border-left: 4px solid #ef4444; }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a1f2b, #252b3b);
        color: #e2e8f0;
        border: 1px solid #2d3748;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #252b3b, #2d3748);
        border-color: #4b5563;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2b;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #252b3b;
        color: #00d4aa;
    }
</style>
""", unsafe_allow_html=True)


# ============ IMPORTS ============
from src.v3.agents.dcf_valuation import DCFValuationAgent
from src.v3.agents.dupont_analyzer import DuPontAnalyzer
from src.v3.agents.risk_manager_cfa import AdvancedRiskManager
from src.v3.news.multi_source import MultiSourceNewsAggregator
from src.v3.selfimprove.engine import SelfImprovementEngine

# LLM
try:
    from src.v2.llm import call_llm, extract_json
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


# ============ HELPER FUNCTIONS ============

def format_inr(value, decimals=2):
    if value is None:
        return "—"
    return f"₹{value:,.{decimals}f}"


def get_rating_badge(rating):
    return f'<div class="rating-badge rating-{rating.lower()}">{rating.upper()}</div>'


def _build_context(symbol, tech, fund, dcf, dupont, risk, news):
    """Build debate context."""
    parts = [f"STOCK: {symbol}"]
    
    if tech:
        parts.append(f"\nTECHNICAL ({tech['score']:.0f}/100): RSI {tech['rsi']}, Trend {tech['trend']}")
    if fund:
        parts.append(f"FUNDAMENTAL ({fund['score']:.0f}/100): P/E {fund.get('pe_ratio', 'N/A')}")
    if dcf:
        parts.append(f"DCF: Intrinsic ₹{dcf['intrinsic_value']:,.0f}, Upside {dcf['upside_pct']:+.1f}%")
    if dupont:
        parts.append(f"ROE: {dupont['roe']:.1f}%, Quality {dupont['quality_score']:.0f}/100")
    if risk:
        parts.append(f"RISK: {risk['risk_category']}, VaR {risk['var_95']:.1f}%")
    if news:
        parts.append(f"NEWS: {news['sentiment_label']} ({news['sentiment_score']}/100)")
    
    return "\n".join(parts)


def _compute_rating(tech, fund, dcf, dupont, risk, debate):
    """Compute final rating."""
    scores = []
    weights = []
    
    if tech:
        scores.append(tech["score"]); weights.append(0.20)
    if fund:
        scores.append(fund["score"]); weights.append(0.25)
    if dcf and dcf.get("upside_pct") is not None:
        scores.append(50 + max(-25, min(25, dcf["upside_pct"] / 2))); weights.append(0.20)
    if dupont and dupont.get("quality_score") is not None:
        scores.append(dupont["quality_score"]); weights.append(0.10)
    if risk and risk.get("risk_score") is not None:
        scores.append(100 - risk["risk_score"]); weights.append(0.10)
    if debate and debate.get("sentiment_score") is not None:
        scores.append(debate["sentiment_score"]); weights.append(0.15)
    
    if scores:
        overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        overall = 50
    
    if overall >= 75: rating = "BUY"
    elif overall >= 62: rating = "OVERWEIGHT"
    elif overall >= 45: rating = "HOLD"
    elif overall >= 32: rating = "UNDERWEIGHT"
    else: rating = "SELL"
    
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


def display_v3_results(session):
    """Display results."""
    result = session.get('v3_result')
    if not result:
        return
    
    symbol = result['symbol']
    st.markdown(f"## {symbol} — Results")
    
    rating = result.get('rating', 'HOLD')
    score = result.get('overall_score', 50)
    
    st.markdown(get_rating_badge(rating), unsafe_allow_html=True)
    st.metric("Score", f"{score}/100", f"Conviction: {result.get('conviction', 0):.0%}")
    
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
        with col3: st.metric("Stop", format_inr(result['stop_loss']))
    
    st.markdown("---")
    
    # Debate
    debate = result.get('debate')
    if debate:
        st.markdown("### ⚖️ Debate")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""<div class="debate-card debate-bull">
                    <b style="color:#10b981;">🐂 BULL ({debate['bull_conviction']:.0%})</b><br>
                    {debate['bull_thesis']}
                </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(
                f"""<div class="debate-card debate-bear">
                    <b style="color:#ef4444;">🐻 BEAR ({debate['bear_conviction']:.0%})</b><br>
                    {debate['bear_thesis']}
                </div>""", unsafe_allow_html=True)
        
        winner = debate.get('winner', 'inconclusive')
        color = {"bull": "#10b981", "bear": "#ef4444", "inconclusive": "#f59e0b"}.get(winner, "#64748b")
        st.markdown(
            f"""<div style="padding:12px;background:#1a1f2b;border-radius:8px;
                        border-left:4px solid {color};">
                <b style="color:{color};">{winner.upper()}</b> — {debate.get('reasoning', '')}
            </div>""", unsafe_allow_html=True)
    
    # News
    news = result.get('news')
    if news:
        st.markdown(f"### 📰 News ({news['sentiment_label']})")
        for item in news.get('items', [])[:5]:
            st.markdown(
                f"<div class='news-item'>• {item['title']} "
                f"<span class='news-source'>([{item['source']}])</span></div>",
                unsafe_allow_html=True)
    
    st.caption("⚠️ Not investment advice")


if __name__ == "__main__":
    pass

# ============ SESSION STATE ============
if 'v3_result' not in st.session_state:
    st.session_state.v3_result = None
if 'analysis_symbol' not in st.session_state:
    st.session_state.analysis_symbol = None


# ============ SIDEBAR ============
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
**Status:**
- LLM: {'✅ Active' if HAS_LLM else '❌ Offline'}
- SearXNG: ✅
- Self-Improve: ✅

**⚠️ Not investment advice**
""")


# ============ DASHBOARD ============
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
        st.metric("LLM", "✅ Active" if HAS_LLM else "❌ Offline")
    
    with col3:
        st.metric("News Sources", "9 feeds")
    
    with col4:
        try:
            engine = SelfImprovementEngine()
            report = engine.get_improvement_report()
            st.metric("Guardrails", report['active_guardrails'])
        except:
            st.metric("Guardrails", "0")
    
    # Quick analyze
    st.subheader("Quick Analysis")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        quick_symbol = st.text_input("NSE Symbol", placeholder="RELIANCE, TCS, INFY...", key="quick_symbol")
    
    with col2:
        if st.button("Analyze", type="primary", use_container_width=True):
            if quick_symbol:
                st.session_state.analysis_symbol = quick_symbol.upper()
                st.rerun()
    
    # Show previous results
    if st.session_state.get('v3_result'):
        st.markdown("---")
        display_v3_results(st.session_state)
    
    st.markdown("---")
    st.caption("⚠️ For personal research only. Not investment advice.")


# ============ ANALYZE PAGE ============
elif page == "🔍 Analyze Stock":
    st.title("🔍 Analyze Stock")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input("NSE Symbol", key="analyze_symbol", placeholder="RELIANCE")
    
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
        
        # Estimate time
        est = 10
        if include_dcf: est += 5
        if include_risk: est += 8
        if include_news: est += 5
        if include_llm: est += 90
        
        st.info(f"⏱️ Estimated: ~{est} seconds. LLM debate is the slowest part.")
        
        progress_bar = st.progress(0)
        
        # ============ RUN ANALYSIS WITH PROGRESS ============
        
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
                from src.v1.analysis.technical import TechnicalAnalyzer
                from src.v1.analysis.technical_scorer import TechnicalScorer
                
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
                from src.v1.analysis.fundamental import FundamentalAnalyzer
                from src.v1.analysis.fundamental_scorer import FundamentalScorer
                
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
                            "items": [
                                {"title": item.title, "source": item.source}
                                for item in news_items[:10]
                            ],
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
                
                context = _build_context(symbol, tech_result, fund_result, dcf_result, dupont_result, risk_result, news_result)
                
                # Bull
                bull_prompt = f"""{context}

Build the strongest BULL case for {symbol}. Use ONLY the data above.

Respond with ONLY JSON:
{{"thesis": "2-3 sentences", "evidence": ["point 1", "point 2", "point 3"], "conviction": 0.7}}"""
                
                bull = extract_json(call_llm(bull_prompt, temperature=0.4, num_predict=400))
                st.write(f"  ✓ Bull: {bull.get('thesis', '')[:60]}..." if bull else "  ⚠️ Bull failed")
                
                progress_bar.progress(80)
                
                # Bear
                st.write("  🐻 Bear researcher thinking... (20-40s)")
                
                bear_prompt = f"""{context}

Build the strongest BEAR case against {symbol}. Use ONLY the data above.

Respond with ONLY JSON:
{{"thesis": "2-3 sentences", "evidence": ["risk 1", "risk 2", "risk 3"], "conviction": 0.6}}"""
                
                bear = extract_json(call_llm(bear_prompt, temperature=0.4, num_predict=400))
                st.write(f"  ✓ Bear: {bear.get('thesis', '')[:60]}..." if bear else "  ⚠️ Bear failed")
                
                progress_bar.progress(90)
                
                # Moderator
                st.write("  ⚖️ Debate moderator evaluating... (15-30s)")
                
                bull_thesis = bull.get("thesis", "N/A") if bull else "N/A"
                bear_thesis = bear.get("thesis", "N/A") if bear else "N/A"
                
                mod_prompt = f"""Evaluate this debate for {symbol}:

BULL: {bull_thesis}
BEAR: {bear_thesis}

Respond with ONLY JSON:
{{"winner": "bull or bear or inconclusive", "reasoning": "1-2 sentences", "sentiment_score": 55}}"""
                
                debate = extract_json(call_llm(mod_prompt, temperature=0.2, num_predict=300))
                st.write(f"  ✓ Winner: {debate.get('winner', 'N/A')}" if debate else "  ⚠️ Moderation failed")
                
                debate_result = {
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
            else:
                st.write("⏭️ Step 7: LLM Debate Skipped")
            
            progress_bar.progress(95)
            
            # Final Rating
            st.write("🎯 **Computing Final Rating...**")
            
            final = _compute_rating(tech_result, fund_result, dcf_result, dupont_result, risk_result, debate_result)
            
            st.write(f"  ✅ **{final['rating']}** (Score: {final['score']:.1f}/100)")
            
            progress_bar.progress(100)
            status.update(label="✓ Analysis Complete!", state="complete")
        
        progress_bar.empty()
        
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
        st.session_state.analysis_symbol = symbol
        
        st.rerun()
    
    # Display results
    if st.session_state.get('v3_result'):
        display_v3_results(st.session_state)


# ============ CFA METRICS ============
elif page == "📊 CFA Metrics":
    st.title("📊 CFA Analysis")
    
    symbol = st.text_input("Symbol", key="cfa_symbol")
    
    if st.button("Run CFA Analysis", type="primary") and symbol:
        symbol = symbol.upper()
        
        tab1, tab2, tab3 = st.tabs(["💰 DCF", "📈 DuPont", "⚠️ Risk"])
        
        with tab1:
            with st.spinner("DCF..."):
                dcf = DCFValuationAgent().analyze(symbol)
            if dcf:
                st.metric("Intrinsic Value", format_inr(dcf.intrinsic_value_per_share, 0))
                st.metric("Market Price", format_inr(dcf.current_price, 0))
                st.metric("Upside", f"{dcf.upside_downside_pct:+.1f}%")
                st.metric("WACC", f"{dcf.wacc:.1f}%")
                st.metric("Status", dcf.valuation_status)
            else:
                st.warning("DCF unavailable")
        
        with tab2:
            with st.spinner("DuPont..."):
                dupont = DuPontAnalyzer().analyze(symbol)
            if dupont:
                st.metric("ROE", f"{dupont.roe:.1f}%")
                st.metric("ROA", f"{dupont.roa:.1f}%")
                st.metric("Quality", f"{dupont.roe_quality_score:.0f}/100")
                st.metric("Driver", dupont.primary_driver)
                st.info(dupont.quality_interpretation)
            else:
                st.warning("DuPont unavailable")
        
        with tab3:
            with st.spinner("Risk..."):
                risk = AdvancedRiskManager().analyze(symbol)
            if risk:
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("VaR 95%", f"{risk.var_95:.1f}%")
                with col2: st.metric("Sharpe", f"{risk.sharpe_ratio:.2f}")
                with col3: st.metric("Beta", f"{risk.beta:.2f}")
                with col4: st.metric("Risk", risk.risk_category)
            else:
                st.warning("Risk unavailable")


# ============ SELF-IMPROVEMENT ============
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
            error_df = pd.DataFrame(
                list(perf['error_breakdown'].items()),
                columns=['Error Type', 'Count']
            )
            st.dataframe(error_df, use_container_width=True, hide_index=True)
        
        if report.get('suggestions'):
            st.subheader("Suggestions")
            for s in report['suggestions']:
                with st.expander(f"{s['improvement_type']} → {s['target']}"):
                    st.write(s['description'])
                    st.write(f"Reason: {s['reason']}")
        
    except Exception as e:
        st.error(f"Error: {e}")


# ============ MARKET ============
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
            fig.update_layout(
                paper_bgcolor='#0e1117', plot_bgcolor='#1a1f2b',
                font_color='#e2e8f0', height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Market News")
        news_agg = MultiSourceNewsAggregator()
        market_news = news_agg.get_market_news(max_items=10)
        
        for item in market_news[:8]:
            st.markdown(
                f"<div class='news-item'>• {item.title} "
                f"<span class='news-source'>([{item.source}])</span></div>",
                unsafe_allow_html=True
            )
        
    except Exception as e:
        st.error(f"Error: {e}")


# ============ HELPER FUNCTIONS (AT BOTTOM) ============