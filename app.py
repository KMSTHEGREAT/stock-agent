"""
Streamlit web UI for the Indian Stock Analysis Agent.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import sqlite3
from datetime import datetime

st.set_page_config(
    page_title="Stock Analysis Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.pipeline import StockAnalysisPipeline
from config.settings import DATABASE, DISCLAIMER

# Session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


def get_pipeline():
    if st.session_state.pipeline is None:
        with st.spinner("Initializing..."):
            st.session_state.pipeline = StockAnalysisPipeline()
    return st.session_state.pipeline


def format_inr(value):
    if value is None:
        return "N/A"
    return f"₹{value:,.2f}"


# ============ SIDEBAR ============
st.sidebar.title("📈 Stock Analysis")
st.sidebar.caption("Indian Market (NSE) • Personal Research")

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🔍 Analyze",
    "📋 Watchlist",
    "📜 History",
    "🌐 Market"
])


# ============ DASHBOARD ============
if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    
    pipeline = get_pipeline()
    watchlist = pipeline.get_watchlist()
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Watchlist", len(watchlist))
    
    with col2:
        try:
            from v1.analysis.market_context import MarketContextAnalyzer
            ctx = MarketContextAnalyzer().get_market_context()
            st.metric("Market", ctx.market_regime.upper())
        except:
            st.metric("Market", "N/A")
    
    with col3:
        try:
            conn = sqlite3.connect(DATABASE["path"])
            today = datetime.now().strftime("%Y-%m-%d")
            count = conn.execute(
                "SELECT COUNT(*) FROM analysis_results WHERE date(analysis_timestamp)=?",
                (today,)
            ).fetchone()[0]
            conn.close()
            st.metric("Today", count)
        except:
            st.metric("Today", 0)
    
    # Quick analyze
    st.subheader("Quick Analysis")
    symbol = st.text_input("NSE Symbol", placeholder="e.g., RELIANCE")
    
    if st.button("Analyze", type="primary") and symbol:
        with st.spinner(f"Analyzing {symbol.upper()}..."):
            result = pipeline.analyze_stock(symbol.upper())
        
        if result:
            st.session_state.last_result = result
            st.success(f"✓ {symbol.upper()} scored {result['overall_score']}/100")
            st.rerun()
        else:
            st.error(f"Failed to analyze {symbol}")
    
    # Recent analyses
    st.subheader("Recent Analyses")
    try:
        conn = sqlite3.connect(DATABASE["path"])
        rows = conn.execute("""
            SELECT symbol, overall_score, recommendation, current_price, analysis_timestamp
            FROM analysis_results ORDER BY analysis_timestamp DESC LIMIT 10
        """).fetchall()
        conn.close()
        
        if rows:
            df = pd.DataFrame(rows, columns=[
                'Symbol', 'Score', 'Rec', 'Price (₹)', 'Date'
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No analyses yet. Analyze a stock above!")
    except:
        st.info("No history available.")
    
    st.caption(DISCLAIMER)


# ============ ANALYZE ============
elif page == "🔍 Analyze":
    st.title("🔍 Analyze Stock")
    
    symbol = st.text_input("NSE Symbol", key="analyze_symbol", placeholder="RELIANCE")
    
    col1, col2 = st.columns(2)
    with col1:
        include_news = st.checkbox("Include News", value=True)
    with col2:
        include_llm = st.checkbox("Include LLM Analysis", value=True)
    
    if st.button("Run Analysis", type="primary") and symbol:
        pipeline = get_pipeline()
        
        with st.spinner(f"Analyzing {symbol.upper()}... (1-3 minutes)"):
            result = pipeline.analyze_stock(
                symbol.upper(),
                include_news=include_news,
                include_llm=include_llm
            )
        
        if result:
            st.session_state.last_result = result
            
            # Display results
            symbol_res = result["symbol"]
            overall = result["overall_score"]
            rec = result["recommendation"].upper()
            scores = result.get("scores", {})
            tech = result.get("technical_metrics", {})
            
            # Header
            st.subheader(f"{symbol_res} - {result.get('company_name', symbol_res)}")
            
            # Score gauge
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=overall,
                    title={'text': "Overall Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightcoral"},
                            {'range': [40, 60], 'color': "lightyellow"},
                            {'range': [60, 80], 'color': "lightgreen"},
                            {'range': [80, 100], 'color': "green"},
                        ],
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                colors = {
                    'STRONG_BUY': 'green', 'BUY': 'lightgreen',
                    'HOLD': 'orange', 'SELL': 'lightcoral', 'STRONG_SELL': 'red'
                }
                color = colors.get(rec, 'gray')
                st.markdown(
                    f"<h2 style='color:{color};text-align:center;'>{rec}</h2>",
                    unsafe_allow_html=True
                )
                
                st.write("**Scores:**")
                st.write(f"Tech: {scores.get('technical_score', 0):.0f}")
                st.write(f"Fund: {scores.get('fundamental_score', 0):.0f}")
                st.write(f"News: {scores.get('news_score', 0):.0f}")
                st.write(f"Macro: {scores.get('macro_score', 0):.0f}")
            
            with col3:
                st.write("**Key Metrics:**")
                st.write(f"Price: {format_inr(tech.get('current_price'))}")
                st.write(f"RSI: {tech.get('rsi', 'N/A')}")
                st.write(f"Trend: {tech.get('trend', 'N/A')}")
                
                rec_det = result.get("recommendation_details", {})
                if rec_det.get('entry_price'):
                    st.write(f"Entry: {format_inr(rec_det['entry_price'])}")
                if rec_det.get('stop_loss'):
                    st.write(f"Stop: {format_inr(rec_det['stop_loss'])}")
                if rec_det.get('target_price'):
                    st.write(f"Target: {format_inr(rec_det['target_price'])}")
            
            # LLM Analysis
            llm = result.get("llm_analysis")
            if llm:
                st.subheader("🤖 LLM Analysis")
                
                with st.expander("Summary", expanded=True):
                    st.write(llm.get("analysis_summary", "N/A"))
                    st.caption(f"Confidence: {llm.get('confidence_level', 0):.0%}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if llm.get("key_risks"):
                        st.markdown("**⚠️ Risks:**")
                        for risk in llm["key_risks"][:3]:
                            st.write(f"• {risk}")
                
                with col2:
                    if llm.get("key_catalysts"):
                        st.markdown("**✅ Catalysts:**")
                        for cat in llm["key_catalysts"][:3]:
                            st.write(f"• {cat}")
            
            # News
            news = result.get("news_analysis")
            if news:
                st.subheader("📰 News")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{news.get('news_score', 50):.0f}/100")
                with col2:
                    st.metric("Sentiment", news.get("sentiment_label", "N/A").title())
                with col3:
                    st.metric("Items", news.get("item_count", 0))
                
                if news.get("summary"):
                    st.info(news["summary"])
            
            # Price chart
            st.subheader("📈 Price Chart")
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{symbol_res}.NS")
                hist = ticker.history(period="1y")
                
                if not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'],
                        name=symbol_res
                    ))
                    
                    # Support/Resistance lines
                    if tech.get('support_level'):
                        fig.add_hline(y=tech['support_level'], line_dash="dot",
                                    line_color="green",
                                    annotation_text=f"Support: ₹{tech['support_level']:.0f}")
                    if tech.get('resistance_level'):
                        fig.add_hline(y=tech['resistance_level'], line_dash="dot",
                                    line_color="red",
                                    annotation_text=f"Resistance: ₹{tech['resistance_level']:.0f}")
                    
                    fig.update_layout(
                        height=400,
                        yaxis_title="Price (₹)",
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Chart unavailable.")
            
            # Add to watchlist
            if st.button(f"Add {symbol_res} to Watchlist"):
                pipeline.add_to_watchlist(symbol_res, result.get("company_name"))
                st.success(f"✓ Added {symbol_res}")
            
            st.caption(DISCLAIMER)
        else:
            st.error("Analysis failed.")


# ============ WATCHLIST ============
elif page == "📋 Watchlist":
    st.title("📋 Watchlist")
    
    pipeline = get_pipeline()
    
    # Add symbol
    col1, col2 = st.columns([3, 1])
    with col1:
        new_sym = st.text_input("Add Symbol", placeholder="NSE symbol")
    with col2:
        if st.button("Add") and new_sym:
            pipeline.add_to_watchlist(new_sym.upper())
            st.success(f"✓ Added {new_sym.upper()}")
            st.rerun()
    
    # Show watchlist
    items = pipeline.get_watchlist()
    
    if items:
        st.subheader(f"Watchlist ({len(items)} stocks)")
        
        # Analyze all button
        if st.button("📊 Analyze All", type="primary"):
            symbols = [item["symbol"] for item in items]
            
            progress = st.progress(0)
            status = st.empty()
            
            results = []
            for i, sym in enumerate(symbols):
                status.text(f"Analyzing {sym} ({i+1}/{len(symbols)})...")
                progress.progress(i / len(symbols))
                
                result = pipeline.analyze_stock(sym, include_news=False)
                if result:
                    results.append(result)
            
            progress.progress(1.0)
            status.text("Complete!")
            
            if results:
                results = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
                
                # Summary table
                st.subheader("Results")
                
                data = []
                for r in results:
                    tech = r.get("technical_metrics", {})
                    data.append({
                        'Symbol': r['symbol'],
                        'Score': r.get('overall_score', 0),
                        'Rec': r.get('recommendation', '').upper(),
                        'Price': f"₹{tech.get('current_price', 0):,.0f}",
                        'Tech': r.get('scores', {}).get('technical_score', 0),
                        'Fund': r.get('scores', {}).get('fundamental_score', 0),
                    })
                
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                
                # Bar chart
                fig = px.bar(
                    x=[r['symbol'] for r in results],
                    y=[r.get('overall_score', 0) for r in results],
                    color=[r.get('recommendation', 'hold') for r in results],
                    title="Overall Scores",
                    labels={'x': 'Symbol', 'y': 'Score'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Individual items
        for item in items:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                name = f" - {item['company_name']}" if item['company_name'] else ""
                st.write(f"**{item['symbol']}**{name}")
            
            with col2:
                try:
                    conn = sqlite3.connect(DATABASE["path"])
                    row = conn.execute("""
                        SELECT overall_score, recommendation FROM analysis_results
                        WHERE symbol=? ORDER BY analysis_timestamp DESC LIMIT 1
                    """, (item['symbol'],)).fetchone()
                    conn.close()
                    
                    if row:
                        st.write(f"{row[0]:.0f}/100 - {row[1].upper()}")
                    else:
                        st.caption("Not analyzed")
                except:
                    st.caption("Not analyzed")
            
            with col3:
                if st.button("✕", key=f"rm_{item['symbol']}", help="Remove"):
                    pipeline.remove_from_watchlist(item['symbol'])
                    st.rerun()
    else:
        st.info("Watchlist is empty. Add symbols above.")
    
    st.caption(DISCLAIMER)


# ============ HISTORY ============
elif page == "📜 History":
    st.title("📜 Analysis History")
    
    try:
        conn = sqlite3.connect(DATABASE["path"])
        
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM analysis_results ORDER BY symbol"
        ).fetchall()]
        
        selected = st.selectbox("Filter", ["All"] + symbols)
        limit = st.selectbox("Limit", [10, 25, 50, 100], index=1)
        
        query = """SELECT symbol, overall_score, technical_score, fundamental_score,
                   recommendation, current_price, analysis_timestamp
                   FROM analysis_results"""
        params = []
        
        if selected != "All":
            query += " WHERE symbol=?"
            params.append(selected)
        
        query += " ORDER BY analysis_timestamp DESC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        if rows:
            df = pd.DataFrame(rows, columns=[
                'Symbol', 'Overall', 'Tech', 'Fund', 'Rec', 'Price (₹)', 'Date'
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Trend chart for specific symbol
            if selected != "All" and len(rows) > 1:
                conn = sqlite3.connect(DATABASE["path"])
                trend = conn.execute("""
                    SELECT analysis_timestamp, overall_score, technical_score, fundamental_score
                    FROM analysis_results WHERE symbol=? ORDER BY analysis_timestamp
                """, (selected,)).fetchall()
                conn.close()
                
                if len(trend) > 1:
                    trend_df = pd.DataFrame(trend, columns=['Date', 'Overall', 'Tech', 'Fund'])
                    trend_df['Date'] = pd.to_datetime(trend_df['Date'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df['Date'], y=trend_df['Overall'],
                        mode='lines+markers', name='Overall'
                    ))
                    fig.add_trace(go.Scatter(
                        x=trend_df['Date'], y=trend_df['Tech'],
                        mode='lines', name='Technical', line=dict(dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=trend_df['Date'], y=trend_df['Fund'],
                        mode='lines', name='Fundamental', line=dict(dash='dot')
                    ))
                    
                    fig.update_layout(
                        title=f"Score Trend: {selected}",
                        yaxis=dict(range=[0, 100]),
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history found.")
            
    except Exception as e:
        st.info("No history available.")


# ============ MARKET ============
elif page == "🌐 Market":
    st.title("🌐 Market Context")
    
    try:
        from v1.analysis.market_context import MarketContextAnalyzer
        
        analyzer = MarketContextAnalyzer()
        context = analyzer.get_market_context()
        score = analyzer.get_macro_score(context)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Regime", context.market_regime.upper())
        with col2:
            st.metric("Nifty", context.nifty_trend.value.upper())
        with col3:
            st.metric("VIX", context.vix_level or "N/A")
        with col4:
            st.metric("Macro", f"{score:.0f}/100")
        
        # Interpretation
        regime = context.market_regime
        if regime == "bull":
            st.success("🐂 Bull market. Good environment for equity investments.")
        elif regime == "bear":
            st.error("🐻 Bear market. Consider defensive positioning.")
        elif regime in ("cautiously_bullish",):
            st.info("📈 Cautiously bullish. Selective stock picking recommended.")
        elif regime in ("cautiously_bearish",):
            st.warning("📉 Cautiously bearish. Look for value opportunities.")
        else:
            st.info("➡️ Neutral market. Focus on fundamentals.")
        
        # Nifty chart
        st.subheader("Nifty 50 (6 Months)")
        try:
            import yfinance as yf
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="6mo")
            
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist['Close'],
                    mode='lines', name='Nifty 50',
                    line=dict(color='blue', width=2)
                ))
                
                if len(hist) > 50:
                    ma = hist['Close'].rolling(50).mean()
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=ma,
                        mode='lines', name='50-day MA',
                        line=dict(dash='dot', color='orange')
                    ))
                
                fig.update_layout(height=350, yaxis_title="Points")
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Chart unavailable.")
        
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.caption(DISCLAIMER)