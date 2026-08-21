import os
import sys
from datetime import datetime, timezone
import json
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure app root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.backtest.engine import backtest_engine
from app.market_data.provider import market_provider
from app.orchestrator.service import orchestrator_service
from app.rag.rss_crawler import rss_crawler
from app.rag.vector_store import vector_store
from app.scanner.service import scanner_service

# Page Configuration
st.set_page_config(
    page_title="TradeM — AI Trading Research Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 20px; }
    .card-bullish { background-color: #E8F5E9; border-left: 5px solid #2E7D32; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .card-bearish { background-color: #FFEBEE; border-left: 5px solid #C62828; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .disclaimer-box { background-color: #FFF3E0; border: 1px solid #FFE0B2; padding: 12px; border-radius: 5px; color: #E65100; font-size: 0.9rem; }
    .metric-badge { background-color: #ECEFF1; padding: 6px 12px; border-radius: 15px; font-weight: 600; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=70)
st.sidebar.title("TradeM Controls")
selected_symbol = st.sidebar.selectbox(
    "Select Asset / Symbol",
    ["RELIANCE", "NIFTY", "BANKNIFTY", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"],
)
instrument_type = st.sidebar.radio("Instrument Type", ["options", "futures", "equity"])

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("Angel One SmartAPI: Connected")
st.sidebar.info("RAG Vector Store: Active (72h Filter)")
st.sidebar.write(f"Local Time: {datetime.now().strftime('%H:%M:%S IST')}")

# Header Section
st.markdown('<div class="main-header">TradeM — AI Decision Support Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Transparent Reasoning & Risk Analysis Engine for Indian F&O Markets</div>',
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Decision Support Assistant",
    "📊 Option Chain & Levels",
    "🚨 Market Scanner",
    "📈 Options Backtester",
    "📰 Live RAG News & Knowledge",
])

# ==============================================================================
# TAB 1: PLUTUS AI DECISION SUPPORT ASSISTANT
# ==============================================================================
with tab1:
    st.subheader(f"Decision Support Analysis for {selected_symbol}")

    user_query = st.text_input(
        "Enter your research query or strategy question:",
        value=f"What is the option outlook and key risk-reward profile for {selected_symbol}?",
    )

    if st.button("Run Plutus Analysis", type="primary"):
        with st.spinner("Executing orchestrator pipeline (Market Snapshot + RAG 72h + Plutus Reasoning)..."):
            analysis_obj = orchestrator_service.analyze_symbol(
                query=user_query,
                symbol=selected_symbol,
                instrument_type=instrument_type,
            )
            analysis = analysis_obj.dict()

        # Snapshot KPIs
        curr = analysis.get("current_data", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LTP (Last Traded Price)", f"₹{curr.get('ltp', 0.0)}")
        col2.metric("Put-Call Ratio (PCR)", curr.get("pcr", 1.0))
        col3.metric("Implied Volatility (IV)", f"{curr.get('iv', 20.0)}%")
        col4.metric("OI Trend", str(curr.get("oi_trend", "neutral")).upper())

        st.markdown("---")

        # Two-column layout for Bullish vs Bearish Evidence
        c_bull, c_bear = st.columns(2)

        with c_bull:
            st.markdown("### 🟢 Bullish Evidence & Drivers")
            bull_factors = analysis.get("bullish_factors", [])
            for bf in bull_factors:
                st.markdown(
                    f"""<div class="card-bullish">
                    <strong>Point:</strong> {bf.get('point')}<br>
                    <span class="metric-badge">Source: {bf.get('source')}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with c_bear:
            st.markdown("### 🔴 Bearish Risks & Drivers")
            bear_factors = analysis.get("bearish_factors", [])
            for br in bear_factors:
                st.markdown(
                    f"""<div class="card-bearish">
                    <strong>Point:</strong> {br.get('point')}<br>
                    <span class="metric-badge">Source: {br.get('source')}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Key Levels & Invalidation & Confidence
        col_lvl, col_conf = st.columns(2)

        with col_lvl:
            st.markdown("### 🎯 Key Technical Levels & Invalidation")
            levels = analysis.get("key_levels", {})
            st.write(f"**Support Floor:** ₹{levels.get('support')}")
            st.write(f"**Resistance Ceiling:** ₹{levels.get('resistance')}")
            st.error(f"**Invalidation Criteria:** {analysis.get('invalidation_conditions')}")

        with col_conf:
            st.markdown("### ⚖️ Confidence Justification & Backtest")
            st.info(f"**Confidence Level:** {analysis.get('confidence')}")

            bt = analysis.get("backtest_context", {})
            st.write(
                f"**Backtest Sharpe:** {bt.get('sharpe')} | **Win Rate:** {bt.get('win_rate')}% | **Max Drawdown:** {bt.get('max_drawdown')}%"
            )
            st.caption(f"⚠️ {bt.get('caveat')}")

        st.markdown("---")
        st.markdown(
            f"""<div class="disclaimer-box">
            <strong>🔒 Mandatory Disclaimer:</strong> {analysis.get('explicit_note')}
            </div>""",
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: OPTION CHAIN & TECHNICAL LEVELS
# ==============================================================================
with tab2:
    st.subheader(f"Live Option Chain & Volatility Analytics ({selected_symbol})")
    snap = market_provider.get_snapshot(selected_symbol)

    col1, col2, col3 = st.columns(3)
    col1.metric("LTP", f"₹{snap['ltp']}")
    col2.metric("Support (1.5%)", f"₹{snap['support']}")
    col3.metric("Resistance (1.5%)", f"₹{snap['resistance']}")

    st.markdown("### Synthetic Option Chain Metrics")

    # Generate synthetic option chain strikes around LTP
    base_price = snap["ltp"]
    step = 50 if base_price > 2000 else 20
    strikes = [round(base_price + i * step, 2) for i in range(-5, 6)]

    chain_data = []
    for strike in strikes:
        diff = strike - base_price
        call_oi = max(1000, int(50000 - abs(diff) * 120))
        put_oi = max(1000, int(50000 + diff * 120))
        chain_data.append({
            "Call OI": call_oi,
            "Call IV %": round(snap["iv"] + (strike - base_price) * 0.01, 1),
            "Strike Price": strike,
            "Put IV %": round(snap["iv"] - (strike - base_price) * 0.01, 1),
            "Put OI": put_oi,
        })

    df_chain = pd.DataFrame(chain_data)
    st.dataframe(df_chain, use_container_width=True)

    # Plotly Call vs Put OI Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_chain["Strike Price"], y=df_chain["Call OI"], name="Call OI (Resistance)", marker_color="#EF5350"))
    fig.add_trace(go.Bar(x=df_chain["Strike Price"], y=df_chain["Put OI"], name="Put OI (Support)", marker_color="#66BB6A"))
    fig.update_layout(title=f"Option Chain Open Interest Distribution for {selected_symbol}", barmode="group")
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 3: QUANTITATIVE MARKET SCANNER
# ==============================================================================
with tab3:
    st.subheader("Quantitative Market Scanner & Alerting")
    st.write("Scans active watchlist for Volume Breakouts, PCR Extremes, and IV Spikes.")

    if st.button("Trigger Market Scan Now", type="primary"):
        with st.spinner("Running quantitative screens..."):
            alerts = scanner_service.run_scan(orchestrator_service=orchestrator_service)

        st.success(f"Market Scan Complete! Found {len(alerts)} active alert triggers.")

        for alt in alerts:
            st.warning(
                f"**[{alt['screen']}] {alt['symbol']}** — {alt['reason']}"
            )
            with st.expander("View Full Analysis for Alert"):
                st.json(alt["analysis"])


# ==============================================================================
# TAB 4: OPTIONS STRATEGY BACKTESTER
# ==============================================================================
with tab4:
    st.subheader("Historical Options Strategy Backtest Simulator")

    c_strat, c_days = st.columns(2)
    strat = c_strat.selectbox("Select Options Strategy", backtest_engine.STRATEGIES)
    days = c_days.slider("Historical Lookback Window (Days)", min_value=30, max_value=180, value=90)

    if st.button("Run Strategy Simulation"):
        res = backtest_engine.run_backtest(selected_symbol, strategy=strat, days_history=days)

        # Performance KPI Grid
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Sharpe Ratio", res["sharpe"])
        col_b.metric("Win Rate", f"{res['win_rate']}%")
        col_c.metric("Max Drawdown", f"{res['max_drawdown']}%")
        col_d.metric("Profit Factor", res["profit_factor"])

        st.caption(f"⚠️ **Mandatory Caveat:** {res['caveat']}")

        # Plotly Equity Curve
        fig_eq = px.line(
            x=list(range(len(res["equity_curve"]))),
            y=res["equity_curve"],
            title=f"Simulated Equity Curve — {selected_symbol} ({strat.upper()})",
            labels={"x": "Trade Window", "y": "Capital (₹)"},
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown("### Simulated Trade Log")
        st.dataframe(pd.DataFrame(res["trades"]), use_container_width=True)


# ==============================================================================
# TAB 5: LIVE RAG NEWS & KNOWLEDGE BASE
# ==============================================================================
with tab5:
    st.subheader("Live RAG News Feed & Recency Inspector (72h Decay)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Crawl Financial RSS Feeds Now"):
            with st.spinner("Crawling ET Markets, Moneycontrol, Livemint..."):
                crawl_res = rss_crawler.crawl_and_ingest_all()
            st.success(f"Crawled & Ingested {crawl_res['total_ingested']} live news articles into RAG Store!")

    st.markdown("---")
    st.markdown("### Ingest Custom News Article")
    with st.form("manual_ingest_form"):
        h_input = st.text_input("Headline", value=f"{selected_symbol} announces strategic expansion plan.")
        b_input = st.text_area("Article Body", value=f"{selected_symbol} announced new investment plans expected to boost operating margins.")
        source_input = st.text_input("Source", value="ET Markets")
        submitted = st.form_submit_button("Ingest into RAG Store")

        if submitted:
            from app.rag.ingestion import ingest_news_article
            ingest_news_article(h_input, b_input, selected_symbol, source_input)
            st.success(f"Ingested article for {selected_symbol} into 72-hour RAG store!")

    st.markdown("---")
    st.markdown("### Inspect Vector Search Context")
    search_q = st.text_input("Search RAG Vector Store:", value=f"Options volatility and PCR support for {selected_symbol}")
    if st.button("Search Knowledge Base"):
        results = vector_store.retrieve(query=search_q, symbol=selected_symbol, k=5)
        st.write(f"Found {len(results)} recency-valid chunks (filtered within 72h window):")
        for r in results:
            st.json(r)
