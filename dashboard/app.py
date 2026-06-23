"""
Trading Intelligence Dashboard

Multi-page Streamlit dashboard covering:
  - Overview / KPIs
  - Equity curve + drawdown
  - Win rate by regime
  - Win rate by session
  - Win rate by weekday
  - Model performance + feature importance
  - Blocked trades analysis
  - Flipped trades analysis
  - Prediction confidence distribution
  - Live recent predictions

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Trading Intelligence",
    page_icon  = "🤖",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Config ────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
REFRESH_SECONDS = 30

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #7c3aed;
    }
    .positive { color: #22c55e; }
    .negative { color: #ef4444; }
    .neutral  { color: #94a3b8; }
    .stMetric label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_SECONDS)
def fetch(endpoint: str) -> Dict:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def fetch_equity():    return fetch("/analytics/equity")
def fetch_regime():    return fetch("/analytics/regime")
def fetch_session():   return fetch("/analytics/session")
def fetch_weekday():   return fetch("/analytics/weekday")
def fetch_models():    return fetch("/analytics/models")
def fetch_blocked():   return fetch("/analytics/blocked")
def fetch_preds():     return fetch("/analytics/predictions/recent?limit=100")
def fetch_health():    return fetch("/health")


# ── Colour helpers ────────────────────────────────────────────

REGIME_COLORS = {
    "strong_bull_trend":  "#22c55e",
    "weak_bull_trend":    "#86efac",
    "strong_bear_trend":  "#ef4444",
    "weak_bear_trend":    "#fca5a5",
    "sideways_range":     "#f59e0b",
    "high_volatility":    "#8b5cf6",
    "low_volatility":     "#06b6d4",
    "news_volatility":    "#f97316",
    "liquidity_grab":     "#ec4899",
}

SESSION_COLORS = {
    "asian":               "#06b6d4",
    "london":              "#3b82f6",
    "new_york":            "#22c55e",
    "overlap_london_ny":   "#7c3aed",
    "overlap_asian_london":"#f59e0b",
    "off_hours":           "#64748b",
}


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/7c3aed/ffffff?text=TradingAI", width=200)
    st.title("Navigation")

    page = st.radio("", [
        "📊 Overview",
        "📈 Equity Curve",
        "🗂️ Regime Analysis",
        "🕐 Session Analysis",
        "📅 Weekday Analysis",
        "🤖 Model Performance",
        "🚫 Blocked Trades",
        "🔄 Flipped Trades",
        "🔮 Recent Predictions",
    ])

    st.divider()

    health = fetch_health()
    if "error" not in health:
        st.markdown("**System Status**")
        st.success("🟢 API Online")
        col1, col2 = st.columns(2)
        col1.metric("Memory", f"{health.get('memory_size',0):,}")
        col2.metric("Uptime", f"{health.get('uptime_seconds',0)/3600:.1f}h")

        trader_ok = health.get("trader_trained", False)
        risk_ok   = health.get("risk_trained", False)
        st.markdown(
            f"Trader AI: {'✅' if trader_ok else '⚠️ Not trained'}  \n"
            f"Risk AI: {'✅' if risk_ok else '⚠️ Not trained'}"
        )
    else:
        st.error("🔴 API Offline")

    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Auto-refresh every {REFRESH_SECONDS}s")


# ══════════════════════════════════════════════════════════════
# HELPER: make_perf_df
# ══════════════════════════════════════════════════════════════

def make_perf_df(data: Dict) -> pd.DataFrame:
    rows = []
    for key, stats in data.items():
        if isinstance(stats, dict):
            rows.append({
                "Name":        key.replace("_", " ").title(),
                "Trades":      stats.get("total_trades", 0),
                "Wins":        stats.get("wins", 0),
                "Losses":      stats.get("losses", 0),
                "Win Rate":    stats.get("win_rate", 0),
                "Total Pips":  stats.get("total_pips", 0),
                "Avg Pips":    stats.get("avg_pips", 0),
            })
    return pd.DataFrame(rows).sort_values("Win Rate", ascending=False)


# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 Trading Intelligence — Overview")
    st.divider()

    # KPI row
    eq   = fetch_equity()
    curve = eq.get("equity_curve", [])
    dds   = eq.get("drawdowns", [])

    total_pips = curve[-1]["equity"] if curve else 0
    max_dd     = max(dds) if dds else 0
    n_trades   = len(curve)

    # Compute overall win rate from equity curve direction
    pnls      = [c.get("equity", 0) - (curve[i-1].get("equity", 0) if i > 0 else 0)
                 for i, c in enumerate(curve)]
    wins      = sum(1 for p in pnls if p > 0)
    win_rate  = wins / n_trades if n_trades else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Pips",   f"{total_pips:+.1f}")
    c2.metric("Total Trades", n_trades)
    c3.metric("Win Rate",     f"{win_rate:.1%}")
    c4.metric("Max Drawdown", f"{max_dd:.1f} pips", delta_color="inverse")
    c5.metric("Memory Size",  f"{health.get('memory_size',0):,}")

    st.divider()

    # Mini equity curve
    if curve:
        df_eq = pd.DataFrame(curve)
        fig   = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df_eq))), y=df_eq["equity"],
            mode="lines", name="Equity",
            line=dict(color="#7c3aed", width=2),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
        ))
        fig.update_layout(
            title="Equity Curve", height=300,
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Regime + session side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win Rate by Regime")
        regime_data = fetch_regime()
        if regime_data and "error" not in regime_data:
            df_r = make_perf_df(regime_data)
            if not df_r.empty:
                fig = px.bar(
                    df_r, x="Win Rate", y="Name",
                    orientation="h", color="Win Rate",
                    color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
                    range_color=[0.3, 0.8],
                )
                fig.update_layout(height=350, paper_bgcolor="#0f172a",
                                  plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Win Rate by Session")
        sess_data = fetch_session()
        if sess_data and "error" not in sess_data:
            df_s = make_perf_df(sess_data)
            if not df_s.empty:
                fig = px.bar(
                    df_s, x="Name", y="Win Rate",
                    color="Win Rate",
                    color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
                    range_color=[0.3, 0.8],
                )
                fig.update_layout(height=350, paper_bgcolor="#0f172a",
                                  plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: EQUITY CURVE
# ══════════════════════════════════════════════════════════════

elif page == "📈 Equity Curve":
    st.title("📈 Equity Curve & Drawdown")
    eq   = fetch_equity()
    curve = eq.get("equity_curve", [])
    dds   = eq.get("drawdowns", [])

    if not curve:
        st.info("No trade history yet.")
    else:
        df_eq = pd.DataFrame(curve)
        x     = list(range(len(df_eq)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=df_eq["equity"],
            name="Equity (pips)", mode="lines",
            line=dict(color="#7c3aed", width=2),
        ))
        fig.update_layout(
            title="Cumulative Equity (Pips)", height=400,
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            yaxis=dict(gridcolor="#1e293b"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x, y=[-d for d in dds],
            name="Drawdown (pips)", mode="lines",
            line=dict(color="#ef4444", width=1.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        ))
        fig2.update_layout(
            title="Drawdown (Pips)", height=250,
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Equity",  f"{curve[-1]['equity']:+.1f} pips")
        col2.metric("Max Drawdown",  f"{max(dds):.1f} pips", delta_color="inverse")
        col3.metric("Best Point",    f"{max(c['equity'] for c in curve):+.1f}")
        col4.metric("Worst Point",   f"{min(c['equity'] for c in curve):+.1f}")


# ══════════════════════════════════════════════════════════════
# PAGE: REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════

elif page == "🗂️ Regime Analysis":
    st.title("🗂️ Win Rate by Market Regime")
    regime_data = fetch_regime()

    if "error" in regime_data:
        st.error(f"API error: {regime_data['error']}")
    elif not regime_data:
        st.info("No regime data yet.")
    else:
        df = make_perf_df(regime_data)
        st.dataframe(df.style.background_gradient(subset=["Win Rate"], cmap="RdYlGn"), use_container_width=True)

        fig = px.scatter(
            df, x="Avg Pips", y="Win Rate",
            size="Trades", color="Win Rate",
            color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
            range_color=[0.3, 0.8],
            text="Name", hover_data=["Trades","Total Pips"],
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            height=450, paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a", font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: SESSION ANALYSIS
# ══════════════════════════════════════════════════════════════

elif page == "🕐 Session Analysis":
    st.title("🕐 Win Rate by Trading Session")
    sess_data = fetch_session()

    if "error" in sess_data:
        st.error(f"API error: {sess_data['error']}")
    elif not sess_data:
        st.info("No session data yet.")
    else:
        df = make_perf_df(sess_data)
        st.dataframe(df.style.background_gradient(subset=["Win Rate"], cmap="RdYlGn"), use_container_width=True)

        fig = px.bar(df, x="Name", y=["Wins","Losses"],
                     barmode="group",
                     color_discrete_map={"Wins":"#22c55e","Losses":"#ef4444"})
        fig.update_layout(height=350, paper_bgcolor="#0f172a",
                          plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: WEEKDAY ANALYSIS
# ══════════════════════════════════════════════════════════════

elif page == "📅 Weekday Analysis":
    st.title("📅 Win Rate by Day of Week")
    week_data = fetch_weekday()

    if "error" in week_data:
        st.error(f"API error: {week_data['error']}")
    elif not week_data:
        st.info("No weekday data yet.")
    else:
        df = make_perf_df(week_data)
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        df["sort"] = df["Name"].apply(lambda x: order.index(x) if x in order else 99)
        df = df.sort_values("sort")

        fig = px.line(df, x="Name", y="Win Rate", markers=True,
                      color_discrete_sequence=["#7c3aed"])
        fig.add_hline(y=0.5, line_dash="dash", line_color="#64748b")
        fig.update_layout(height=350, paper_bgcolor="#0f172a",
                          plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.drop(columns=["sort"]).style.background_gradient(
            subset=["Win Rate"], cmap="RdYlGn"), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    st.title("🤖 AI Model Performance")
    model_data = fetch_models()

    if "error" in model_data:
        st.error(f"API error: {model_data['error']}")
    else:
        col1, col2 = st.columns(2)

        for col, key, label in [
            (col1, "trader_ai",   "Trader AI"),
            (col2, "risk_manager","Risk Manager AI"),
        ]:
            with col:
                st.subheader(label)
                m = model_data.get(key, {})
                if m.get("is_trained"):
                    metrics = m.get("metrics", {})
                    st.metric("Algorithm", m.get("algorithm","?"))
                    st.metric("ROC-AUC",   f"{metrics.get('roc_auc',0):.4f}")
                    st.metric("WF Mean",   f"{metrics.get('wf_mean',0):.4f}")
                    st.metric("WF Std",    f"{metrics.get('wf_std',0):.4f}")

                    fi = m.get("feature_importance", {})
                    if fi:
                        df_fi = pd.DataFrame(
                            list(fi.items())[:20],
                            columns=["Feature","Importance"]
                        )
                        fig = px.bar(df_fi, x="Importance", y="Feature",
                                     orientation="h",
                                     color="Importance",
                                     color_continuous_scale="Purples")
                        fig.update_layout(
                            height=500, yaxis={"categoryorder":"total ascending"},
                            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                            font=dict(color="#94a3b8"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Model not trained yet")


# ══════════════════════════════════════════════════════════════
# PAGE: BLOCKED TRADES
# ══════════════════════════════════════════════════════════════

elif page == "🚫 Blocked Trades":
    st.title("🚫 Blocked Trades Analysis")
    blocked_data = fetch_blocked()
    trades = blocked_data.get("blocked_trades", [])

    st.metric("Total Blocked", len(trades))

    if trades:
        # Flatten block reasons
        reasons = {}
        for t in trades:
            raw = t.get("risk_block_reasons") or "[]"
            try:
                rs = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                rs = []
            for r in rs:
                base = r.split(":")[0]
                reasons[base] = reasons.get(base, 0) + 1

        df_r = pd.DataFrame(
            sorted(reasons.items(), key=lambda x: x[1], reverse=True),
            columns=["Reason","Count"],
        )
        fig = px.bar(df_r, x="Count", y="Reason", orientation="h",
                     color="Count", color_continuous_scale="Reds")
        fig.update_layout(height=350, paper_bgcolor="#0f172a",
                          plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)

        # Confidence distribution of blocked trades
        confs = [t.get("trader_confidence",0) or 0 for t in trades]
        fig2 = px.histogram(x=confs, nbins=20, color_discrete_sequence=["#ef4444"])
        fig2.update_layout(title="Trader Confidence Distribution (Blocked)",
                           xaxis_title="Confidence", height=300,
                           paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                           font=dict(color="#94a3b8"))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: FLIPPED TRADES
# ══════════════════════════════════════════════════════════════

elif page == "🔄 Flipped Trades":
    st.title("🔄 Flipped Trades Analysis")
    preds = fetch_preds().get("predictions", [])
    flipped = [p for p in preds if p.get("is_flip")]

    st.metric("Recent Flipped Trades", len(flipped))
    if flipped:
        df = pd.DataFrame(flipped)[[
            "predicted_at","ea_signal","trader_direction",
            "final_decision","trader_confidence","risk_quality_score"
        ]]
        df.columns = ["Time","EA Signal","AI Direction","Final","Confidence","Quality"]
        st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: RECENT PREDICTIONS
# ══════════════════════════════════════════════════════════════

elif page == "🔮 Recent Predictions":
    st.title("🔮 Recent Predictions")
    preds = fetch_preds().get("predictions", [])

    if not preds:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(preds)

        # Confidence scatter
        if "trader_confidence" in df.columns and "risk_quality_score" in df.columns:
            df["color"] = df["final_decision"].apply(
                lambda x: "#22c55e" if "ALLOW" in str(x)
                else "#ef4444" if "BLOCK" in str(x)
                else "#f59e0b"
            )
            fig = px.scatter(
                df, x="trader_confidence", y="risk_quality_score",
                color="final_decision", hover_data=["ea_signal","predicted_at"],
                title="Trader Confidence vs Risk Quality",
            )
            fig.update_layout(height=400, paper_bgcolor="#0f172a",
                              plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
            st.plotly_chart(fig, use_container_width=True)

        # Decision distribution
        if "final_decision" in df.columns:
            dist = df["final_decision"].value_counts().reset_index()
            dist.columns = ["Decision","Count"]
            fig2 = px.pie(dist, values="Count", names="Decision",
                          color_discrete_sequence=px.colors.qualitative.Vivid)
            fig2.update_layout(height=350, paper_bgcolor="#0f172a",
                               font=dict(color="#94a3b8"))
            st.plotly_chart(fig2, use_container_width=True)

        cols_show = [c for c in [
            "predicted_at","ea_signal","trader_direction",
            "final_decision","trader_confidence","risk_quality_score","inference_ms"
        ] if c in df.columns]
        st.dataframe(df[cols_show].head(50), use_container_width=True)

# ── Auto-refresh ──────────────────────────────────────────────
time.sleep(REFRESH_SECONDS)
st.rerun()
