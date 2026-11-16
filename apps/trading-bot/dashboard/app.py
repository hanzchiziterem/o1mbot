"""Interactive Streamlit & Plotly Trading Dashboard.

Displays KPI cards, cumulative R-multiple curve, pattern win/loss breakdown,
candlestick chart with Volume Profile overlay, and filterable trade metadata table.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtester import HistoricalBacktester
from src.database import DatabaseManager
from src.mt5_connector import MT5Connector
from src.strategy import TopDownVPStrategy


def load_data(db_path: str = "data/trades_metadata.db") -> pd.DataFrame:
    db = DatabaseManager(db_path=db_path)
    df = db.get_trades_dataframe()
    return df


def render_kpis(df: pd.DataFrame) -> None:
    closed = df[df["outcome"].isin(["WIN", "LOSS", "BE"])]
    total = len(closed)

    if total == 0:
        st.info("No closed trades recorded yet.")
        cols = st.columns(5)
        cols[0].metric("Total Trades", 0)
        cols[1].metric("Win Rate %", "0.0%")
        cols[2].metric("Loss Rate %", "0.0%")
        cols[3].metric("Avg Planned RR", "0.0x")
        cols[4].metric("Avg Realized RR", "0.0x")
        return

    wins = closed[closed["outcome"] == "WIN"]
    losses = closed[closed["outcome"] == "LOSS"]

    win_rate = (len(wins) / total) * 100.0
    loss_rate = (len(losses) / total) * 100.0
    avg_planned_rr = closed["planned_rr"].mean()
    avg_realized_rr = closed["realized_rr"].dropna().mean()
    cum_r = closed["realized_rr"].dropna().sum()

    cols = st.columns(6)
    cols[0].metric("Total Trades", total)
    cols[1].metric("Win Rate", f"{win_rate:.1f}%")
    cols[2].metric("Loss Rate", f"{loss_rate:.1f}%")
    cols[3].metric("Avg Planned RR", f"{avg_planned_rr:.2f}R")
    cols[4].metric("Avg Realized RR", f"{avg_realized_rr:.2f}R")
    cols[5].metric("Cumulative R", f"{cum_r:+.2f}R")


def render_equity_curve(df: pd.DataFrame) -> None:
    closed = df[df["outcome"].isin(["WIN", "LOSS", "BE"])].copy()
    if closed.empty:
        return

    closed["cum_r"] = closed["realized_rr"].fillna(0.0).cumsum()
    closed["trade_num"] = range(1, len(closed) + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=closed["trade_num"],
            y=closed["cum_r"],
            mode="lines+markers",
            name="Cumulative R",
            line=dict(color="#00FFAA", width=2.5),
            marker=dict(size=6, color="#00FFAA"),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 170, 0.1)",
        )
    )

    fig.update_layout(
        title="Cumulative R-Multiple Growth",
        xaxis_title="Trade Count",
        yaxis_title="Total R-Multiple",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pattern_distribution(df: pd.DataFrame) -> None:
    closed = df[df["outcome"].isin(["WIN", "LOSS", "BE"])].copy()
    if closed.empty:
        return

    grouped = closed.groupby(["entry_pattern", "outcome"]).size().reset_index(name="count")

    fig = go.Figure()
    for outcome, color in [("WIN", "#00E676"), ("LOSS", "#FF5252"), ("BE", "#FFD700")]:
        subset = grouped[grouped["outcome"] == outcome]
        fig.add_trace(
            go.Bar(
                x=subset["entry_pattern"],
                y=subset["count"],
                name=outcome,
                marker_color=color,
            )
        )

    fig.update_layout(
        barmode="group",
        title="Win / Loss Distribution by Entry Pattern (QMR vs AMD)",
        xaxis_title="Entry Pattern",
        yaxis_title="Count",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_candlestick_with_vp(symbol: str, htf: str = "H4") -> None:
    st.subheader(f"Price Action & Fixed Range Volume Profile: {symbol} ({htf})")
    connector = MT5Connector(mock_mode=True)
    df = connector.get_rates(symbol, htf, count=100)

    # Calculate Volume Profile
    strategy = TopDownVPStrategy()
    vp = strategy.vp_analyzer.compute_profile(df, 0, len(df) - 1)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.02,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # POC line
    fig.add_hline(
        y=vp.poc,
        line_dash="dash",
        line_color="#FF3366",
        annotation_text=f"POC: {vp.poc:.2f}",
        annotation_position="bottom right",
        row=1,
        col=1,
    )
    # VAH / VAL lines
    fig.add_hline(
        y=vp.vah,
        line_dash="dot",
        line_color="#00D2FF",
        annotation_text=f"VAH: {vp.vah:.2f}",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=vp.val,
        line_dash="dot",
        line_color="#00D2FF",
        annotation_text=f"VAL: {vp.val:.2f}",
        row=1,
        col=1,
    )

    # Horizontal Volume Profile Histogram
    bin_mids = [b.price_mid for b in vp.bins]
    bin_vols = [b.volume for b in vp.bins]
    colors = [
        "#FF3366" if abs(m - vp.poc) < (vp.vah - vp.val) * 0.05 else "#336699" for m in bin_mids
    ]

    fig.add_trace(
        go.Bar(
            y=bin_mids,
            x=bin_vols,
            orientation="h",
            marker_color=colors,
            name="Volume Profile",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="o1mbot | Exness MT5 Quantitative Trading Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("⚡ Exness MT5 Algorithmic Trading Metadata Dashboard")
    st.caption(
        "Top-Down Macro Bias • Fixed Range Volume Profile • QMR & AMD Precision Entries • Min 1:5 RR"
    )

    db_path = "data/trades_metadata.db"
    df = load_data(db_path)

    # Sidebar Filter Controls
    st.sidebar.header("Filter & Controls")

    modes = (
        ["ALL"] + sorted(df["mode"].unique().tolist())
        if not df.empty
        else ["ALL", "BACKTEST", "LIVE_PAPER"]
    )
    selected_mode = st.sidebar.selectbox("Execution Mode", modes, index=0)

    symbols = (
        ["All"] + sorted(df["symbol"].unique().tolist())
        if not df.empty
        else ["All", "XAUUSDm", "BTCUSDm", "EURUSDm", "GBPUSDm"]
    )
    selected_symbol = st.sidebar.selectbox("Symbol", symbols, index=0)

    patterns = ["All", "QMR", "AMD"]
    selected_pattern = st.sidebar.selectbox("Entry Pattern", patterns, index=0)

    # Filter dataframe
    filtered_df = df.copy() if not df.empty else pd.DataFrame()
    if not filtered_df.empty:
        if selected_mode != "ALL":
            filtered_df = filtered_df[filtered_df["mode"] == selected_mode]
        if selected_symbol != "All":
            filtered_df = filtered_df[filtered_df["symbol"] == selected_symbol]
        if selected_pattern != "All":
            filtered_df = filtered_df[filtered_df["entry_pattern"] == selected_pattern]

    tabs = st.tabs(
        [
            "📊 Performance KPIs & Analytics",
            "🕯️ Live Chart & Volume Profile",
            "📜 Trades Metadata Table",
            "🚀 Run Historical Backtest",
        ]
    )

    with tabs[0]:
        st.subheader("Strategy Performance Metrics")
        render_kpis(filtered_df)

        c1, c2 = st.columns(2)
        with c1:
            render_equity_curve(filtered_df)
        with c2:
            render_pattern_distribution(filtered_df)

    with tabs[1]:
        sym = selected_symbol if selected_symbol != "All" else "XAUUSDm"
        render_candlestick_with_vp(sym)

    with tabs[2]:
        st.subheader("Interactive Trades Metadata")
        if filtered_df.empty:
            st.info("No trade records match current filters.")
        else:
            st.dataframe(filtered_df, use_container_width=True)
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv,
                file_name="o1mbot_trades_metadata.csv",
                mime="text/csv",
            )

    with tabs[3]:
        st.subheader("Execute In-Dashboard Historical Simulation")
        b_sym = st.selectbox(
            "Backtest Symbol", ["XAUUSDm", "BTCUSDm", "EURUSDm", "GBPUSDm"], key="bt_sym"
        )
        b_balance = st.number_input("Initial Balance ($)", value=10000.0, step=1000.0)

        if st.button("Run Backtest Now", type="primary"):
            with st.spinner(f"Simulating top-down strategy for {b_sym}..."):
                connector = MT5Connector(mock_mode=True)
                rates: Dict[str, Any] = {
                    "MN1": connector.get_rates(b_sym, "MN1", count=100),
                    "W1": connector.get_rates(b_sym, "W1", count=100),
                    "D1": connector.get_rates(b_sym, "D1", count=200),
                    "H4": connector.get_rates(b_sym, "H4", count=200),
                    "H1": connector.get_rates(b_sym, "H1", count=200),
                    "M15": connector.get_rates(b_sym, "M15", count=300),
                    "M5": connector.get_rates(b_sym, "M5", count=300),
                }
                strategy = TopDownVPStrategy()
                backtester = HistoricalBacktester(strategy=strategy, initial_balance=b_balance)
                result = backtester.run(b_sym, rates, log_to_db=True)

                st.success(
                    f"Backtest completed! Generated {result.total_trades} trades. Final Balance: ${result.final_balance:,.2f}"
                )
                st.rerun()


if __name__ == "__main__":
    main()
