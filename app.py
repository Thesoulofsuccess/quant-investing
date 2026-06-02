"""
Dual-Agreement Trading Terminal
Bloomberg-style investor-grade dashboard
Run: streamlit run app.py
"""

import io
import os
import sys
from contextlib import redirect_stdout
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Must be first ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAE Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Bloomberg Terminal palette
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#000000"
BG1       = "#0a0a0a"
BG2       = "#111111"
BG3       = "#1a1a1a"
BORDER    = "#2a2a2a"
BORDER2   = "#333333"
TEXT      = "#d4d4d4"
TEXT2     = "#888888"
TEXT3     = "#555555"
GREEN     = "#00ff88"
GREEN2    = "#00cc66"
RED       = "#ff3b3b"
RED2      = "#cc2222"
AMBER     = "#ffaa00"
BLUE      = "#4488ff"
CYAN      = "#00ccdd"
WHITE     = "#ffffff"

CSS = f"""
<style>
/* ── Reset & base ─────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

.stApp {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Courier New', 'Consolas', monospace !important;
}}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* ── Scrollbar ────────────────────────────── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG1}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 2px; }}

/* ── Top bar ──────────────────────────────── */
.topbar {{
    background: {BG1};
    border-bottom: 1px solid {BORDER};
    padding: 0 20px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
}}
.topbar-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.topbar-title {{
    font-size: 13px;
    font-weight: 700;
    color: {AMBER};
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}
.topbar-sub {{
    font-size: 10px;
    color: {TEXT3};
    letter-spacing: 0.08em;
}}
.topbar-right {{
    display: flex;
    align-items: center;
    gap: 20px;
    font-size: 10px;
    color: {TEXT3};
    letter-spacing: 0.05em;
}}
.dot-green {{ color: {GREEN}; }}
.dot-red   {{ color: {RED}; }}
.dot-amber {{ color: {AMBER}; }}

/* ── Tabs ─────────────────────────────────── */
[data-testid="stTabs"] {{
    background: {BG1};
    border-bottom: 1px solid {BORDER};
    padding: 0 20px;
}}
[data-testid="stTabs"] button {{
    background: transparent !important;
    color: {TEXT3} !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 16px !important;
    border-radius: 0 !important;
    transition: all 0.15s !important;
}}
[data-testid="stTabs"] button:hover {{
    color: {TEXT} !important;
    background: {BG2} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {AMBER} !important;
    border-bottom: 2px solid {AMBER} !important;
    background: transparent !important;
}}
[data-testid="stTabsContent"] {{
    background: {BG} !important;
    padding: 0 !important;
}}

/* ── Tab content padding ─────────────────── */
.tab-body {{ padding: 16px 20px; }}

/* ── Config strip ────────────────────────── */
.config-strip {{
    background: {BG1};
    border: 1px solid {BORDER};
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 14px;
}}
.config-label {{
    font-size: 9px;
    color: {TEXT3};
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 3px;
}}
.config-value {{
    font-size: 12px;
    color: {TEXT};
    font-weight: 600;
}}

/* ── KPI strip ───────────────────────────── */
.kpi-strip {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    border: 1px solid {BORDER};
    margin-bottom: 14px;
}}
.kpi-cell {{
    padding: 10px 14px;
    border-right: 1px solid {BORDER};
}}
.kpi-cell:last-child {{ border-right: none; }}
.kpi-lbl {{
    font-size: 9px;
    color: {TEXT3};
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.kpi-val {{
    font-size: 18px;
    font-weight: 700;
    color: {TEXT};
    white-space: nowrap;
}}
.kpi-val.g {{ color: {GREEN}; }}
.kpi-val.r {{ color: {RED}; }}
.kpi-val.a {{ color: {AMBER}; }}
.kpi-val.b {{ color: {BLUE}; }}
.kpi-sub {{
    font-size: 9px;
    color: {TEXT3};
    margin-top: 2px;
    white-space: nowrap;
}}

/* ── Section header ──────────────────────── */
.sec-hdr {{
    font-size: 9px;
    color: {TEXT3};
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 5px;
    margin-bottom: 10px;
    margin-top: 14px;
}}

/* ── Data table ──────────────────────────── */
.dt {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    font-family: 'Courier New', monospace;
}}
.dt th {{
    background: {BG2};
    color: {TEXT3};
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 6px 10px;
    text-align: right;
    border-bottom: 1px solid {BORDER};
    font-weight: 400;
    white-space: nowrap;
}}
.dt th:first-child {{ text-align: left; }}
.dt td {{
    padding: 5px 10px;
    text-align: right;
    border-bottom: 1px solid {BG2};
    color: {TEXT};
    white-space: nowrap;
}}
.dt td:first-child {{ text-align: left; color: {TEXT}; }}
.dt tr:hover td {{ background: {BG2}; }}
.dt tr:last-child td {{ border-bottom: none; }}
.pos {{ color: {GREEN} !important; }}
.neg {{ color: {RED} !important; }}
.neu {{ color: {TEXT2} !important; }}
.amb {{ color: {AMBER} !important; }}
.dash {{ color: {TEXT3} !important; }}

/* ── Signal badge ────────────────────────── */
.badge {{
    display: inline-block;
    padding: 1px 7px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    border-radius: 2px;
}}
.badge-buy  {{ background: #002a14; color: {GREEN}; border: 1px solid #004d24; }}
.badge-sell {{ background: #2a0000; color: {RED};   border: 1px solid #4d0000; }}
.badge-neu  {{ background: {BG2};   color: {TEXT3}; border: 1px solid {BORDER}; }}
.badge-skip {{ background: {BG2};   color: {TEXT3}; border: 1px solid {BORDER}; }}
.badge-conf {{ background: #1a1500; color: {AMBER}; border: 1px solid #3a3000; }}

/* ── Alert box ───────────────────────────── */
.alert-warn {{
    background: #1a1200;
    border: 1px solid #4d3800;
    border-left: 3px solid {AMBER};
    padding: 8px 12px;
    font-size: 11px;
    color: {AMBER};
    margin-bottom: 12px;
}}
.alert-err {{
    background: #1a0000;
    border: 1px solid #4d0000;
    border-left: 3px solid {RED};
    padding: 8px 12px;
    font-size: 11px;
    color: {RED};
    margin-bottom: 12px;
}}
.alert-info {{
    background: #001220;
    border: 1px solid #003366;
    border-left: 3px solid {BLUE};
    padding: 8px 12px;
    font-size: 11px;
    color: {BLUE};
    margin-bottom: 12px;
}}

/* ── Buttons ─────────────────────────────── */
div[data-testid="stButton"] > button {{
    background: {BG2} !important;
    color: {AMBER} !important;
    border: 1px solid {AMBER} !important;
    border-radius: 0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 8px 20px !important;
    transition: all 0.15s !important;
}}
div[data-testid="stButton"] > button:hover {{
    background: {AMBER} !important;
    color: {BG} !important;
}}

/* ── Form inputs ─────────────────────────── */
.stDateInput > div > div > input,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {{
    background: {BG2} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
}}
.stSlider [data-testid="stTickBar"] {{ color: {TEXT3}; font-size: 9px; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: {AMBER} !important;
    border-color: {AMBER} !important;
}}

/* ── Expander ────────────────────────────── */
[data-testid="stExpander"] {{
    background: {BG1} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 0 !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 10px !important;
    color: {TEXT3} !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}

/* ── Labels ──────────────────────────────── */
label, .stSlider label, .stDateInput label {{
    font-size: 9px !important;
    color: {TEXT3} !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
}}

/* ── Spinner ─────────────────────────────── */
.stSpinner > div > div {{
    border-top-color: {AMBER} !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(label, val, sub="", cls=""):
    return (f'<div class="kpi-cell">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val {cls}">{val}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>')


def badge(sig):
    sig = sig.upper()
    cls = {"BUY": "badge-buy", "SELL": "badge-sell",
           "NEUTRAL": "badge-neu", "SKIP": "badge-skip",
           "CONFIRMED": "badge-conf"}.get(sig, "badge-neu")
    return f'<span class="badge {cls}">{sig}</span>'


def sec(title):
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)


def fmt_inr(v, pct=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"₹{abs(v):,.0f}"
    sign = "+" if v >= 0 else "−"
    s = f"{sign}{s}"
    if pct is not None:
        psign = "+" if pct >= 0 else ""
        s += f" ({psign}{pct:.2f}%)"
    return s


def plotly_cfg():
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Courier New", color=TEXT2, size=10),
        margin=dict(l=4, r=4, t=28, b=4),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG2, bordercolor=BORDER2,
                        font=dict(family="Courier New", color=TEXT, size=10)),
    )


def axis_style(fmt="", prefix=""):
    return dict(
        showgrid=True, gridcolor=BG2, gridwidth=1,
        zeroline=True, zerolinecolor=BORDER, zerolinewidth=1,
        linecolor=BORDER, tickcolor=BORDER,
        tickfont=dict(family="Courier New", color=TEXT2, size=9),
        tickprefix=prefix, tickformat=fmt,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
from datetime import timezone
now_ist = datetime.now(tz=timezone.utc) + pd.Timedelta(hours=5, minutes=30)
market_open  = now_ist.hour >= 9 and (now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute <= 30))
market_badge = (f'<span class="dot-green">● MARKET OPEN</span>'
                if market_open else f'<span class="dot-red">● MARKET CLOSED</span>')

st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <span class="topbar-title">⚡ DAE TERMINAL</span>
    <span class="topbar-sub">DUAL-AGREEMENT ENGINE · NIFTY 100 · NSE EQUITIES · INTRADAY L/S</span>
  </div>
  <div class="topbar-right">
    {market_badge}
    <span>IST {now_ist.strftime('%H:%M:%S')}</span>
    <span class="dot-amber">{now_ist.strftime('%d %b %Y')}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_bt, tab_p1, tab_p2, tab_orb = st.tabs([
    "BACKTEST",
    "PHASE 1 · SIGNALS",
    "PHASE 2 · TRADES",
    "ORB · LIVE",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown('<div class="tab-body">', unsafe_allow_html=True)

    # Config row
    c1, c2, c3, c4, c5, c_run = st.columns([2, 2, 2, 1.5, 1.5, 1.5])
    with c1:
        bt_start = st.date_input("Start Date", value=date(2025, 1, 1),
                                 key="bt_start", label_visibility="visible")
    with c2:
        bt_end = st.date_input("End Date", value=date(2025, 6, 30),
                               key="bt_end", label_visibility="visible")
    with c3:
        bt_cap = st.number_input("Capital / Leg (₹)", min_value=50_000,
                                 max_value=50_00_000, value=10_00_000,
                                 step=50_000, key="bt_cap",
                                 format="%d")
    with c4:
        bt_topn = st.number_input("Top N (stocks/side)", min_value=1,
                                  max_value=10, value=2, step=1, key="bt_topn")
    with c5:
        st.markdown(f"""
        <div style='margin-top:8px;'>
          <div style='font-size:9px;color:{TEXT3};letter-spacing:.1em;text-transform:uppercase;'>Deployed</div>
          <div style='font-size:13px;color:{AMBER};font-weight:700;'>₹{bt_cap*2:,.0f}</div>
          <div style='font-size:9px;color:{TEXT3};'>Entry @ 9:16 AM  |  SL 1%</div>
        </div>
        """, unsafe_allow_html=True)
    with c_run:
        st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
        bt_run = st.button("▶ RUN", key="bt_run")
        st.markdown("</div>", unsafe_allow_html=True)

    if bt_run:
        if bt_start >= bt_end:
            st.markdown(f'<div class="alert-err">End date must be after start date.</div>',
                        unsafe_allow_html=True)
        else:
            from backtest import run_backtest
            with st.spinner("Running backtest…"):
                log_buf = io.StringIO()
                with redirect_stdout(log_buf):
                    stats_df, trades_df = run_backtest(
                        start=bt_start.strftime("%Y-%m-%d"),
                        end=bt_end.strftime("%Y-%m-%d"),
                        capital=bt_cap, top_n=bt_topn,
                    )
            st.session_state["bt"] = dict(
                stats=stats_df, trades=trades_df,
                cap=bt_cap, log=log_buf.getvalue(),
                start=bt_start.strftime("%Y-%m-%d"),
                end=bt_end.strftime("%Y-%m-%d"),
            )
            st.rerun()

    if "bt" in st.session_state and st.session_state["bt"]:
        r        = st.session_state["bt"]
        stats_df = r["stats"]; trades_df = r["trades"]; cap = r["cap"]
        deployed = 2 * cap
        pnl      = stats_df["pnl"]
        equity   = pnl.cumsum()
        dd       = equity - equity.cummax()

        # ── KPI strip ────────────────────────────────────────────────────
        cum      = pnl.sum();  cum_pct  = cum / deployed * 100
        avg_d    = pnl.mean(); avg_pct  = avg_d / deployed * 100
        std_d    = pnl.std()
        sharpe   = avg_d / std_d * np.sqrt(252) if std_d > 0 else np.nan
        max_dd   = dd.min();   dd_pct   = max_dd / deployed * 100
        t_days   = (pnl != 0).sum()
        win_days = (pnl > 0).sum()
        day_wr   = win_days / t_days * 100 if t_days > 0 else 0
        n_tr     = len(trades_df) if trades_df is not None else 0
        tr_wr    = (trades_df['ret_net'] > 0).mean() * 100 if trades_df is not None and n_tr > 0 else 0

        pnl_cls = "g" if cum >= 0 else "r"
        sh_cls  = "g" if not np.isnan(sharpe) and sharpe >= 1 else ("r" if not np.isnan(sharpe) and sharpe < 0 else "a")

        st.markdown(
            f'<div class="kpi-strip">'
            + kpi("Cumulative P&L",
                  fmt_inr(cum).split("(")[0].strip(),
                  f"{'+' if cum_pct>=0 else ''}{cum_pct:.2f}% of deployed", pnl_cls)
            + kpi("Sharpe Ratio",
                  f"{sharpe:.2f}" if not np.isnan(sharpe) else "—",
                  "annualised ×√252", sh_cls)
            + kpi("Max Drawdown",
                  f"₹{abs(max_dd):,.0f}",
                  f"{dd_pct:.2f}%", "r")
            + kpi("Day Win Rate",
                  f"{day_wr:.1f}%",
                  f"{win_days}W / {t_days-win_days}L / {(pnl==0).sum()}flat",
                  "g" if day_wr >= 50 else "r")
            + kpi("Trade Win Rate",
                  f"{tr_wr:.1f}%",
                  f"{n_tr} total trades",
                  "g" if tr_wr >= 50 else "r")
            + kpi("Calmar",
                  f"{cum/abs(max_dd):.2f}" if max_dd < 0 else "—",
                  "cum P&L / |max DD|",
                  "g" if max_dd < 0 and cum/abs(max_dd) >= 1 else "r")
            + '</div>',
            unsafe_allow_html=True,
        )

        # ── Chart + Stats side by side ────────────────────────────────
        col_chart, col_stats = st.columns([3, 1])

        with col_chart:
            sec("EQUITY CURVE")
            lc = GREEN if equity.iloc[-1] >= 0 else RED
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.68, 0.32], vertical_spacing=0.02)

            fig.add_trace(go.Scatter(
                x=equity.index, y=equity.values, mode='lines', name='Equity',
                line=dict(color=lc, width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba(0,255,136,0.05)' if lc == GREEN else 'rgba(255,59,59,0.05)',
                hovertemplate='%{x|%d %b %Y}  ₹%{y:,.0f}<extra></extra>',
            ), row=1, col=1)
            fig.add_hline(y=0, line=dict(color=BORDER2, width=1, dash='dot'), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[equity.idxmax()], y=[equity.max()], mode='markers',
                marker=dict(color=GREEN, size=7, symbol='circle'),
                showlegend=False, hovertemplate=f'Peak ₹{equity.max():,.0f}<extra></extra>',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[equity.idxmin()], y=[equity.min()], mode='markers',
                marker=dict(color=RED, size=7, symbol='circle'),
                showlegend=False, hovertemplate=f'Trough ₹{equity.min():,.0f}<extra></extra>',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values, mode='lines', name='Drawdown',
                line=dict(color=RED, width=1),
                fill='tozeroy', fillcolor='rgba(255,59,59,0.15)',
                hovertemplate='%{x|%d %b %Y}  ₹%{y:,.0f}<extra></extra>',
            ), row=2, col=1)

            fig.update_layout(height=380, showlegend=False,
                              **plotly_cfg(),
                              xaxis={**axis_style(), 'showgrid': False},
                              xaxis2={**axis_style(), 'showgrid': False},
                              yaxis=dict(**axis_style(fmt=",.0f", prefix="₹")),
                              yaxis2=dict(**axis_style(fmt=",.0f", prefix="₹")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_stats:
            sec("STATISTICS")
            best  = pnl.max(); worst = pnl.min()
            rows_stat = [
                ("Period",        f"{r['start']} → {r['end']}"),
                ("Calendar days", str(len(stats_df))),
                ("Trading days",  str(int(t_days))),
                ("Flat days",     str(int((pnl == 0).sum()))),
                ("Win days",      str(int(win_days))),
                ("Loss days",     str(int((pnl < 0).sum()))),
                ("Avg daily P&L", fmt_inr(avg_d)),
                ("Daily std dev", f"₹{std_d:,.0f}"),
                ("Best day",      fmt_inr(best)),
                ("Worst day",     fmt_inr(worst)),
                ("Avg net ret",   f"{(trades_df['ret_net'].mean()*100 if trades_df is not None else 0):.3f}%"),
                ("Cost rate",     "~0.0355% RT"),
            ]
            rows_html = "".join(
                f'<tr><td style="color:{TEXT3}">{k}</td>'
                f'<td style="color:{TEXT};text-align:right">{v}</td></tr>'
                for k, v in rows_stat
            )
            st.markdown(
                f'<table class="dt" style="margin-top:8px">{rows_html}</table>',
                unsafe_allow_html=True,
            )

        # ── Daily P&L table ───────────────────────────────────────────
        sec("DAILY P&L")

        header = (
            '<tr><th>Date</th><th>P&L (₹)</th><th>P&L %</th>'
            '<th>Long %</th><th>Short %</th><th>Trades</th><th>Win %</th></tr>'
        )
        trows = []
        for dt, row in stats_df.iterrows():
            if row['n_trades'] == 0:
                trows.append(
                    f'<tr><td class="neu">{dt.strftime("%Y-%m-%d")}</td>'
                    + '<td class="dash">—</td>' * 6 + '</tr>'
                )
            else:
                pv   = row['pnl']; pp = row['pnl_pct']
                pc   = "pos" if pv > 0 else "neg"
                lc_  = "pos" if row['long_pnl_pct']  > 0 else "neg"
                sc_  = "pos" if row['short_pnl_pct'] > 0 else "neg"
                trows.append(
                    f'<tr>'
                    f'<td class="neu">{dt.strftime("%Y-%m-%d")}</td>'
                    f'<td class="{pc}">₹{pv:>10,.0f}</td>'
                    f'<td class="{pc}">{pp:+.2f}%</td>'
                    f'<td class="{lc_}">{row["long_pnl_pct"]:+.2f}%</td>'
                    f'<td class="{sc_}">{row["short_pnl_pct"]:+.2f}%</td>'
                    f'<td class="neu">{int(row["n_trades"])}</td>'
                    f'<td class="neu">{row["win_rate"]:.1f}%</td>'
                    f'</tr>'
                )

        st.markdown(
            f'<div style="max-height:380px;overflow-y:auto;border:1px solid {BORDER};">'
            f'<table class="dt"><thead>{header}</thead><tbody>'
            + "".join(trows)
            + '</tbody></table></div>',
            unsafe_allow_html=True,
        )

        with st.expander("EXECUTION LOG"):
            st.code(r["log"] or "(empty)", language=None)

    else:
        st.markdown(f"""
        <div style='text-align:center;padding:80px 0;color:{TEXT3};'>
          <div style='font-size:40px;margin-bottom:16px;opacity:.3;'>▶</div>
          <div style='font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{TEXT2};'>
            Configure parameters above and click RUN
          </div>
          <div style='font-size:10px;margin-top:8px;'>
            Momentum Ensemble (D-1)  ×  Kakushadze 4-Factor (D)
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PHASE 1 · SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_p1:
    st.markdown('<div class="tab-body">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 1.5, 1, 5])
    with c1:
        p1_date = st.date_input(
            "Signal Date  (D-1 close)",
            value=date.today(),
            key="p1_date",
        )
    with c2:
        p1_cap = st.number_input("Capital / Leg (₹)", min_value=50_000,
                                 max_value=50_00_000, value=10_00_000,
                                 step=50_000, key="p1_cap", format="%d")
    with c3:
        p1_topn = st.number_input("Top N", min_value=1, max_value=10,
                                  value=2, step=1, key="p1_topn")
    with c4:
        st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
        p1_run = st.button("▶ RUN PHASE 1", key="p1_run")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style='margin-top:10px;font-size:10px;color:{TEXT3};line-height:2;'>
          Run <strong style='color:{TEXT2}'>after market close</strong> on the evening before your trade day.
          Computes 8-alpha momentum signals for all Nifty 100 stocks.
          Saves <span style='color:{AMBER}'>momentum_signals.csv</span> for Phase 2.
        </div>
        """, unsafe_allow_html=True)

    if p1_run:
        from combined_strategy import phase1
        with st.spinner("Computing momentum signals…"):
            log_buf = io.StringIO()
            with redirect_stdout(log_buf):
                result = phase1(signal_date=p1_date.strftime("%Y-%m-%d"), verbose=True)
        st.session_state["p1"] = {**result, "log": log_buf.getvalue()}
        st.rerun()

    if "p1" in st.session_state and st.session_state["p1"]:
        r   = st.session_state["p1"]
        sig = r["signals_df"]
        sm  = r["summary"]

        # Summary KPIs
        st.markdown(
            f'<div class="kpi-strip" style="grid-template-columns:repeat(4,1fr);margin-top:12px;">'
            + kpi("Total Stocks", str(sm["total"]), r["signal_date"])
            + kpi("BUY Signals",  str(sm["buy"]),   f"{sm['buy']/sm['total']*100:.0f}% of universe", "g")
            + kpi("SELL Signals", str(sm["sell"]),  f"{sm['sell']/sm['total']*100:.0f}% of universe", "r")
            + kpi("NEUTRAL",      str(sm["neutral"]), f"{sm['neutral']/sm['total']*100:.0f}% of universe", "")
            + '</div>',
            unsafe_allow_html=True,
        )

        # Signals table
        sec("MOMENTUM SIGNAL TABLE  ·  8-ALPHA ENSEMBLE")

        alpha_cols = [c for c in sig.columns if c.startswith("a")]
        header_cells = (
            '<th>Ticker</th><th>Signal</th>'
            '<th>Score</th><th>Confidence</th>'
            + "".join(f'<th>{c.upper()}</th>' for c in alpha_cols)
        )

        trows = []
        for _, row in sig.sort_values("score", ascending=False).iterrows():
            def alpha_cell(v):
                if v == 1:  return f'<td class="pos">+1</td>'
                if v == -1: return f'<td class="neg">-1</td>'
                return f'<td class="dash">0</td>'

            alpha_cells = "".join(alpha_cell(row[c]) for c in alpha_cols) if alpha_cols else ""
            sc = row["score"]; sc_cls = "pos" if sc > 0 else ("neg" if sc < 0 else "dash")
            conf_cls = "g" if row["confidence"] >= 75 else ("a" if row["confidence"] >= 50 else "")
            trows.append(
                f'<tr>'
                f'<td class="neu">{row["ticker"]}</td>'
                f'<td>{badge(row["signal"])}</td>'
                f'<td class="{sc_cls}">{int(sc):+d}</td>'
                f'<td class="{conf_cls}">{row["confidence"]:.1f}%</td>'
                + alpha_cells
                + '</tr>'
            )

        st.markdown(
            f'<div style="max-height:500px;overflow-y:auto;border:1px solid {BORDER};">'
            f'<table class="dt"><thead><tr>{header_cells}</tr></thead>'
            f'<tbody>{"".join(trows)}</tbody></table></div>',
            unsafe_allow_html=True,
        )

        with st.expander("EXECUTION LOG"):
            st.code(r["log"] or "(empty)", language=None)

    else:
        st.markdown(f"""
        <div style='text-align:center;padding:80px 0;color:{TEXT3};'>
          <div style='font-size:40px;margin-bottom:16px;opacity:.3;'>🌙</div>
          <div style='font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{TEXT2};'>
            Run after market close — select D-1 date and click RUN PHASE 1
          </div>
          <div style='font-size:10px;margin-top:8px;'>
            8 Alpha Signals  ·  Majority Vote  ·  Saves to momentum_signals.csv
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PHASE 2 · TRADES
# ══════════════════════════════════════════════════════════════════════════════
with tab_p2:
    st.markdown('<div class="tab-body">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 1.5, 1, 3.5])
    with c1:
        p2_date = st.date_input("Trade Date  (D execution)", value=date.today(), key="p2_date")
    with c2:
        p2_cap = st.number_input("Capital / Leg (₹)", min_value=50_000,
                                 max_value=50_00_000, value=10_00_000,
                                 step=50_000, key="p2_cap", format="%d")
    with c3:
        p2_topn = st.number_input("Top N", min_value=1, max_value=10,
                                  value=2, step=1, key="p2_topn")
    with c4:
        st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
        p2_run = st.button("▶ RUN PHASE 2", key="p2_run")
        st.markdown("</div>", unsafe_allow_html=True)

    # Filter controls
    st.markdown(f'<div class="sec-hdr">FILTERS</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1.5, 3.5])
    with fc1:
        f_results = st.toggle("Block Results Day", value=True, key="f_results")
    with fc2:
        f_news    = st.toggle("Block Bad News Sentiment", value=True, key="f_news")
    with fc3:
        f_x       = st.toggle("Block Bad X Sentiment", value=False, key="f_x")
    with fc4:
        x_token   = st.text_input("X Bearer Token (optional)",
                                  value="", placeholder="paste token to enable X sentiment",
                                  type="password", key="x_token", label_visibility="visible")

    if p2_run:
        from combined_strategy import phase2
        with st.spinner("Running 4-Factor confirmation + sentiment filters…"):
            log_buf = io.StringIO()
            with redirect_stdout(log_buf):
                result = phase2(
                    trade_date=p2_date.strftime("%Y-%m-%d"),
                    verbose=True,
                    capital=p2_cap,
                    top_n=p2_topn,
                    x_bearer_token=x_token or None,
                    block_results_day=f_results,
                    block_bad_news=f_news,
                    block_bad_x=f_x,
                )
        if result:
            st.session_state["p2"] = {**result, "log": log_buf.getvalue()}
        st.rerun()

    if "p2" in st.session_state and st.session_state["p2"]:
        r = st.session_state["p2"]

        # Alerts
        if r.get("error"):
            st.markdown(f'<div class="alert-err">{r["error"]}</div>',
                        unsafe_allow_html=True)
        if r.get("stale_warning"):
            st.markdown(f'<div class="alert-warn">⚠  {r["stale_warning"]}</div>',
                        unsafe_allow_html=True)
        if not r.get("today_available", True):
            st.markdown(
                f'<div class="alert-info">ℹ  Today\'s open not yet in yfinance '
                f'— overnight gap computed from D-2→D-1. Re-run after ~09:30.</div>',
                unsafe_allow_html=True,
            )

        # ── Signal agreement table ────────────────────────────────────
        if r.get("signal_table") is not None:
            sec(f"SIGNAL TABLE  ·  TRADE DATE {r['trade_date']}  ·  SIGNALS FROM {r['signal_date']}")

            st_df = r["signal_table"]
            header = '<tr><th>Stock</th><th>4-Factor</th><th>Momentum</th><th>Final</th><th>Residual</th><th>ε Norm</th></tr>'
            trows  = []
            for _, row in st_df.iterrows():
                fin_cls = "pos" if row["final"] in ("BUY",) else ("neg" if row["final"] == "SELL" else "neu")
                trows.append(
                    f'<tr>'
                    f'<td class="neu">{row["stock"]}</td>'
                    f'<td>{badge(row["4factor"])}</td>'
                    f'<td>{badge(row["momentum"])}</td>'
                    f'<td>{badge(row["final"])}</td>'
                    f'<td class="neu">{row["residual"]:.6f}</td>'
                    f'<td class="neu">{row["eps_norm"]:.6f}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="border:1px solid {BORDER};">'
                f'<table class="dt"><thead>{header}</thead><tbody>{"".join(trows)}</tbody></table></div>',
                unsafe_allow_html=True,
            )

        # ── Confirmed trades ──────────────────────────────────────────
        confirmed = r.get("confirmed")
        if confirmed is not None and len(confirmed) > 0:
            sec("CONFIRMED TRADE ORDERS")

            has_pnl = "pnl" in confirmed.columns and r.get("pnl") is not None

            if has_pnl:
                # Historical / sim mode — show full P&L
                pnl_s = r["pnl"]
                st.markdown(
                    f'<div class="kpi-strip" style="grid-template-columns:repeat(5,1fr);margin-bottom:12px;">'
                    + kpi("Total P&L",   fmt_inr(pnl_s["total_pnl"], pnl_s["total_pnl"]/pnl_s["deployed"]*100),
                          f"on ₹{pnl_s['deployed']:,.0f} deployed",
                          "g" if pnl_s["total_pnl"] >= 0 else "r")
                    + kpi("Long P&L",    fmt_inr(pnl_s["long_pnl"]),  "", "g" if pnl_s["long_pnl"] >= 0 else "r")
                    + kpi("Short P&L",   fmt_inr(pnl_s["short_pnl"]), "", "g" if pnl_s["short_pnl"] >= 0 else "r")
                    + kpi("Win Rate",    f"{pnl_s['win_rate']:.1f}%", f"{len(confirmed)} trades",
                          "g" if pnl_s["win_rate"] >= 50 else "r")
                    + kpi("Avg Net Ret", f"{pnl_s['avg_ret']:.3f}%",  "per trade", "")
                    + '</div>',
                    unsafe_allow_html=True,
                )
                header = ('<tr><th>Stock</th><th>Side</th><th>Entry</th>'
                          '<th>Exit</th><th>Stop</th><th>Position</th>'
                          '<th>Ret Net</th><th>P&L</th></tr>')
                trows = []
                for _, row in confirmed.iterrows():
                    pc = "pos" if row["pnl"] >= 0 else "neg"
                    trows.append(
                        f'<tr>'
                        f'<td class="neu">{row["stock"]}</td>'
                        f'<td>{badge(row["side"])}</td>'
                        f'<td class="neu">₹{row["entry"]:,.2f}</td>'
                        f'<td class="neu">₹{row["exit"]:,.2f}</td>'
                        f'<td class="neg">₹{row["stop"]:,.2f}</td>'
                        f'<td class="neu">₹{row["position"]:,.0f}</td>'
                        f'<td class="{pc}">{row["ret_net"]*100:+.3f}%</td>'
                        f'<td class="{pc}">₹{row["pnl"]:,.0f}</td>'
                        f'</tr>'
                    )
            else:
                # Live mode — entry + stop only
                st.markdown(
                    f'<div class="alert-info">LIVE MODE — Market open. '
                    f'Entry prices confirmed. Exit and P&L pending market close.</div>',
                    unsafe_allow_html=True,
                )
                header = ('<tr><th>Stock</th><th>Side</th><th>Entry Price</th>'
                          '<th>Stop-Loss</th><th>Position</th><th>Status</th></tr>')
                trows = []
                for _, row in confirmed.iterrows():
                    trows.append(
                        f'<tr>'
                        f'<td class="neu">{row["stock"]}</td>'
                        f'<td>{badge(row["side"])}</td>'
                        f'<td class="amb">₹{row["entry"]:,.2f}</td>'
                        f'<td class="neg">₹{row["stop"]:,.2f}</td>'
                        f'<td class="neu">₹{row["position"]:,.0f}</td>'
                        f'<td>{badge("CONFIRMED")}</td>'
                        f'</tr>'
                    )

            st.markdown(
                f'<div style="border:1px solid {BORDER};">'
                f'<table class="dt"><thead>{header}</thead>'
                f'<tbody>{"".join(trows)}</tbody></table></div>',
                unsafe_allow_html=True,
            )

        elif r.get("signal_table") is not None:
            st.markdown(
                f'<div class="alert-warn">No trades confirmed — '
                f'all candidates blocked by dual-agreement filter.</div>',
                unsafe_allow_html=True,
            )

        # ── Sentiment & filter details ─────────────────────────────────
        fr = r.get("filter_results", {})
        if fr:
            sec("SENTIMENT  &  RISK FILTERS")
            hdr = ('<tr><th>Stock</th><th>Side</th>'
                   '<th>News Signal</th><th>News Score</th>'
                   '<th>X Signal</th><th>Results Day</th><th>Verdict</th></tr>')
            rows_s = []
            for stock, res in fr.items():
                side_val = ""
                confirmed_df = r.get("confirmed")
                for _, conf_row in (confirmed_df if confirmed_df is not None else pd.DataFrame()).iterrows():
                    if conf_row.get("stock", "").replace(".NS","") == stock:
                        side_val = conf_row["side"]
                        break

                news = res["details"].get("news", {})
                xdat = res["details"].get("x", {})
                eday = res["details"].get("results", {})

                ns  = news.get("signal", "—")
                nsc = news.get("score", 0)
                nsc_cls = "pos" if nsc > 0.05 else ("neg" if nsc < -0.05 else "neu")

                xs  = xdat.get("signal", "N/A") if xdat.get("available") else "NO TOKEN"
                rd  = "YES ⚠" if eday.get("is_results_day") else "NO"
                rd_cls = "neg" if eday.get("is_results_day") else "pos"

                verdict = "PASS" if res["pass"] else "BLOCKED"
                v_cls   = "pos" if res["pass"] else "neg"

                rows_s.append(
                    f'<tr>'
                    f'<td class="neu">{stock}</td>'
                    f'<td>{badge(side_val) if side_val else "—"}</td>'
                    f'<td style="font-size:9px;color:{TEXT3};">{res["reason"]}</td>'
                    f'<td>{badge(ns) if ns not in ("—","") else "—"}</td>'
                    f'<td class="{nsc_cls}">{nsc:+.3f}</td>'
                    f'<td class="neu">{xs}</td>'
                    f'<td class="{rd_cls}">{rd}</td>'
                    f'<td class="{v_cls}" style="font-weight:700;">{verdict}</td>'
                    f'</tr>'
                )

            st.markdown(
                f'<div style="border:1px solid {BORDER};margin-bottom:12px;">'
                f'<table class="dt"><thead>{hdr}</thead>'
                f'<tbody>{"".join(rows_s)}</tbody></table></div>',
                unsafe_allow_html=True,
            )

            # News headlines for each stock (expandable)
            for stock, res in fr.items():
                articles = res["details"].get("news", {}).get("articles", [])
                if articles:
                    with st.expander(f"News headlines — {stock}"):
                        for a in articles:
                            sc  = a.get("score", 0)
                            cls = "🟢" if sc > 0.05 else ("🔴" if sc < -0.05 else "⚪")
                            st.markdown(
                                f'<div style="font-size:10px;color:{TEXT};padding:4px 0;'
                                f'border-bottom:1px solid {BORDER};">'
                                f'{cls} <span style="color:{TEXT3};">[{sc:+.3f}]</span>'
                                f'  {a["title"]}'
                                f'  <span style="color:{TEXT3};font-size:9px;">— {a.get("publisher","")}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        with st.expander("EXECUTION LOG"):
            st.code(r.get("log", "(empty)"), language=None)

    else:
        st.markdown(f"""
        <div style='text-align:center;padding:80px 0;color:{TEXT3};'>
          <div style='font-size:40px;margin-bottom:16px;opacity:.3;'>📋</div>
          <div style='font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{TEXT2};'>
            Run at 09:15 IST — ensure Phase 1 has been run the evening before
          </div>
          <div style='font-size:10px;margin-top:8px;'>
            4-Factor Cross-Sectional OLS  ·  Dual-Agreement Gate  ·  Trade Orders
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ORB · LIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_orb:
    st.markdown('<div class="tab-body">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 5])
    with c1:
        orb_date = st.date_input("Trading Date", value=date.today(), key="orb_date")
    with c2:
        st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
        orb_run = st.button("▶ RUN", key="orb_run")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style='margin-top:8px;font-size:10px;color:{TEXT3};line-height:2;'>
          Run <strong style='color:{TEXT2};'>after 9:30 AM IST</strong> on trade day.
          Compares today's first 15-min opening range vs each stock's 45-day baseline.
          Stocks with <strong style='color:{GREEN};'>Composite Z ≥ 2.0</strong>
          show abnormal opening energy — high-conviction ORB candidates.
        </div>
        """, unsafe_allow_html=True)

    if orb_run:
        from orb_opening_momentum_screener import run_screener as _run_orb_live

        _date_str = orb_date.strftime("%Y-%m-%d")
        log_buf   = io.StringIO()
        orb_df    = None
        orb_err   = None

        try:
            with st.spinner("Fetching 5-min data & computing Z-scores — ~40 seconds…"):
                with redirect_stdout(log_buf):
                    orb_df = _run_orb_live(target_date=_date_str)
        except Exception as e:
            orb_err = str(e)

        st.session_state["orb"] = {
            "orb_df":  orb_df,
            "date":    _date_str,
            "log":     log_buf.getvalue(),
            "orb_err": orb_err,
        }
        st.rerun()

    if "orb" in st.session_state and st.session_state["orb"]:
        r      = st.session_state["orb"]
        orb_df = r["orb_df"]

        if r.get("orb_err"):
            st.markdown(f'<div class="alert-err">Screener error: {r["orb_err"]}</div>',
                        unsafe_allow_html=True)

        if orb_df is not None:
            z2_df  = orb_df[orb_df["Composite Z"] >= 2.0]
            best_z = orb_df["Composite Z"].max()
            up_n   = len(orb_df[orb_df["Direction"].str.strip() == "UP"])
            dn_n   = len(orb_df[orb_df["Direction"].str.strip() == "DOWN"])

            # ── KPI strip ────────────────────────────────────────────────
            st.markdown(
                f'<div class="kpi-strip" style="grid-template-columns:repeat(5,1fr);margin-top:12px;">'
                + kpi("Stocks Screened",  str(len(orb_df)),  r["date"])
                + kpi("Anomaly  (Z ≥ 2)", str(len(z2_df)),  "strong ORB candidates",
                      "g" if len(z2_df) > 0 else "")
                + kpi("Best Composite Z", f"{best_z:.2f}",  orb_df.iloc[0]["Ticker"],
                      "g" if best_z >= 2 else "a")
                + kpi("LONG Candidates",  str(up_n),         "first 15 min UP",   "g")
                + kpi("SHORT Candidates", str(dn_n),         "first 15 min DOWN", "r")
                + '</div>',
                unsafe_allow_html=True,
            )

            # ── Anomaly cards (Z >= 2) ────────────────────────────────────
            sec(f"ORB ANOMALY ALERTS  ·  COMPOSITE Z ≥ 2.0  ·  {r['date']}")
            if len(z2_df) > 0:
                cards_html = ""
                for _, row in z2_df.iterrows():
                    is_up   = row["Direction"].strip() == "UP"
                    txt_col = GREEN if is_up else RED
                    bg_col  = "#001a08" if is_up else "#1a0000"
                    brd_col = GREEN if is_up else RED
                    arrow   = "▲" if is_up else "▼"
                    cards_html += (
                        f'<div style="display:inline-block;background:{bg_col};'
                        f'border:2px solid {brd_col};padding:10px 20px;'
                        f'margin:6px 8px 6px 0;font-family:Courier New,monospace;">'
                        f'<div style="font-size:16px;font-weight:700;color:{txt_col};'
                        f'letter-spacing:.12em;">{arrow} {row["Ticker"]}</div>'
                        f'<div style="font-size:10px;color:{txt_col};opacity:.75;margin-top:3px;">'
                        f'Z = {row["Composite Z"]:.2f}  ·  {row["Direction"].strip()}</div>'
                        f'</div>'
                    )
                st.markdown(f'<div style="margin-bottom:14px;">{cards_html}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-info">No stocks above Z = 2.0 — '
                    f'market may still be in its normal opening phase.</div>',
                    unsafe_allow_html=True,
                )

            # ── Full ranked table ─────────────────────────────────────────
            sec("ALL STOCKS  ·  RANKED BY COMPOSITE Z-SCORE")

            def _fv(v, d=2):
                return f"{v:.{d}f}" if isinstance(v, float) and not np.isnan(v) else "—"

            tbl_hdr = (
                '<tr><th>Rank</th><th>Ticker</th><th>Composite Z</th>'
                '<th>Range Z</th><th>Dir Z</th><th>Range Ratio</th>'
                '<th>Today Range</th><th>Hist Avg</th><th>Dir</th><th>Days</th></tr>'
            )
            tbl_rows = []
            for _, row in orb_df.iterrows():
                is_hot  = row["Composite Z"] >= 2.0
                is_up   = row["Direction"].strip() == "UP"
                bg      = ('style="background:#001a08;"' if (is_hot and is_up)
                           else 'style="background:#1a0000;"' if (is_hot and not is_up)
                           else "")
                cz      = row["Composite Z"]
                cz_cls  = "pos" if cz >= 2 else ("amb" if cz >= 1 else "neu")
                tkr_col = (GREEN if (is_hot and is_up)
                           else RED if (is_hot and not is_up)
                           else TEXT)
                tkr_wt  = "700" if is_hot else "400"
                dir_td  = (f'<td class="pos">▲ UP</td>' if is_up
                           else f'<td class="neg">▼ DN</td>')
                tbl_rows.append(
                    f'<tr {bg}>'
                    f'<td class="neu">{int(row["Rank"])}</td>'
                    f'<td style="color:{tkr_col};font-weight:{tkr_wt};">'
                    f'{row["Ticker"]}{"  ★" if is_hot else ""}</td>'
                    f'<td class="{cz_cls}">{_fv(cz, 3)}</td>'
                    f'<td class="neu">{_fv(row["Range Z"], 3)}</td>'
                    f'<td class="neu">{_fv(row["Dir Z"], 3)}</td>'
                    f'<td class="neu">{_fv(row["Range Ratio"], 2)}</td>'
                    f'<td class="neu">{_fv(row["Today Range"], 2)}</td>'
                    f'<td class="neu">{_fv(row["Hist Avg Range"], 2)}</td>'
                    + dir_td
                    + f'<td class="neu">{int(row["Hist Days"])}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="max-height:520px;overflow-y:auto;border:1px solid {BORDER};">'
                f'<table class="dt"><thead>{tbl_hdr}</thead>'
                f'<tbody>{"".join(tbl_rows)}</tbody></table></div>',
                unsafe_allow_html=True,
            )

        elif not r.get("orb_err"):
            st.markdown(
                f'<div class="alert-warn">No data returned — ensure the market has been open '
                f'for at least 15 minutes and try again.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("EXECUTION LOG"):
            st.code(r.get("log", "(empty)"), language=None)

    else:
        st.markdown(f"""
        <div style='text-align:center;padding:80px 0;color:{TEXT3};'>
          <div style='font-size:40px;margin-bottom:16px;opacity:.3;'>📡</div>
          <div style='font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{TEXT2};'>
            Run after 9:30 AM IST — compares first 15 min vs 45-day historical baseline
          </div>
          <div style='font-size:10px;margin-top:8px;'>
            5-Min OHLCV  ·  Range Z + Direction Z  ·  Composite Z ≥ 2.0 = ORB Candidate
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
