# DAE Terminal — Dual-Agreement Intraday Trading Engine

A Bloomberg-style quantitative trading terminal for NSE equities. It screens the entire Nifty 100 universe using two independent signal engines and only places a trade when both agree — the "dual-agreement" gate.

## What It Does

The system runs a two-layer signal confirmation process across all Nifty 100 stocks before producing a trade order:

**Layer 1 — Momentum Ensemble (run evening before)**
Scores every stock using 8 independent price-momentum alphas. Each alpha votes +1 (bullish), -1 (bearish), or 0 (neutral). The majority vote determines the signal (BUY / SELL / NEUTRAL) along with a confidence score.

**Layer 2 — 4-Factor Cross-Sectional Model (run morning of trade)**
Applies a Kakushadze-style 4-factor OLS regression to intraday price data to estimate each stock's residual alpha (the return unexplained by market/sector factors). High positive residual = potential long. High negative residual = potential short.

**Dual-Agreement Gate**
A trade is confirmed only when both Layer 1 and Layer 2 point in the same direction. Stocks where the two models disagree are discarded.

**Sentiment & Risk Filters**
Before finalising orders, confirmed candidates are screened for:
- Results day (earnings announcement — position blocked)
- News sentiment (negative headline scan via yfinance)
- X/Twitter sentiment (optional, requires bearer token)

**Execution**
- Entry at 09:16 IST (one minute after market open)
- Stop-loss at 1% from entry
- Equal capital allocation per leg (long and short independently sized)
- Round-trip transaction cost ~0.0355%

## Three Modules

| Tab | Purpose |
|---|---|
| Backtest | Run the full strategy over any historical date range with P&L, Sharpe, drawdown, and trade-level breakdown |
| Phase 1 · Signals | Evening scan — compute and save momentum signals for next day |
| Phase 2 · Trades | Morning confirmation — apply 4-factor model, dual-agreement gate, sentiment filters, produce trade orders |

## Tech Stack

Python · Streamlit · yfinance · pandas · NumPy · SciPy · Plotly · VADER Sentiment

---

*This tool is for research and informational purposes. Not financial advice.*
