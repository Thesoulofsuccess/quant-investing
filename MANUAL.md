# DAE Terminal — Process Manual

## Overview

The strategy runs on a two-phase daily cycle:

- **Phase 1** — Run the evening before (after market close, ~15:45 IST onwards)
- **Phase 2** — Run the morning of the trade (before market open, ~09:00–09:14 IST)

---

## Daily Workflow

### Evening Before (D-1)

**Step 1 — Open the terminal**
Navigate to the **PHASE 1 · SIGNALS** tab.

**Step 2 — Set parameters**

| Field | What to enter |
|---|---|
| Signal Date | Yesterday's date (D-1, the last trading day) |
| Capital / Leg | Amount allocated to the long side (same amount will go to short side) |
| Top N | How many stocks to select per side (2 is recommended to start) |

**Step 3 — Run**
Click **▶ RUN PHASE 1**.

The engine downloads price data for all Nifty 100 stocks and scores each one across 8 momentum alphas. This takes ~1–2 minutes.

**Step 4 — Review output**
- Check the signal table — BUY signals are long candidates, SELL signals are short candidates
- Higher confidence % = stronger agreement across the 8 alphas
- The signals are automatically saved to `momentum_signals.csv` for Phase 2

---

### Morning of Trade (D)

**Step 1 — Navigate to Phase 2**
Open the **PHASE 2 · TRADES** tab.

**Step 2 — Set parameters**

| Field | What to enter |
|---|---|
| Trade Date | Today's date |
| Capital / Leg | Must match what you used in Phase 1 |
| Top N | Must match Phase 1 |

**Step 3 — Set filters**

| Filter | Recommended | What it does |
|---|---|---|
| Block Results Day | ON | Skips stocks announcing earnings today |
| Block Bad News Sentiment | ON | Blocks stocks with negative headline scores |
| Block Bad X Sentiment | OFF (unless you have a token) | Checks Twitter/X for negative sentiment |

**Step 4 — Run**
Click **▶ RUN PHASE 2** at or just before 09:15 IST.

The engine applies the 4-factor model, checks dual-agreement, runs sentiment filters, and produces the final trade list.

**Step 5 — Review confirmed orders**
The confirmed trades table shows:
- Stock ticker
- Side (BUY = long, SELL = short)
- Entry price (@ 09:16 open)
- Stop-loss level (1% from entry)
- Position size in ₹

**Step 6 — Execute**
Place the orders manually in your broker at 09:16 IST. Both the long and short legs are intraday — close all positions before 15:15 IST.

---

## Backtesting

Use the **BACKTEST** tab to test the strategy over any historical period before deploying capital.

| Field | Description |
|---|---|
| Start / End Date | Date range to test over |
| Capital / Leg | Notional capital per side |
| Top N | Number of stocks per side |

Click **▶ RUN**. The backtest will show:
- Cumulative P&L and return %
- Sharpe ratio (annualised)
- Maximum drawdown
- Day-level and trade-level win rates
- Calmar ratio
- Full equity curve with drawdown panel
- Daily P&L table

---

## Important Notes

- Phase 1 **must** be run before Phase 2. If `momentum_signals.csv` is missing or stale, Phase 2 will warn you.
- If yfinance hasn't updated today's open price yet (before ~09:30), Phase 2 will show a blue info banner — re-run after 09:30 for accurate entry prices.
- The strategy is market-neutral (long + short), so total capital deployed = Capital / Leg × 2.
- Stop-loss is 1% per position. Size your capital accordingly.
- All costs (~0.0355% round-trip) are baked into the backtest results.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Phase 2 says signals are stale | Re-run Phase 1 with today's D-1 date |
| No confirmed trades | Both models disagreed on all candidates — no trade today |
| yfinance data error | Market may be closed or data delayed — retry after a few minutes |
| X sentiment not working | Requires a valid X Developer bearer token pasted into the token field |
