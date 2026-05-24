# Combined Strategy — Technical Reference

**Momentum Ensemble (D-1) + Kakushadze 4-Factor Mean-Reversion (D)**  
Universe: Nifty 100 | Market: NSE Equities | Style: Intraday Long/Short

---

## 1. Strategy Overview

A dual-engine intraday framework that trades Nifty 100 equities.  
Two independent signal engines must **agree** on direction before a trade fires.

| Engine | Signal Timing | Role |
|---|---|---|
| Momentum Ensemble | Evening of D-1 (pre-market) | Directional filter |
| 4-Factor Mean-Reversion | Open of D (intraday) | Primary alpha source |

The agreement gate is the core edge: trades only execute when both engines point the same way. This eliminates low-conviction setups and reduces false signals from either engine in isolation.

---

## 2. Universe

- **Index**: Nifty 100 (top 100 NSE-listed companies by free-float market cap)
- **Source**: `nifty100.csv` — NSE ticker symbols with `.NS` suffix for yfinance
- **Data Provider**: Yahoo Finance via `yfinance` (free, real-time delayed)
- **Adjusted Prices**: Momentum engine uses split/dividend-adjusted closes; 4-Factor uses raw OHLCV with `Adj Close` for the price factor only

---

## 3. Engine 1 — Momentum Ensemble (D-1)

Computed on **closing data of the day before trade date (D-1)**.  
An ensemble of 8 independent alpha signals votes majority-rule.

### Alpha Signals

| Alpha | Logic | Source |
|---|---|---|
| **α-101** | Sign of (Close−Open)/(High−Low) | WorldQuant 101 Alphas |
| **α-7** | 7-day price delta direction, conditioned on volume > ADV₂₀ | WorldQuant 101 Alphas |
| **α-19** | Sign of 7-day price change | WorldQuant 101 Alphas |
| **α-39** | 7-day price delta direction when volume < ADV₂₀ (low-volume confirmation) | WorldQuant 101 Alphas |
| **α-83** | Intraday direction when HL-range exceeds 3% of 5-day MA | WorldQuant 101 Alphas |
| **α-35** | 32-day return rank: top tercile = BUY, bottom tercile = SELL | WorldQuant 101 Alphas |
| **α-71** | Sign of (5-day MA − 20-day MA) crossover | WorldQuant 101 Alphas |
| **α-56** | Sign of 10-day return vs 30-day return ratio | WorldQuant 101 Alphas |

**ADV₂₀** = 20-day rolling average of dollar volume (Close × Volume).  
**Score** = sum of individual votes (each ∈ {−1, 0, +1}).

```
Signal = BUY   if score > 0
       = SELL  if score < 0
       = NEUTRAL (no trade) if score = 0
```

**Minimum history required**: 100 bars. Stocks with insufficient history are skipped.

---

## 4. Engine 2 — Kakushadze 4-Factor (D)

Computed using **overnight data visible at market open of D**.  
Implements the cross-sectional mean-reversion model from Kakushadze (2015).

### Factors

| Factor | Formula | Intuition |
|---|---|---|
| **prc** | log(Adj Close_{D-1}) | Price level (size proxy) |
| **mom** | log(Close_{D-1} / Open_{D-1}) | Previous-day intraday momentum |
| **hlv** | 0.5 × log(mean((H−L)²/C²) over D days) | High-low volatility (Garman-Klass proxy) |
| **vol** | log(mean(Volume) over D days) | Average log volume |

**Overnight return** (target variable):
```
overnight = log(Open_D / Close_{D-1})
```

### Cross-Sectional OLS

Each day, for all stocks with sufficient data:

```
overnight_i = β₀ + β₁·prc_i + β₂·mom_i + β₃·hlv_n_i + β₄·vol_n_i + ε_i
```

`hlv` and `vol` are Gaussian-rank-normalised cross-sectionally before regression (`cross_normalize`).

### Signal Construction

Residuals `ε_i` are Gaussian-rank-normalised (`normalize_residuals`):

```
ε_norm_i = μ + σ × Φ⁻¹((rank_i − 0.5) / N)
```

**Mean-reversion logic**: stocks with the **highest** residual overnight return are expected to revert → **SELL**. Stocks with the **lowest** residual → **BUY**.

```
Top TOP_N by residual  → SELL candidates
Bot TOP_N by residual  → BUY  candidates
```

---

## 5. Dual-Agreement Gate

For each candidate (up to `TOP_N` per side), a trade is confirmed **only if**:

```
4-Factor side == Momentum signal
```

| 4-Factor | Momentum | Trade? |
|---|---|---|
| BUY | BUY | ✅ Confirmed |
| SELL | SELL | ✅ Confirmed |
| BUY | SELL or NEUTRAL | ❌ Blocked |
| SELL | BUY or NEUTRAL | ❌ Blocked |

If no candidates are confirmed, the day produces no trades (flat).  
If fewer than `TOP_N × 2` candidates are confirmed, a partial-deployment warning is logged.

---

## 6. Execution Model

All trades are **intraday**: entry at open, exit at close.

### Entry
| Side | Entry Price |
|---|---|
| BUY | `Open × (1 + slippage)` |
| SELL | `Open × (1 − slippage)` |

### Exit
A **stop-loss** check is applied against the intraday high/low:

| Side | Stop Triggered If | Stop Exit | Normal Exit |
|---|---|---|---|
| BUY | `Low ≤ entry × (1 − SL)` | `entry × (1 − SL) × (1 − slip)` | `Close × (1 − slip)` |
| SELL | `High ≥ entry × (1 + SL)` | `entry × (1 + SL) × (1 + slip)` | `Close × (1 + slip)` |

**Default parameters**: slippage = 0.05%, stop-loss = 1.0%

### Position Sizing
Fixed notional per slot:
```
position_per_trade = CAPITAL / TOP_N
```
Total long exposure = `CAPITAL`, total short exposure = `CAPITAL` (market-neutral by design).

---

## 7. Cost Model

Full Indian NSE intraday cost stack, applied as a constant round-trip rate:

| Component | Rate | Applied |
|---|---|---|
| Exchange transaction charge | 0.00297% | Both legs |
| SEBI turnover fee | 0.001% | Both legs |
| IPFT | 0.001% | Both legs |
| GST (18% on above three) | 0.18 × above | Both legs |
| Stamp duty | 0.003% | Entry only |
| STT | 0.025% | Exit only |

**Effective round-trip COST_RATE ≈ 0.0355% of notional**

```python
_BASE_RATE = 0.0000297 + 0.000001 + 0.000001
COST_RATE  = 2 * _BASE_RATE * (1 + 0.18) + 0.00003 + 0.00025
```

Net return per trade: `ret_net = ret_gross − COST_RATE`

---

## 8. Performance Metrics

| Metric | Definition |
|---|---|
| Cumulative P&L | Sum of daily net P&L across all trades |
| Avg Daily P&L | Mean of daily P&L over **all** calendar days in window |
| Daily Std Dev | Std of daily P&L over **all** calendar days (flat days = 0) |
| Sharpe (annualised) | `(avg_daily / std_daily) × √252` — includes flat days |
| Max Drawdown | Min of (cumulative P&L − running peak) |
| Trade Win Rate | % of individual trades with positive net return |
| Deployed Capital | `2 × CAPITAL` (long leg + short leg) |

> **Sharpe note**: computed over all business days in the window, including flat days (which contribute 0 to mean and pull Sharpe down vs. a trading-days-only calculation). This is the conservative, standard methodology.

---

## 9. Data Pipeline

```
prefetch_all()          ← download once for full window + history buffer
    │
    ├── raw_4f          auto_adjust=False  (4-Factor needs raw OHLCV + Adj Close)
    └── raw_mom         auto_adjust=True   (Momentum uses adjusted closes)

For each trading day D:
    get_momentum_signals(raw_mom, D-1)   → {ticker: BUY/SELL/NEUTRAL}
    run_day(raw_4f, signals, D)          → trades DataFrame or None
```

**History buffer**: `max(D×2, LOOKBACK×3) = 600` calendar days pre-start to ensure full lookback for all signals on day 1 of the window.  
**End buffer**: 5 business days beyond `BACKTEST_END` to avoid clipping the final week.

---

## 10. CLI Usage

```bash
# Default run (2025-01-01 to 2025-06-30, ₹10L per leg, TOP_N=2)
python backtest.py

# Custom date range
python backtest.py --start 2024-01-01 --end 2024-12-31

# Full parameter control
python backtest.py --start 2024-01-01 --end 2025-06-30 --capital 500000 --top-n 3
```

| Flag | Default | Description |
|---|---|---|
| `--start` | 2025-01-01 | Backtest start date |
| `--end` | 2025-06-30 | Backtest end date |
| `--capital` | 10,00,000 | Rupee notional per leg |
| `--top-n` | 2 | Stocks per side from 4-Factor engine |

**Outputs**:
- Console: per-day P&L table + aggregate stats block
- `equity_curve.png`: dark-theme equity curve with drawdown panel

---

## 11. Key Design Decisions

**Why dual-agreement?**  
Each engine has independent alpha sources. Agreement filters out days where one engine is noisy. Empirically, this reduces trade frequency but improves per-trade quality.

**Why mean-reversion + momentum?**  
The 4-Factor model exploits overnight gaps reverting to fair value (short-term mean-reversion). The momentum ensemble confirms that the broader directional trend supports the trade, reducing adverse selection from true trending names.

**Why intraday (open-to-close)?**  
Eliminates overnight gap risk. The overnight signal is used for ranking, not for holding. Intraday execution is fully self-contained within market hours (09:15–15:30 IST).

**Why fixed notional sizing?**  
Phase 1 baseline. Volatility-scaled sizing (e.g., `CAPITAL / (TOP_N × σ_hl)`) is a natural Phase 2 enhancement.

---

## 12. Files

| File | Purpose |
|---|---|
| `backtest.py` | Main engine — run this |
| `nifty100.csv` | Nifty 100 universe (NSE symbols with .NS suffix) |
| `equity_curve.png` | Output chart (generated on run) |
| `STRATEGY.md` | This document |
| `101 alphas.pdf` | WorldQuant alpha reference (Kakushadze & Tulchinsky, 2015) |
| `4 factor overnight returns.pdf` | 4-Factor model source paper |
| `Combined.txt` | Reference implementation notes |
| `Core Logic - 4 Factor.txt` | 4-Factor derivation notes |
| `Momentum alphas - Filter Logic.txt` | Alpha selection rationale |
