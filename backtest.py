"""
Backtest Engine — Combined Strategy
Momentum Ensemble (D-1) + Kakushadze 4-Factor (D)

Usage:
    python backtest.py
    python backtest.py --start 2024-01-01 --end 2024-12-31
    python backtest.py --start 2025-01-01 --end 2025-06-30 --capital 500000 --top-n 3

Outputs:
  - Per-day P&L table (console)
  - Multi-day aggregate stats (console)
  - equity_curve.png
"""

import argparse
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, rankdata
from datetime import timedelta

# Suppress noisy yfinance / pandas FutureWarnings only
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ═════════════════════════════════════════════
#  BACKTEST CONFIG  —  defaults (override via CLI)
# ═════════════════════════════════════════════
DEFAULT_START   = "2025-01-01"
DEFAULT_END     = "2025-06-30"
DEFAULT_CAPITAL = 10_00_000   # ₹ per leg
DEFAULT_TOP_N   = 2

# ═════════════════════════════════════════════
#  STRATEGY SETTINGS
# ═════════════════════════════════════════════
D          = 21
SLIPPAGE        = 0.0005
ENTRY_916_SLIP  = 0.0005  # extra buffer for 9:16 execution vs 9:15 open price
STOP_LOSS       = 0.01
LOOKBACK   = 200
ADV_PERIOD = 20

# Round-trip cost rate as a fraction of notional (constant — turnover cancels out).
# Entry leg: exchange + SEBI + IPFT + GST on above + stamp duty
# Exit  leg: exchange + SEBI + IPFT + GST on above + STT
_BASE_RATE = 0.0000297 + 0.000001 + 0.000001          # exchange + SEBI + IPFT
COST_RATE  = 2 * _BASE_RATE * (1 + 0.18) + 0.00003 + 0.00025  # ≈ 0.0355%

stocks_raw = pd.read_csv("nifty100.csv")['Symbol'].dropna().tolist()
STOCKS_4F  = stocks_raw
STOCKS_MOM = sorted(list(set(
    [s if s.endswith(".NS") else f"{s}.NS" for s in stocks_raw]
)))


# ─────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────
def normalize_residuals(residuals):
    n     = len(residuals)
    mu    = residuals.mean()
    sig   = residuals.std()
    ranks = rankdata(residuals)
    return mu + sig * norm.ppf((ranks - 0.5) / n)


def cross_normalize(series):
    sig   = series.std()
    ranks = rankdata(series.values)
    return pd.Series(sig * norm.ppf((ranks - 0.5) / len(series)),
                     index=series.index)


def compute_factors(df, idx):
    t    = df.iloc[idx]
    prev = df.iloc[idx - 1]
    hist = df.iloc[max(0, idx - D) : idx]
    if len(hist) < D // 2:
        return None
    prc     = np.log(prev.get('Adj Close', prev['Close']))
    mom     = np.log(prev['Close'] / prev['Open'])
    hl_sq   = ((hist['High'] - hist['Low']) / hist['Close']) ** 2
    hlv     = 0.5 * np.log(hl_sq.mean())
    avg_vol = hist['Volume'].mean()
    if avg_vol <= 0:
        return None
    vol       = np.log(avg_vol)
    overnight = np.log(t['Open'] / prev['Close'])
    return dict(prc=prc, mom=mom, hlv=hlv, vol=vol,
                overnight=overnight,
                open=t['Open'], close=t['Close'],
                high=t['High'], low=t['Low'])


def execute_trade(row):
    op, cl, hi, lo = row['open'], row['close'], row['high'], row['low']
    # Entry at 9:16 AM: open + base slippage + 1-minute execution buffer
    total_slip = SLIPPAGE + ENTRY_916_SLIP
    if row['side'] == "BUY":
        entry  = op * (1 + total_slip)
        exit_p = (entry * (1 - STOP_LOSS) * (1 - SLIPPAGE)
                  if lo <= entry * (1 - STOP_LOSS) else cl * (1 - SLIPPAGE))
        ret = (exit_p - entry) / entry
    else:
        entry  = op * (1 - total_slip)
        exit_p = (entry * (1 + STOP_LOSS) * (1 + SLIPPAGE)
                  if hi >= entry * (1 + STOP_LOSS) else cl * (1 + SLIPPAGE))
        ret = (entry - exit_p) / entry
    return pd.Series({'entry': entry, 'exit': exit_p, 'ret': ret})


# ─────────────────────────────────────────────
#  MOMENTUM ALPHAS
# ─────────────────────────────────────────────
def alpha101(df):
    r = df.iloc[-1]
    return int(np.sign((r["Close"] - r["Open"]) / (r["High"] - r["Low"] + 0.001)))

def alpha7(df):
    if len(df) < 60: return 0
    d7 = df["Close"].diff(7).iloc[-1]
    return int(np.sign(d7)) if df["Volume"].iloc[-1] > df["adv20"].iloc[-1] else 0

def alpha19(df):
    if len(df) < 10: return 0
    return int(np.sign(df["Close"].iloc[-1] - df["Close"].iloc[-7]))

def alpha39(df):
    if len(df) < 20: return 0
    d7 = df["Close"].diff(7).iloc[-1]
    rv = df["Volume"].iloc[-1] / df["adv20"].iloc[-1]
    return int(np.sign(d7)) if rv < 1 else 0

def alpha83(df):
    if len(df) < 10: return 0
    r  = df.iloc[-1]
    rr = (r["High"] - r["Low"]) / df["Close"].rolling(5).mean().iloc[-1]
    return int(np.sign(r["Close"] - r["Open"])) if rr > 0.03 else 0

def alpha35(df):
    if len(df) < 35: return 0
    rk = df["returns"].tail(32).rank().iloc[-1]
    return 1 if rk > 24 else (-1 if rk < 8 else 0)

def alpha71(df):
    if len(df) < 20: return 0
    return int(np.sign(df["Close"].rolling(5).mean().iloc[-1] -
                       df["Close"].rolling(20).mean().iloc[-1]))

def alpha56(df):
    if len(df) < 30: return 0
    sr = df["returns"].tail(10).sum()
    lr = df["returns"].tail(30).sum()
    return int(np.sign(sr / (abs(lr) + 1e-6)))

def compute_momentum_signal(ticker, df):
    sigs  = dict(a101=alpha101(df), a7=alpha7(df),   a19=alpha19(df),
                 a39=alpha39(df),   a83=alpha83(df), a35=alpha35(df),
                 a71=alpha71(df),   a56=alpha56(df))
    score = sum(sigs.values())
    return "BUY" if score > 0 else ("SELL" if score < 0 else "NEUTRAL")


# ─────────────────────────────────────────────
#  DATA PREFETCH  (download once for the whole window)
# ─────────────────────────────────────────────
def prefetch_all(start: str, end: str):
    bt_start = pd.to_datetime(start)
    bt_end   = pd.to_datetime(end)

    buffer_days = max(D * 2, LOOKBACK * 3)
    dl_start    = (bt_start - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    # Use 5 business days buffer so weekends/holidays don't clip the window
    dl_end      = (bt_end + pd.tseries.offsets.BDay(5)).strftime("%Y-%m-%d")

    print(f"  Downloading 4-Factor data  ({dl_start} → {dl_end}) …")
    raw_4f = yf.download(
        STOCKS_4F, start=dl_start, end=dl_end,
        group_by='ticker', auto_adjust=False, progress=False
    )

    print(f"  Downloading Momentum data  ({dl_start} → {dl_end}) …")
    raw_mom = yf.download(
        STOCKS_MOM, start=dl_start, end=dl_end,
        group_by='ticker', auto_adjust=True, progress=False
    )

    return raw_4f, raw_mom


# ─────────────────────────────────────────────
#  PER-DAY SIGNAL ENGINE  (uses prefetched data)
# ─────────────────────────────────────────────
def get_momentum_signals(raw_mom, d_minus_1: str) -> dict:
    d1_ts   = pd.to_datetime(d_minus_1)
    signals = {}
    for tkr in STOCKS_MOM:
        try:
            df = raw_mom[tkr].dropna() if len(STOCKS_MOM) > 1 else raw_mom.copy().dropna()
            df = df[df.index <= d1_ts].tail(LOOKBACK)
            if len(df) < 100:
                continue
            df = df.copy()
            df["returns"] = df["Close"].pct_change()
            df["adv20"]   = (df["Close"] * df["Volume"]).rolling(ADV_PERIOD).mean()
            sig  = compute_momentum_signal(tkr, df)
            base = tkr.replace(".NS", "")
            signals[base] = sig
        except Exception:
            pass
    return signals


def run_day(raw_4f, momentum_signals: dict, trade_date: str,
            top_n: int, capital: float):
    ts   = pd.to_datetime(trade_date)
    rows = []
    for stock in STOCKS_4F:
        try:
            df = raw_4f[stock].dropna() if len(STOCKS_4F) > 1 else raw_4f.copy().dropna()
            df = df[df.index <= ts]
            if len(df) < D + 2:
                continue
            feat = compute_factors(df, len(df) - 1)
            if feat is None:
                continue
            feat['stock'] = stock
            rows.append(feat)
        except Exception:
            continue

    df_day = pd.DataFrame(rows).dropna()
    if len(df_day) < top_n * 2 + 1:
        return None

    df_day['hlv_n'] = cross_normalize(df_day['hlv'])
    df_day['vol_n'] = cross_normalize(df_day['vol'])

    ones = np.ones(len(df_day))
    X    = np.column_stack([ones, df_day['prc'].values, df_day['mom'].values,
                            df_day['hlv_n'].values, df_day['vol_n'].values])
    y    = df_day['overnight'].values
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    df_day['expected'] = X @ coeffs
    df_day['residual'] = df_day['overnight'] - df_day['expected']
    df_day['eps_norm'] = normalize_residuals(df_day['residual'])
    df_day['weight']   = -df_day['eps_norm'] / df_day['eps_norm'].abs().sum()

    df_day    = df_day.sort_values('residual', ascending=False)
    sell_df   = df_day.head(top_n).copy(); sell_df['side'] = "SELL"
    buy_df    = df_day.tail(top_n).copy(); buy_df['side']  = "BUY"
    candidates = pd.concat([sell_df, buy_df])

    confirmed = []
    for _, row in candidates.iterrows():
        base = row['stock'].replace(".NS", "")
        if momentum_signals.get(base, "NEUTRAL") == row['side']:
            confirmed.append(row)

    if not confirmed:
        return None

    trades      = pd.DataFrame(confirmed)
    n_confirmed = len(trades)
    if n_confirmed < top_n * 2:
        print(f"    [filter] {n_confirmed}/{top_n * 2} slots confirmed — partial deployment")

    # Fixed notional per slot: under-deployment is logged above, not silently absorbed
    trades['position'] = capital / top_n
    trades[['entry', 'exit', 'ret']] = trades.apply(execute_trade, axis=1)
    trades['cost_rate'] = COST_RATE
    trades['ret_net']   = trades['ret'] - trades['cost_rate']
    trades['pnl']       = trades['position'] * trades['ret_net']
    return trades


# ─────────────────────────────────────────────
#  EQUITY CURVE PLOT
# ─────────────────────────────────────────────
def plot_equity_curve(daily_pnl: pd.Series, output_path: str = "equity_curve.png"):
    daily_pnl  = daily_pnl.sort_index()
    equity     = daily_pnl.cumsum()
    running_max = equity.cummax()
    drawdown    = equity - running_max

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        facecolor='#0d1117'
    )
    fig.suptitle(
        'Combined Strategy — Equity Curve',
        fontsize=16, fontweight='bold',
        color='#e6edf3', y=0.98, fontfamily='monospace'
    )

    ax1 = axes[0]
    ax1.set_facecolor('#0d1117')
    ax1.axhline(0, color='#30363d', linewidth=0.8, linestyle='--')
    ax1.fill_between(equity.index, equity, 0,
                     where=(equity >= 0), alpha=0.15, color='#3fb950', interpolate=True)
    ax1.fill_between(equity.index, equity, 0,
                     where=(equity < 0),  alpha=0.15, color='#f85149', interpolate=True)
    color_line = '#3fb950' if equity.iloc[-1] >= 0 else '#f85149'
    ax1.plot(equity.index, equity, color=color_line, linewidth=1.8, zorder=3)
    peak_idx   = equity.idxmax()
    trough_idx = equity.idxmin()
    ax1.scatter([peak_idx],   [equity[peak_idx]],   color='#3fb950', s=60, zorder=5)
    ax1.scatter([trough_idx], [equity[trough_idx]], color='#f85149', s=60, zorder=5)
    ax1.set_ylabel('Cumulative P&L  (₹)', color='#8b949e',
                   fontsize=10, fontfamily='monospace')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    ax1.grid(axis='y', color='#21262d', linewidth=0.6)
    ax1.set_xlim(equity.index[0], equity.index[-1])

    ax2 = axes[1]
    ax2.set_facecolor('#0d1117')
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.6, color='#f85149')
    ax2.plot(drawdown.index, drawdown, color='#f85149', linewidth=1)
    ax2.set_ylabel('Drawdown  (₹)', color='#8b949e',
                   fontsize=9, fontfamily='monospace')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#30363d')
    ax2.grid(axis='y', color='#21262d', linewidth=0.6)
    ax2.set_xlim(equity.index[0], equity.index[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\n  Equity curve saved → {output_path}")


# ─────────────────────────────────────────────
#  BACKTEST MAIN
# ─────────────────────────────────────────────
def run_backtest(start: str, end: str, capital: float, top_n: int):
    print(f"\n{'█'*60}")
    print(f"  BACKTEST ENGINE")
    print(f"  Period  : {start}  →  {end}")
    print(f"  Capital : ₹{capital:,.0f} per leg")
    print(f"  TOP_N   : {top_n} stocks per side")
    print(f"{'█'*60}\n")

    all_days = pd.bdate_range(start=start, end=end)
    print(f"  Trading days in window : {len(all_days)}\n")

    raw_4f, raw_mom = prefetch_all(start, end)

    all_trades  = []
    daily_stats = []

    for trade_date in all_days:
        d_str   = trade_date.strftime("%Y-%m-%d")
        d1_str  = (trade_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        mom_signals = get_momentum_signals(raw_mom, d1_str)
        trades      = run_day(raw_4f, mom_signals, d_str, top_n, capital)

        if trades is not None and len(trades) > 0:
            trades['date'] = trade_date
            all_trades.append(trades)
            day_pnl   = trades['pnl'].sum()
            long_pnl  = trades.loc[trades.side == "BUY",  'pnl'].sum()
            short_pnl = trades.loc[trades.side == "SELL", 'pnl'].sum()
            n_trades  = len(trades)
            win_rate  = (trades['ret_net'] > 0).mean() * 100
        else:
            day_pnl = long_pnl = short_pnl = 0.0
            n_trades = 0
            win_rate = 0.0

        daily_stats.append({
            'date'      : trade_date,
            'pnl'       : day_pnl,
            'long_pnl'  : long_pnl,
            'short_pnl' : short_pnl,
            'n_trades'  : n_trades,
            'win_rate'  : win_rate,
        })

        status = f"₹{day_pnl:>+10,.0f}" if n_trades else "  no trades"
        print(f"  {d_str}  {status}   trades={n_trades}")

    # ── Aggregate stats ──────────────────────────────────────────────────
    stats_df  = pd.DataFrame(daily_stats).set_index('date')
    daily_pnl = stats_df['pnl']

    trading_days = (daily_pnl != 0).sum()
    win_days     = (daily_pnl > 0).sum()
    loss_days    = (daily_pnl < 0).sum()
    flat_days    = (daily_pnl == 0).sum()
    cum_pnl      = daily_pnl.sum()

    # Sharpe uses ALL days (flat days contribute 0), annualised with √252
    avg_daily = daily_pnl.mean()
    std_daily = daily_pnl.std()
    sharpe    = (avg_daily / std_daily * np.sqrt(252)
                 if std_daily and std_daily > 0 else np.nan)

    max_dd    = (daily_pnl.cumsum() - daily_pnl.cumsum().cummax()).min()
    best_day  = daily_pnl.max()
    worst_day = daily_pnl.min()

    deployed      = 2 * capital
    cum_pct       = cum_pnl   / deployed * 100
    avg_daily_pct = avg_daily / deployed * 100
    std_daily_pct = std_daily / deployed * 100
    max_dd_pct    = max_dd    / deployed * 100
    best_day_pct  = best_day  / deployed * 100
    worst_day_pct = worst_day / deployed * 100

    if all_trades:
        combined   = pd.concat(all_trades)
        overall_wr = (combined['ret_net'] > 0).mean() * 100
        avg_ret    = combined['ret_net'].mean() * 100
        stats_df['pnl_pct']       = stats_df['pnl']       / deployed * 100
        stats_df['long_pnl_pct']  = stats_df['long_pnl']  / capital  * 100
        stats_df['short_pnl_pct'] = stats_df['short_pnl'] / capital  * 100
    else:
        combined   = None
        overall_wr = avg_ret = 0.0
        stats_df['pnl_pct'] = stats_df['long_pnl_pct'] = stats_df['short_pnl_pct'] = 0.0

    def fmt(rupees, pct):
        sign = '+' if pct >= 0 else ''
        return f"₹{rupees:>12,.0f}  ({sign}{pct:.2f}%)"

    print(f"\n{'═'*60}")
    print(f"  BACKTEST RESULTS  [{start}  →  {end}]")
    print(f"{'═'*60}")
    print(f"  Total calendar days   : {len(all_days)}")
    print(f"  Days with trades      : {trading_days}")
    print(f"  Days flat (no signal) : {flat_days}")
    print(f"  Winning days          : {win_days}")
    print(f"  Losing  days          : {loss_days}")
    print(f"  Deployed capital/day  : ₹{deployed:,.0f}  (long ₹{capital:,.0f} + short ₹{capital:,.0f})")
    print(f"{'─'*60}")
    print(f"  Cumulative P&L        : {fmt(cum_pnl,      cum_pct)}")
    print(f"  Avg Daily P&L         : {fmt(avg_daily,    avg_daily_pct)}")
    print(f"  Daily Std Dev         : {fmt(std_daily,    std_daily_pct)}")
    print(f"  Sharpe (ann. ×√252)   : {sharpe:>12.2f}")
    print(f"  Max Drawdown          : {fmt(max_dd,       max_dd_pct)}")
    print(f"  Best Day              : {fmt(best_day,     best_day_pct)}")
    print(f"  Worst Day             : {fmt(worst_day,    worst_day_pct)}")
    print(f"{'─'*60}")
    print(f"  Trade-level Win Rate  : {overall_wr:>11.1f}%")
    print(f"  Avg Net Return/Trade  : {avg_ret:>11.3f}%")
    print(f"{'═'*60}\n")

    print(f"  {'Date':<12} {'P&L (₹)':>12} {'P&L %':>7} {'Long %':>7} {'Short %':>8} {'Trades':>7} {'WinRate':>9}")
    print(f"  {'─'*12} {'─'*12} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*9}")
    for dt, row in stats_df.iterrows():
        if row['n_trades'] == 0:
            print(f"  {dt.strftime('%Y-%m-%d'):<12}  {'—':>11}  {'—':>6}  {'—':>6}  {'—':>7}  {'—':>6}  {'—':>8}")
        else:
            sign = '+' if row['pnl_pct']       >= 0 else ''
            ls   = '+' if row['long_pnl_pct']  >= 0 else ''
            ss   = '+' if row['short_pnl_pct'] >= 0 else ''
            print(f"  {dt.strftime('%Y-%m-%d'):<12}"
                  f"  ₹{row['pnl']:>10,.0f}"
                  f"  {sign}{row['pnl_pct']:>5.2f}%"
                  f"  {ls}{row['long_pnl_pct']:>5.2f}%"
                  f"  {ss}{row['short_pnl_pct']:>6.2f}%"
                  f"  {int(row['n_trades']):>6}"
                  f"  {row['win_rate']:>7.1f}%")

    plot_equity_curve(daily_pnl, output_path="equity_curve.png")

    return stats_df, combined


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Combined Momentum + 4-Factor Intraday Backtest (Nifty 100)"
    )
    p.add_argument("--start",   default=DEFAULT_START,   metavar="YYYY-MM-DD",
                   help=f"Backtest start date (default: {DEFAULT_START})")
    p.add_argument("--end",     default=DEFAULT_END,     metavar="YYYY-MM-DD",
                   help=f"Backtest end date   (default: {DEFAULT_END})")
    p.add_argument("--capital", default=DEFAULT_CAPITAL, type=float, metavar="N",
                   help=f"Rupee notional per leg (default: {DEFAULT_CAPITAL:,.0f})")
    p.add_argument("--top-n",   default=DEFAULT_TOP_N,   type=int,   metavar="N",
                   help=f"Stocks per side (default: {DEFAULT_TOP_N})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats, trades = run_backtest(
        start=args.start,
        end=args.end,
        capital=args.capital,
        top_n=args.top_n,
    )
