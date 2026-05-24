"""
Combined Strategy: Momentum Ensemble (D-1) + Kakushadze 4-Factor (D)

CLI Usage:
    # Phase 1 — run after market close (D-1 evening)
    python combined_strategy.py phase1
    python combined_strategy.py phase1 --date 2026-05-22

    # Phase 2 — run at 09:15 on trade day D
    python combined_strategy.py phase2
    python combined_strategy.py phase2 --date 2026-05-23
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, rankdata

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

SIGNALS_FILE = "momentum_signals.csv"

# ─────────────────────────────────────────────
#  SHARED SETTINGS
# ─────────────────────────────────────────────
D          = 21
CAPITAL    = 10_00_000
TOP_N      = 2
SLIPPAGE   = 0.0005
STOP_LOSS  = 0.01
LOOKBACK   = 200
ADV_PERIOD = 20

# Round-trip cost rate (constant — turnover cancels in the derivation)
_BASE_RATE = 0.0000297 + 0.000001 + 0.000001
COST_RATE       = 2 * _BASE_RATE * (1 + 0.18) + 0.00003 + 0.00025  # ≈ 0.0355%
ENTRY_916_SLIP  = SLIPPAGE   # extra slippage buffer for 9:16 execution (vs 9:15 open)

stocks_raw = pd.read_csv("nifty100.csv")['Symbol'].dropna().tolist()
STOCKS_4F  = stocks_raw
STOCKS_MOM = sorted(list(set(
    [s if s.endswith(".NS") else f"{s}.NS" for s in stocks_raw]
)))


# ─────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────
def normalize_residuals(residuals):
    n, mu, sig = len(residuals), residuals.mean(), residuals.std()
    return mu + sig * norm.ppf((rankdata(residuals) - 0.5) / n)


def cross_normalize(series):
    sig = series.std()
    return pd.Series(
        sig * norm.ppf((rankdata(series.values) - 0.5) / len(series)),
        index=series.index,
    )


def compute_factors(df, idx):
    t, prev = df.iloc[idx], df.iloc[idx - 1]
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
    return dict(
        prc=prc, mom=mom, hlv=hlv, vol=np.log(avg_vol),
        overnight=np.log(t['Open'] / prev['Close']),
        open=t['Open'], close=t['Close'],
        high=t['High'], low=t['Low'],
    )


def execute_trade(row):
    op, cl, hi, lo = row['open'], row['close'], row['high'], row['low']
    if row['side'] == "BUY":
        entry  = op * (1 + SLIPPAGE)
        exit_p = (entry * (1 - STOP_LOSS) * (1 - SLIPPAGE)
                  if lo <= entry * (1 - STOP_LOSS) else cl * (1 - SLIPPAGE))
        ret = (exit_p - entry) / entry
    else:
        entry  = op * (1 - SLIPPAGE)
        exit_p = (entry * (1 + STOP_LOSS) * (1 + SLIPPAGE)
                  if hi >= entry * (1 + STOP_LOSS) else cl * (1 + SLIPPAGE))
        ret = (entry - exit_p) / entry
    return pd.Series({'entry': entry, 'exit': exit_p, 'ret': ret})


def prev_business_day(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    prev = dt - pd.tseries.offsets.BDay(1)
    return prev.strftime("%Y-%m-%d")


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
    return int(np.sign(d7)) if df["Volume"].iloc[-1] / df["adv20"].iloc[-1] < 1 else 0

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
    return int(np.sign(
        df["Close"].rolling(5).mean().iloc[-1] -
        df["Close"].rolling(20).mean().iloc[-1]
    ))

def alpha56(df):
    if len(df) < 30: return 0
    sr = df["returns"].tail(10).sum()
    lr = df["returns"].tail(30).sum()
    return int(np.sign(sr / (abs(lr) + 1e-6)))

def compute_momentum_signal(ticker, df):
    sigs  = dict(a101=alpha101(df), a7=alpha7(df),   a19=alpha19(df),
                 a39=alpha39(df),   a83=alpha83(df), a35=alpha35(df),
                 a71=alpha71(df),   a56=alpha56(df))
    score  = sum(sigs.values())
    votes  = list(sigs.values())
    conf   = round(max(votes.count(1), votes.count(-1)) / len(votes) * 100, 1)
    signal = "BUY" if score > 0 else ("SELL" if score < 0 else "NEUTRAL")
    return signal, score, conf, sigs


# ═════════════════════════════════════════════
#  PHASE 1 — Evening: compute & save momentum signals
# ═════════════════════════════════════════════
def phase1(signal_date: str = None, verbose: bool = True) -> dict:
    """
    Run after market close on D-1.
    Returns dict with signals_df and summary for UI consumption.
    Saves momentum_signals.csv as handoff to Phase 2.
    """
    if signal_date is None:
        signal_date = datetime.today().strftime("%Y-%m-%d")

    if verbose:
        print(f"\n{'█'*60}")
        print(f"  PHASE 1 — MOMENTUM SIGNAL ENGINE")
        print(f"  Signal date : {signal_date}  (D-1 close data)")
        print(f"{'█'*60}\n")

    end   = datetime.strptime(signal_date, "%Y-%m-%d")
    start = end - timedelta(days=LOOKBACK * 3)

    if verbose:
        print("  Downloading momentum data …")

    raw = yf.download(
        STOCKS_MOM,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )

    records = []
    for tkr in STOCKS_MOM:
        try:
            df = raw[tkr].copy() if len(STOCKS_MOM) > 1 else raw.copy()
            df.dropna(inplace=True)
            df = df[df.index <= signal_date].tail(LOOKBACK)
            if len(df) < 100:
                continue
            df["returns"] = df["Close"].pct_change()
            df["adv20"]   = (df["Close"] * df["Volume"]).rolling(ADV_PERIOD).mean()
            signal, score, conf, alpha_votes = compute_momentum_signal(tkr, df)
            base = tkr.replace(".NS", "")
            records.append({
                "ticker":      base,
                "signal":      signal,
                "score":       score,
                "confidence":  conf,
                "signal_date": signal_date,
                **{f"a{k}": v for k, v in alpha_votes.items()},
            })
        except Exception:
            pass

    signals_df = pd.DataFrame(records)

    # Save handoff file for Phase 2
    cols_to_save = ["ticker", "signal", "score", "confidence", "signal_date"]
    signals_df[cols_to_save].to_csv(SIGNALS_FILE, index=False)

    buy_ct  = (signals_df['signal'] == 'BUY').sum()
    sell_ct = (signals_df['signal'] == 'SELL').sum()
    neu_ct  = (signals_df['signal'] == 'NEUTRAL').sum()

    if verbose:
        print(f"\n  {'Ticker':<18} {'Signal':<8} {'Score':>5} {'Conf':>7}")
        print(f"  {'─'*18} {'─'*7} {'─'*5} {'─'*7}")
        for _, r in signals_df.iterrows():
            print(f"  {r['ticker']:<18} {r['signal']:<8} {int(r['score']):>5} {r['confidence']:>6.1f}%")
        print(f"\n  Stocks : {len(signals_df)}  |  BUY: {buy_ct}  SELL: {sell_ct}  NEUTRAL: {neu_ct}")
        print(f"  Signals saved → {SIGNALS_FILE}")
        print(f"  Run Phase 2 at 09:15 tomorrow:  python combined_strategy.py phase2\n")

    return {
        "signal_date": signal_date,
        "signals_df":  signals_df,
        "summary": {
            "total":   len(signals_df),
            "buy":     int(buy_ct),
            "sell":    int(sell_ct),
            "neutral": int(neu_ct),
        },
    }


# ═════════════════════════════════════════════
#  PHASE 2 — 09:15 AM: load signals, run 4-Factor, confirm trades
# ═════════════════════════════════════════════
def phase2(trade_date: str = None, verbose: bool = True,
           capital: float = None, top_n: int = None,
           x_bearer_token: str = None,
           block_results_day: bool = True,
           block_bad_news: bool = True,
           block_bad_x: bool = True) -> dict | None:
    """
    Run at 09:15 AM on trade day D.
    Entry price uses 9:16 buffer (SLIPPAGE * 2 from open) for realistic execution.
    Applies sentiment + results-day filters before finalising trade list.
    Reads momentum_signals.csv saved by Phase 1.
    """
    cap   = capital or CAPITAL
    top_n = top_n   or TOP_N
    if trade_date is None:
        trade_date = datetime.today().strftime("%Y-%m-%d")

    if verbose:
        print(f"\n{'█'*60}")
        print(f"  PHASE 2 — 4-FACTOR CONFIRMATION")
        print(f"  Trade date  : {trade_date}  (D execution)")
        print(f"{'█'*60}\n")

    # ── Load momentum signals ──────────────────────────────────────────
    try:
        mom_df = pd.read_csv(SIGNALS_FILE)
    except FileNotFoundError:
        msg = f"  ERROR: {SIGNALS_FILE} not found. Run Phase 1 first."
        if verbose: print(msg)
        return {"error": msg}

    signal_date = mom_df['signal_date'].iloc[0]

    # ── Staleness check: signal_date must be the prior business day ────
    expected_d1 = prev_business_day(trade_date)
    if signal_date != expected_d1:
        msg = (f"  WARNING: Signals are from {signal_date}, "
               f"expected {expected_d1} (prior business day to {trade_date}). "
               f"Signals may be stale — re-run Phase 1 before trading.")
        if verbose: print(msg)
        stale_warning = msg
    else:
        stale_warning = None

    if verbose:
        print(f"  Momentum signals loaded : {SIGNALS_FILE}  (date: {signal_date})")
        if stale_warning:
            print(f"\n  ⚠  {stale_warning}\n")

    momentum_signals = dict(zip(mom_df['ticker'], mom_df['signal']))

    # ── Download 4-Factor data ─────────────────────────────────────────
    ts = pd.to_datetime(trade_date)
    # 90 calendar days covers D+2 buffer for compute_factors
    dl_start = (ts - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    dl_end   = (ts + pd.tseries.offsets.BDay(2)).strftime("%Y-%m-%d")

    if verbose:
        print(f"  Downloading 4-Factor data …")

    data = yf.download(
        STOCKS_4F,
        start=dl_start, end=dl_end,
        group_by='ticker', auto_adjust=False,
        progress=False, threads=True,
    )

    # ── Check if today's data is available (09:15 timing issue) ───────
    available_dates = data.index if hasattr(data.index, 'date') else pd.DatetimeIndex([])
    today_available = ts.date() in [d.date() for d in available_dates]
    if not today_available and verbose:
        print(f"\n  ℹ  Today ({trade_date}) not yet in yfinance — "
              f"using most recent available data. Overnight signal is D-2→D-1.")

    # ── Compute factors ────────────────────────────────────────────────
    rows = []
    for stock in STOCKS_4F:
        try:
            df   = data[stock].dropna() if len(STOCKS_4F) > 1 else data.copy().dropna()
            df   = df[df.index <= ts]
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
        msg = "  Not enough stocks with valid data for 4-Factor regression."
        if verbose: print(msg)
        return {"error": msg}

    # ── Cross-sectional OLS ────────────────────────────────────────────
    df_day['hlv_n'] = cross_normalize(df_day['hlv'])
    df_day['vol_n'] = cross_normalize(df_day['vol'])

    ones = np.ones(len(df_day))
    X    = np.column_stack([ones, df_day['prc'].values, df_day['mom'].values,
                            df_day['hlv_n'].values, df_day['vol_n'].values])
    y    = df_day['overnight'].values
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"error": "OLS regression failed."}

    df_day['expected'] = X @ coeffs
    df_day['residual'] = df_day['overnight'] - df_day['expected']
    df_day['eps_norm'] = normalize_residuals(df_day['residual'])
    df_day['weight']   = -df_day['eps_norm'] / df_day['eps_norm'].abs().sum()

    # ── Select candidates ──────────────────────────────────────────────
    df_day   = df_day.sort_values('residual', ascending=False)
    sell_df  = df_day.head(top_n).copy(); sell_df['side'] = "SELL"
    buy_df   = df_day.tail(top_n).copy(); buy_df['side']  = "BUY"
    candidates = pd.concat([sell_df, buy_df])

    # ── Dual-agreement filter ──────────────────────────────────────────
    signal_rows = []
    confirmed   = []

    for _, row in candidates.iterrows():
        four_f  = row['side']
        base    = row['stock'].replace(".NS", "")
        mom_sig = momentum_signals.get(base, "NEUTRAL")
        final   = four_f if mom_sig == four_f else "SKIP"
        signal_rows.append({
            "stock":    row['stock'],
            "4factor":  four_f,
            "momentum": mom_sig,
            "final":    final,
            "residual": round(row['residual'], 6),
            "eps_norm": round(row['eps_norm'], 6),
        })
        if final != "SKIP":
            confirmed.append(row)

    signal_table_df = pd.DataFrame(signal_rows)

    if verbose:
        print(f"\n  {'Stock':<15} {'4-Factor':<12} {'Momentum':<12} {'Final':<10}")
        print(f"  {'─'*15} {'─'*11} {'─'*11} {'─'*9}")
        for r in signal_rows:
            print(f"  {r['stock']:<15} {r['4factor']:<12} {r['momentum']:<12} {r['final']:<10}")

    if not confirmed:
        if verbose:
            print("\n  No confirmed trades after momentum filter.")
        return {
            "trade_date":      trade_date,
            "signal_date":     signal_date,
            "stale_warning":   stale_warning,
            "today_available": today_available,
            "signal_table":    signal_table_df,
            "confirmed":       None,
            "filter_results":  {},
            "pnl":             None,
        }

    # ── Sentiment + results-day filter ────────────────────────────────
    from sentiment import run_filters
    filtered    = []
    filter_log  = {}
    filter_results = {}

    if verbose:
        print(f"\n  Running sentiment & results filters…")

    for _, row in pd.DataFrame(confirmed).iterrows():
        base = row['stock'].replace(".NS", "")
        fr   = run_filters(
            base, row['side'], trade_date,
            x_bearer_token    = x_bearer_token,
            block_results_day = block_results_day,
            block_bad_news    = block_bad_news,
            block_bad_x       = block_bad_x,
        )
        filter_results[base] = fr
        status = "PASS" if fr["pass"] else f"BLOCKED — {fr['reason']}"
        if verbose:
            print(f"    {base:<18}  {row['side']:<5}  {status}")
        if fr["pass"]:
            filtered.append(row)

    confirmed = filtered

    if not confirmed:
        if verbose:
            print("\n  All trades blocked by sentiment/results filters.")
        return {
            "trade_date":      trade_date,
            "signal_date":     signal_date,
            "stale_warning":   stale_warning,
            "today_available": today_available,
            "signal_table":    signal_table_df,
            "confirmed":       None,
            "filter_results":  filter_results,
            "pnl":             None,
        }

    # ── Build confirmed trades ─────────────────────────────────────────
    trades = pd.DataFrame(confirmed)
    n_confirmed = len(trades)
    if n_confirmed < top_n * 2 and verbose:
        print(f"\n  [filter] {n_confirmed}/{top_n*2} slots after all filters — partial deployment")

    trades['position'] = cap / top_n

    # 9:16 entry: open price + SLIPPAGE (base) + ENTRY_916_SLIP (execution buffer)
    total_slip = SLIPPAGE + ENTRY_916_SLIP

    def entry_stop(row):
        if row['side'] == "BUY":
            entry = row['open'] * (1 + total_slip)
            stop  = entry * (1 - STOP_LOSS)
        else:
            entry = row['open'] * (1 - total_slip)
            stop  = entry * (1 + STOP_LOSS)
        return pd.Series({'entry': entry, 'stop': stop})

    trades[['entry', 'stop']] = trades.apply(entry_stop, axis=1)

    # Simulate P&L if close is available (historical/backtest mode)
    if today_available:
        trades[['entry', 'exit', 'ret']] = trades.apply(execute_trade, axis=1)
        trades['cost_rate'] = COST_RATE
        trades['ret_net']   = trades['ret'] - trades['cost_rate']
        trades['pnl']       = trades['position'] * trades['ret_net']

        long_pnl  = trades.loc[trades.side == "BUY",  'pnl'].sum()
        short_pnl = trades.loc[trades.side == "SELL", 'pnl'].sum()
        total_pnl = trades['pnl'].sum()
        pnl_summary = {
            "long_pnl":  long_pnl,
            "short_pnl": short_pnl,
            "total_pnl": total_pnl,
            "deployed":  2 * cap,
            "win_rate":  (trades['ret_net'] > 0).mean() * 100,
            "avg_ret":   trades['ret_net'].mean() * 100,
        }
        if verbose:
            print(f"\n  Long  P&L : ₹{long_pnl:>10,.0f}  ({long_pnl/cap*100:+.2f}%)")
            print(f"  Short P&L : ₹{short_pnl:>10,.0f}  ({short_pnl/cap*100:+.2f}%)")
            print(f"  Total P&L : ₹{total_pnl:>10,.0f}  ({total_pnl/(2*cap)*100:+.2f}%)")
    else:
        pnl_summary = None
        if verbose:
            print(f"\n  Confirmed trades — entry orders for 09:15:")
            print(f"  {'Stock':<15} {'Side':<6} {'Entry':>10} {'Stop':>10} {'Position':>12}")
            print(f"  {'─'*15} {'─'*5} {'─'*10} {'─'*10} {'─'*12}")
            for _, r in trades.iterrows():
                print(f"  {r['stock']:<15} {r['side']:<6}"
                      f"  ₹{r['entry']:>8,.2f}  ₹{r['stop']:>8,.2f}"
                      f"  ₹{r['position']:>10,.0f}")

    return {
        "trade_date":      trade_date,
        "signal_date":     signal_date,
        "stale_warning":   stale_warning,
        "today_available": today_available,
        "signal_table":    signal_table_df,
        "confirmed":       trades,
        "filter_results":  filter_results,
        "pnl":             pnl_summary,
    }


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Dual-Agreement Strategy — Phase 1 (evening) / Phase 2 (09:15)"
    )
    sub = p.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("phase1", help="Evening: compute momentum signals (D-1)")
    p1.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                    help="Signal date (default: today)")

    p2 = sub.add_parser("phase2", help="09:15 AM: 4-Factor confirmation + trade list (D)")
    p2.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                    help="Trade date (default: today)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "phase1":
        phase1(signal_date=args.date)
    elif args.phase == "phase2":
        phase2(trade_date=args.date)
