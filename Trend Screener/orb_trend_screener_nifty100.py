"""
ORB Trend Quality Screener — Nifty 100 Pre-Session Filter
==========================================================
Ranks all Nifty 100 securities by how "trendy" each has been recently.
Run before market open to select ORB candidates for the day.

OUTPUT: Ranked table printed to console + saved as CSV in the same folder.

HOW TO USE:
  - Update TARGET_DATE to today's date each morning
  - Adjust LOOKBACK_DAYS, weights, TOP_N as needed
  - Run from IDLE; no command-line arguments needed
"""

import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Force UTF-8 output so Unicode characters print correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit here only
# ─────────────────────────────────────────────────────────────────────────────

# Date config — yfinance end is exclusive, so data fetched UP TO (not including) TARGET_DATE + 1
# Update this to today's date each morning before running
TARGET_DATE = "2026-05-29"   # Format: YYYY-MM-DD

WATCHLIST = [
    "ABB.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
    "ADANIPOWER.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "DMART.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS",
    "BANKBARODA.NS", "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BOSCHLTD.NS",
    "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS",
    "COALINDIA.NS", "CUMMINSIND.NS", "DLF.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "ETERNAL.NS", "GAIL.NS", "GODREJCP.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HAL.NS", "HINDUNILVR.NS", "HINDZINC.NS", "HYUNDAI.NS", "ICICIBANK.NS",
    "ITC.NS", "INDHOTEL.NS", "IOC.NS", "IRFC.NS", "INFY.NS",
    "INDIGO.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "JIOFIN.NS", "KOTAKBANK.NS",
    "LTM.NS", "LT.NS", "LODHA.NS", "M&M.NS", "MARUTI.NS",
    "MAXHEALTH.NS", "MAZDOCK.NS", "MUTHOOTFIN.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "PIDILITIND.NS", "PFC.NS", "POWERGRID.NS", "PNB.NS",
    "RECLTD.NS", "RELIANCE.NS", "SBILIFE.NS", "MOTHERSON.NS", "SHREECEM.NS",
    "SHRIRAMFIN.NS", "ENRIN.NS", "SIEMENS.NS", "SOLARINDS.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TVSMOTOR.NS", "TATACAP.NS", "TCS.NS", "TATACONSUM.NS",
    "TMCV.NS", "TMPV.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "ULTRACEMCO.NS", "UNIONBANK.NS",
    "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "WIPRO.NS", "ZYDUSLIFE.NS",
]

LOOKBACK_DAYS     = 60      # Calendar days of data to pull (~42 trading sessions)
ER_PERIOD         = 10      # Efficiency Ratio: net-move window (trading bars)
ADX_PERIOD        = 14      # ADX smoothing period
REGRESSION_PERIOD = 10      # R² linear regression lookback (trading bars)
MIN_TRADING_BARS  = 8       # Drop ticker if fewer bars available (bad data guard)

# Composite score weights — must sum to 1.0
WEIGHTS = {
    "efficiency_ratio": 0.45,   # Directly measures trend cleanliness
    "adx_norm":         0.35,   # Directional strength (normalized 0–1)
    "r_squared":        0.20,   # Price linearity
}

TOP_N    = 15       # How many tickers to flag as "ORB Ready" (more tickers = raise this)
SAVE_CSV = True
CSV_PATH = "orb_screener_nifty100_results.csv"

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def efficiency_ratio(close, period):
    """
    Kaufman's Efficiency Ratio over the last `period` bars.
    = |close[-1] - close[-period]| / sum(|daily bar changes|)
    Range: 0 (random/choppy) to 1 (perfect clean trend).
    """
    if len(close) < period + 1:
        return np.nan
    window = close.iloc[-(period + 1):]
    net_move = abs(window.iloc[-1] - window.iloc[0])
    path_length = window.diff().abs().sum()
    if path_length == 0:
        return np.nan
    return net_move / path_length


def compute_adx(high, low, close, period):
    """
    True ADX (Wilder smoothed). Returns the most recent ADX value.
    Minimum bars required: period * 2 + 1
    """
    # Need enough bars so valid_dx has at least `period` entries for ADX seed
    needed = period * 2 + 1
    if len(close) < needed:
        return np.nan

    h, l, c = high.values, low.values, close.values

    tr_list, plus_dm, minus_dm = [], [], []
    for i in range(1, len(c)):
        tr_list.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
        up   = h[i] - h[i-1]
        down = l[i-1] - l[i]
        plus_dm.append(up   if (up > down and up > 0)   else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

    def wilder(arr, n):
        result = [np.nan] * len(arr)
        result[n-1] = sum(arr[:n])
        for i in range(n, len(arr)):
            result[i] = result[i-1] - (result[i-1] / n) + arr[i]
        return result

    sm_tr    = wilder(tr_list, period)
    sm_plus  = wilder(plus_dm, period)
    sm_minus = wilder(minus_dm, period)

    dx_list = []
    for i in range(len(sm_tr)):
        if sm_tr[i] and not np.isnan(sm_tr[i]) and sm_tr[i] > 0:
            dip  = 100 * sm_plus[i]  / sm_tr[i]
            dim  = 100 * sm_minus[i] / sm_tr[i]
            diff = abs(dip - dim)
            summ = dip + dim
            dx_list.append(100 * diff / summ if summ > 0 else 0)
        else:
            dx_list.append(np.nan)

    valid_dx = [x for x in dx_list if not np.isnan(x)]
    if len(valid_dx) < period:
        return np.nan

    # FIX: seed ADX as the AVERAGE of the first `period` DX values (divide by period)
    adx_val = sum(valid_dx[:period]) / period
    for i in range(period, len(valid_dx)):
        adx_val = (adx_val * (period - 1) + valid_dx[i]) / period
    return adx_val


def r_squared_regression(close, period):
    """
    R² of closing prices vs OLS straight line over last `period` bars.
    Close to 1 → trending cleanly. Close to 0 → choppy/random.
    """
    if len(close) < period:
        return np.nan
    y = close.iloc[-period:].values
    x = np.arange(len(y))
    x_mean, y_mean = x.mean(), y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return np.nan
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    y_pred = slope * x + (y_mean - slope * x_mean)
    ss_res = np.sum((y - y_pred) ** 2)
    return 1 - ss_res / ss_tot


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH + SCREENING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(target_date=None):
    _td = target_date if target_date else TARGET_DATE
    end_date   = datetime.strptime(_td, "%Y-%m-%d") + timedelta(days=1)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS + 1)

    print(f"\n{'='*68}")
    print(f"  ORB TREND QUALITY SCREENER  |  Signal Date: {_td}")
    print(f"  Universe: {len(WATCHLIST)} tickers (Nifty 100)  |  Lookback: {LOOKBACK_DAYS} calendar days")
    print(f"{'='*68}\n")
    print("Fetching data for all 100 tickers (this takes ~20 seconds)...", flush=True)

    raw = yf.download(
        tickers=WATCHLIST,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    print("Done. Running metrics...\n")

    results = []
    skipped = []

    for ticker in WATCHLIST:
        try:
            df = raw[ticker].copy() if len(WATCHLIST) > 1 else raw.copy()
            df.dropna(subset=["Close", "High", "Low"], inplace=True)

            if len(df) < MIN_TRADING_BARS:
                skipped.append(ticker)
                continue

            close = df["Close"]
            high  = df["High"]
            low   = df["Low"]

            er      = efficiency_ratio(close, ER_PERIOD)
            adx_raw = compute_adx(high, low, close, ADX_PERIOD)
            r2      = r_squared_regression(close, REGRESSION_PERIOD)

            # Normalize ADX to 0–1 (cap at 60; anything above = max trending)
            adx_norm = min(adx_raw / 60.0, 1.0) if not np.isnan(adx_raw) else np.nan

            components = {"efficiency_ratio": er, "adx_norm": adx_norm, "r_squared": r2}
            valid = {k: v for k, v in components.items() if not np.isnan(v)}

            if not valid:
                score = np.nan
            else:
                weight_sum = sum(WEIGHTS[k] for k in valid)
                score = sum(WEIGHTS[k] * valid[k] for k in valid) / weight_sum

            results.append({
                "Ticker":     ticker.replace(".NS", ""),
                "Score":      round(score, 4)   if not np.isnan(score)   else np.nan,
                "Eff.Ratio":  round(er, 4)      if not np.isnan(er)      else np.nan,
                "ADX":        round(adx_raw, 2) if not np.isnan(adx_raw) else np.nan,
                "R²":         round(r2, 4)      if not np.isnan(r2)      else np.nan,
                "Bars":       len(df),
            })

        except Exception as e:
            skipped.append(ticker)

    if not results:
        print("No results. Check internet connection or yfinance availability.")
        return

    df_out = (
        pd.DataFrame(results)
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )
    df_out.insert(0, "Rank", df_out.index + 1)

    # ── Print ranked table ──────────────────────────────────────────────────
    hdr = f"{'Rank':<5} {'Ticker':<14} {'Score':>7}  {'Eff.Ratio':>9}  {'ADX':>6}  {'R²':>7}  {'Bars':>5}  {'':10}"
    print(hdr)
    print("-" * 72)

    def fmt(val, decimals=4):
        return f"{val:.{decimals}f}" if not (isinstance(val, float) and np.isnan(val)) else "  N/A"

    for _, row in df_out.iterrows():
        flag = "<<< ORB READY" if row["Rank"] <= TOP_N else ""
        print(
            f"{int(row['Rank']):<5} "
            f"{row['Ticker']:<14} "
            f"{fmt(row['Score']):>7}  "
            f"{fmt(row['Eff.Ratio']):>9}  "
            f"{fmt(row['ADX'], 1):>6}  "
            f"{fmt(row['R²']):>7}  "
            f"{int(row['Bars']):>5}  "
            f"{flag}"
        )

    print()

    # ── Score guide ─────────────────────────────────────────────────────────
    print("SCORE GUIDE:")
    print("  > 0.60     ->  Strong trend regime       (high ORB confidence)")
    print("  0.40-0.60  ->  Moderate trend            (use with caution)")
    print("  < 0.40     ->  Choppy / mean-reverting   (skip for ORB)")
    print()
    print("INDIVIDUAL METRICS:")
    print("  Eff. Ratio > 0.50  ->  Clean directional move, low noise")
    print("  ADX > 25           ->  Directional trend confirmed")
    print("  R^2 > 0.75         ->  Price tracking a linear slope well")
    print()

    # ── Top picks summary ───────────────────────────────────────────────────
    top = df_out[df_out["Rank"] <= TOP_N]["Ticker"].tolist()
    print(f"TOP {TOP_N} ORB CANDIDATES FOR {_td}:")
    print("  " + ", ".join(top))
    print()

    if skipped:
        print(f"SKIPPED ({len(skipped)} tickers — insufficient data or fetch error):")
        print("  " + ", ".join([t.replace(".NS", "") for t in skipped]))
        print()

    # ── Save CSV ────────────────────────────────────────────────────────────
    if SAVE_CSV:
        df_out.to_csv(CSV_PATH, index=False)
        print(f"Full results saved → {CSV_PATH}")

    print(f"{'='*68}\n")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_screener()
