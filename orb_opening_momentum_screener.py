"""
ORB Opening Momentum Screener — NSE First-15-Min Anomaly Detector
==================================================================
Compares today's first 15 minutes of price action against each stock's
own historical first-15-min behaviour (last 60 days). Stocks showing
significantly more movement than their own normal get ranked highest.

WHEN TO RUN: After 9:30 AM IST on the day you want to trade.
TARGET_DATE: Set to today's date. The script pulls today's first 15 min
             plus 60 days of historical 5m data for comparison.

OUTPUT: Ranked table (descending by composite z-score) + CSV saved.

Logic:
  - Pull 5-min OHLCV data for all tickers
  - For each trading day, extract the first 3 bars (9:15–9:30 AM IST)
  - Compute historical distribution of 15-min range and directional move
  - Z-score today's values against that distribution
  - Composite Score = 0.55 * Range_Z + 0.45 * Direction_Z
  - Sort descending — high score = unusual opening energy = ORB candidate
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit here only
# ─────────────────────────────────────────────────────────────────────────────

# Set to TODAY's date — run this script after 9:30 AM IST
TARGET_DATE = "2026-05-27"   # Format: YYYY-MM-DD

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

OR_MINUTES        = 15       # Opening range window in minutes
BAR_SIZE_MINUTES  = 5        # yfinance interval (5m bars; 3 bars = 15 min)
LOOKBACK_DAYS     = 45       # Calendar days of 5m history (yfinance hard limit = 60 days from TODAY)
MIN_HISTORY_DAYS  = 10       # Minimum valid historical days per ticker

# Composite score weights
RANGE_Z_WEIGHT    = 0.55     # Range z-score weight (how wide vs normal)
DIRECTION_Z_WEIGHT = 0.45    # Directional move z-score weight (how strong vs normal)

TOP_N    = 15
SAVE_CSV = True
CSV_PATH = "orb_momentum_results.csv"

# NSE session open time in IST
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MINUTE = 15

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

OR_BARS = OR_MINUTES // BAR_SIZE_MINUTES   # = 3 bars for 15-min OR on 5m chart

def extract_or_stats(day_df):
    """
    Given a DataFrame of intraday bars for one day (sorted by time),
    return (or_range, or_net_move) for the first OR_BARS bars.
    or_range    = high of window - low of window  (absolute volatility)
    or_net_move = close of last bar - open of first bar (signed direction)
    Returns (nan, nan) if not enough bars.
    """
    if len(day_df) < OR_BARS:
        return np.nan, np.nan
    window = day_df.iloc[:OR_BARS]
    or_range    = window["High"].max() - window["Low"].min()
    or_net_move = window["Close"].iloc[-1] - window["Open"].iloc[0]
    return or_range, or_net_move


def get_or_data(df):
    """
    Split a multi-day 5m DataFrame into per-day groups,
    extract OR stats for each day, return as DataFrame.
    Converts index to IST if needed.
    """
    # Ensure timezone is IST
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    elif str(df.index.tz) != "Asia/Kolkata":
        df.index = df.index.tz_convert("Asia/Kolkata")

    # Filter to market open window only (9:15 to 9:15 + OR_MINUTES)
    open_time  = pd.Timestamp(f"{MARKET_OPEN_HOUR:02d}:{MARKET_OPEN_MINUTE:02d}:00")
    close_time = open_time + pd.Timedelta(minutes=OR_MINUTES)

    df = df[(df.index.time >= open_time.time()) &
            (df.index.time <  close_time.time())]

    records = []
    for date, group in df.groupby(df.index.date):
        group = group.sort_index()
        or_range, or_move = extract_or_stats(group)
        if not np.isnan(or_range):
            records.append({
                "date":     date,
                "or_range": or_range,
                "or_move":  or_move,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCREENER
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(target_date=None):
    _td        = target_date if target_date else TARGET_DATE
    target_dt  = datetime.strptime(_td, "%Y-%m-%d").date()
    end_date   = datetime.strptime(_td, "%Y-%m-%d") + timedelta(days=1)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS + 1)
    # Hard safety cap — yfinance 5m data is strictly limited to 60 calendar days from TODAY
    earliest_allowed = datetime.today() - timedelta(days=58)
    if start_date < earliest_allowed:
        start_date = earliest_allowed

    print(f"\n{'='*72}")
    print(f"  ORB OPENING MOMENTUM SCREENER  |  Trading Date: {_td}")
    print(f"  Universe: {len(WATCHLIST)} tickers  |  OR Window: first {OR_MINUTES} min  |  History: {LOOKBACK_DAYS} cal. days")
    print(f"{'='*72}\n")
    print("Fetching 5-minute data (this takes ~40 seconds for 100 tickers)...", flush=True)

    raw = yf.download(
        tickers=WATCHLIST,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="5m",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    print("Done. Computing opening range statistics...\n")

    results = []
    skipped = []

    for ticker in WATCHLIST:
        try:
            df = raw[ticker].copy() if len(WATCHLIST) > 1 else raw.copy()
            df.dropna(subset=["Close", "High", "Low", "Open"], inplace=True)

            if df.empty:
                skipped.append(ticker)
                continue

            # Extract per-day OR stats
            or_df = get_or_data(df)

            if or_df.empty:
                skipped.append(ticker)
                continue

            # Split into today vs history
            today_row = or_df[or_df["date"] == target_dt]
            hist_df   = or_df[or_df["date"] <  target_dt]

            if today_row.empty:
                skipped.append(ticker)
                continue

            if len(hist_df) < MIN_HISTORY_DAYS:
                skipped.append(ticker)
                continue

            today_range = today_row["or_range"].values[0]
            today_move  = today_row["or_move"].values[0]

            hist_range_mean = hist_df["or_range"].mean()
            hist_range_std  = hist_df["or_range"].std()
            hist_move_mean  = hist_df["or_move"].abs().mean()   # avg absolute move
            hist_move_std   = hist_df["or_move"].abs().std()

            # Z-scores
            range_z = (today_range - hist_range_mean) / hist_range_std \
                      if hist_range_std > 0 else np.nan
            move_z  = (abs(today_move) - hist_move_mean) / hist_move_std \
                      if hist_move_std > 0 else np.nan

            if np.isnan(range_z) and np.isnan(move_z):
                skipped.append(ticker)
                continue

            # Composite score — weighted, handle partial NaN
            components = {k: v for k, v in
                          [("range_z", range_z), ("move_z", move_z)]
                          if not np.isnan(v)}
            weights = {"range_z": RANGE_Z_WEIGHT, "move_z": DIRECTION_Z_WEIGHT}
            w_sum = sum(weights[k] for k in components)
            composite = sum(weights[k] * components[k] for k in components) / w_sum \
                        if components else np.nan

            # Range expansion ratio (today vs historical avg)
            ratio = today_range / hist_range_mean if hist_range_mean > 0 else np.nan

            direction = "UP  " if today_move > 0 else "DOWN"

            results.append({
                "Ticker":        ticker.replace(".NS", ""),
                "Composite Z":   round(composite, 3)        if not np.isnan(composite)     else np.nan,
                "Range Z":       round(range_z, 3)          if not np.isnan(range_z)        else np.nan,
                "Dir Z":         round(move_z, 3)           if not np.isnan(move_z)         else np.nan,
                "Range Ratio":   round(ratio, 2)            if not np.isnan(ratio)          else np.nan,
                "Today Range":   round(today_range, 2),
                "Hist Avg Range":round(hist_range_mean, 2),
                "Direction":     direction,
                "Hist Days":     len(hist_df),
            })

        except Exception as e:
            skipped.append(ticker)

    if not results:
        print("No results. Ensure the market has been open for at least 15 minutes today.")
        return

    df_out = (
        pd.DataFrame(results)
        .sort_values("Composite Z", ascending=False)
        .reset_index(drop=True)
    )
    df_out.insert(0, "Rank", df_out.index + 1)

    # ── Print ranked table ──────────────────────────────────────────────────
    print(f"{'Rank':<5} {'Ticker':<13} {'Comp Z':>7}  {'Range Z':>7}  {'Dir Z':>6}  "
          f"{'Ratio':>6}  {'Today Rng':>10}  {'Hist Avg':>9}  {'Dir':>5}  {'Days':>5}")
    print("-" * 83)

    for _, row in df_out.iterrows():
        flag = "◀" if row["Rank"] <= TOP_N else ""

        def f(val, d=2):
            return f"{val:.{d}f}" if isinstance(val, float) and not np.isnan(val) else "  N/A"

        print(
            f"{int(row['Rank']):<5} "
            f"{row['Ticker']:<13} "
            f"{f(row['Composite Z'], 3):>7}  "
            f"{f(row['Range Z'], 3):>7}  "
            f"{f(row['Dir Z'], 3):>6}  "
            f"{f(row['Range Ratio'], 2):>6}  "
            f"{f(row['Today Range'], 2):>10}  "
            f"{f(row['Hist Avg Range'], 2):>9}  "
            f"{row['Direction']:>5}  "
            f"{int(row['Hist Days']):>5}  "
            f"{flag}"
        )

    print()

    # ── Interpretation guide ────────────────────────────────────────────────
    print("READING THE OUTPUT:")
    print("  Composite Z > 2.0  →  Very unusual opening energy  (strong ORB candidate)")
    print("  Composite Z 1.0–2.0  →  Elevated activity          (monitor for breakout)")
    print("  Composite Z < 1.0  →  Normal opening               (skip)")
    print()
    print("  Range Z    = how much wider today's 15-min H-L is vs its own historical avg")
    print("  Dir Z      = how much stronger the net move (9:15 close - open) is vs history")
    print("  Range Ratio = today_range / hist_avg_range  (e.g. 2.5x = 2.5x wider than usual)")
    print("  Direction   = which way the first 15 min moved (UP/DOWN) — trade that side")
    print()

    # ── Top picks ───────────────────────────────────────────────────────────
    top = df_out[df_out["Rank"] <= TOP_N]
    up_picks   = top[top["Direction"].str.strip() == "UP"]["Ticker"].tolist()
    down_picks = top[top["Direction"].str.strip() == "DOWN"]["Ticker"].tolist()

    print(f"TOP {TOP_N} ORB CANDIDATES  |  {_td}")
    if up_picks:
        print(f"  LONG  side: {', '.join(up_picks)}")
    if down_picks:
        print(f"  SHORT side: {', '.join(down_picks)}")
    print()

    if skipped:
        names = [t.replace(".NS", "") for t in skipped]
        print(f"SKIPPED ({len(skipped)}): {', '.join(names)}")
        print()

    if SAVE_CSV:
        df_out.to_csv(CSV_PATH, index=False)
        print(f"Full results saved → {CSV_PATH}")

    print(f"{'='*72}\n")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_screener()
