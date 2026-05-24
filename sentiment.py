"""
Sentiment & Risk Filter Module

Sources:
  1. Yahoo Finance news headlines + VADER (always free, no setup)
  2. X/Twitter via official API v2 (requires bearer token — set X_BEARER_TOKEN in env or pass directly)
  3. Earnings/results day detection via news keyword scan (primary) + yfinance calendar (fallback)

Usage:
  from sentiment import run_filters, get_sentiment_summary

  result = run_filters("RELIANCE", "BUY", "2026-05-23")
  # result = {"pass": True/False, "reason": "...", "details": {...}}
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore', category=FutureWarning)

# ── VADER sentiment analyser (optional — gracefully degrades if not installed) ─
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    VADER_OK = True
except ImportError:
    VADER_OK = False


# ── Thresholds ─────────────────────────────────────────────────────────────────
# VADER compound score: > +0.05 = positive, < -0.05 = negative, else neutral
SENT_THRESHOLD   = 0.05
RESULTS_BUFFER   = 1    # skip if earnings within ±N calendar days

# Keywords that indicate a results/earnings announcement in a news headline
_RESULTS_KEYWORDS = {
    "results", "quarterly", "earnings", "q4", "q3", "q2", "q1",
    "pat", "financial results", "net profit", "annual results",
    "declares dividend", "revenue", "profit after tax",
}


# ══════════════════════════════════════════════════════════════════════════════
#  1. RESULTS / EARNINGS DAY CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_results_day(ticker: str, trade_date: str) -> dict:
    """
    Returns whether trade_date is within RESULTS_BUFFER days of an earnings announcement.
    Method 1 (primary): scan recent Yahoo Finance news headlines for results keywords.
    Method 2 (fallback): yfinance earnings calendar (often empty for NSE stocks).
    """
    ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    td = pd.to_datetime(trade_date).date()

    # ── Method 1: News headline keyword scan ──────────────────────────────────
    # Reliable for NSE stocks where the earnings calendar is typically empty.
    # Scans article titles for results-related keywords published near trade_date.
    try:
        news = yf.Ticker(ticker_ns).news or []
        for item in news:
            # Extract publish timestamp (yfinance returns both old and new schemas)
            pub_ts = (
                (item.get("content", {}) or {}).get("pubDate")
                or item.get("providerPublishTime")
            )
            if pub_ts is None:
                continue
            if isinstance(pub_ts, (int, float)):
                pub_date = datetime.utcfromtimestamp(pub_ts).date()
            else:
                pub_date = pd.to_datetime(pub_ts).date()

            days_away = abs((pub_date - td).days)
            # Use a slightly wider window (RESULTS_BUFFER + 4) to catch
            # articles published a couple of days after the announcement
            if days_away > RESULTS_BUFFER + 4:
                continue

            title = (
                (item.get("content", {}) or {}).get("title", "")
                or item.get("title", "")
            ).lower()

            if any(kw in title for kw in _RESULTS_KEYWORDS):
                return {
                    "is_results_day": True,
                    "earnings_date":  pub_date.strftime("%Y-%m-%d"),
                    "days_away":      int(days_away),
                    "source":         "news_scan",
                }
    except Exception:
        pass

    # ── Method 2: yfinance earnings calendar (fallback) ───────────────────────
    try:
        t = yf.Ticker(ticker_ns)
        earnings = t.get_earnings_dates(limit=12)
        if earnings is not None and not earnings.empty:
            for ed in earnings.index:
                days_away = abs((ed.date() - td).days)
                if days_away <= RESULTS_BUFFER:
                    return {
                        "is_results_day": True,
                        "earnings_date":  ed.strftime("%Y-%m-%d"),
                        "days_away":      int(days_away),
                        "source":         "earnings_calendar",
                    }
    except Exception:
        pass

    return {"is_results_day": False, "earnings_date": None, "days_away": None, "source": "none"}


# ══════════════════════════════════════════════════════════════════════════════
#  2. NEWS SENTIMENT (Yahoo Finance + VADER)
# ══════════════════════════════════════════════════════════════════════════════
def get_news_sentiment(ticker: str, n_articles: int = 10) -> dict:
    """
    Fetches recent Yahoo Finance news headlines and scores them with VADER.
    Returns composite sentiment signal: POSITIVE / NEUTRAL / NEGATIVE
    """
    ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"

    if not VADER_OK:
        return {
            "signal": "NEUTRAL", "score": 0.0, "confidence": 0.0,
            "n_articles": 0, "articles": [],
            "source": "vader_not_installed — run: pip install vaderSentiment",
        }

    try:
        news = yf.Ticker(ticker_ns).news
        if not news:
            return {"signal": "NEUTRAL", "score": 0.0, "confidence": 0.0,
                    "n_articles": 0, "articles": [], "source": "no_news_available"}

        articles = []
        scores   = []
        for item in news[:n_articles]:
            title = (item.get("content", {}) or {}).get("title", "") or item.get("title", "")
            if not title:
                continue
            vs    = _VADER.polarity_scores(title)
            scores.append(vs["compound"])
            articles.append({
                "title":     title,
                "score":     round(vs["compound"], 3),
                "publisher": (item.get("content", {}) or {}).get("provider", {}).get("displayName", "")
                             or item.get("publisher", ""),
            })

        if not scores:
            return {"signal": "NEUTRAL", "score": 0.0, "confidence": 0.0,
                    "n_articles": 0, "articles": [], "source": "yahoo_finance"}

        avg   = sum(scores) / len(scores)
        signal = ("POSITIVE" if avg > SENT_THRESHOLD
                  else "NEGATIVE" if avg < -SENT_THRESHOLD
                  else "NEUTRAL")

        return {
            "signal":     signal,
            "score":      round(avg, 4),
            "confidence": round(abs(avg) / SENT_THRESHOLD * 100, 1),
            "n_articles": len(articles),
            "articles":   articles,
            "source":     "yahoo_finance",
        }
    except Exception as e:
        return {"signal": "NEUTRAL", "score": 0.0, "confidence": 0.0,
                "n_articles": 0, "articles": [],
                "source": f"error: {str(e)[:60]}"}


# ══════════════════════════════════════════════════════════════════════════════
#  3. X / TWITTER SENTIMENT (official API v2 — bearer token required)
# ══════════════════════════════════════════════════════════════════════════════
def get_x_sentiment(ticker: str, bearer_token: str = None,
                    max_results: int = 20) -> dict:
    """
    Fetches recent X posts mentioning the ticker and scores them with VADER.
    bearer_token: X API v2 Bearer Token.
                  If None, falls back to env var X_BEARER_TOKEN.
    Setup: https://developer.x.com → Create app → Get Bearer Token (free basic tier).
    """
    token = bearer_token or os.environ.get("X_BEARER_TOKEN", "")
    if not token:
        return {
            "available": False,
            "reason":    "X Bearer Token not set. Pass bearer_token= or set env X_BEARER_TOKEN.",
        }

    if not VADER_OK:
        return {"available": False, "reason": "vaderSentiment not installed."}

    try:
        import requests

        base   = ticker.replace(".NS", "")
        query  = f"({base} OR ${base}) lang:en -is:retweet"
        resp   = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers={"Authorization": f"Bearer {token}"},
            params={"query": query, "max_results": max(10, max_results),
                    "tweet.fields": "created_at,public_metrics"},
            timeout=10,
        )

        if resp.status_code == 401:
            return {"available": False, "reason": "Invalid X Bearer Token."}
        if resp.status_code != 200:
            return {"available": False,
                    "reason": f"X API returned HTTP {resp.status_code}"}

        tweets = resp.json().get("data", [])
        if not tweets:
            return {"available": True, "signal": "NEUTRAL", "score": 0.0,
                    "confidence": 0.0, "n_tweets": 0, "tweets": []}

        scores     = []
        tweet_list = []
        for tw in tweets:
            text = tw.get("text", "")
            vs   = _VADER.polarity_scores(text)
            scores.append(vs["compound"])
            tweet_list.append({
                "text":  text[:120],
                "score": round(vs["compound"], 3),
                "likes": tw.get("public_metrics", {}).get("like_count", 0),
            })

        avg    = sum(scores) / len(scores)
        signal = ("POSITIVE" if avg > SENT_THRESHOLD
                  else "NEGATIVE" if avg < -SENT_THRESHOLD
                  else "NEUTRAL")

        return {
            "available":  True,
            "signal":     signal,
            "score":      round(avg, 4),
            "confidence": round(abs(avg) / SENT_THRESHOLD * 100, 1),
            "n_tweets":   len(tweets),
            "tweets":     tweet_list[:5],
        }
    except ImportError:
        return {"available": False, "reason": "requests library not installed."}
    except Exception as e:
        return {"available": False, "reason": str(e)[:100]}


# ══════════════════════════════════════════════════════════════════════════════
#  4. COMBINED FILTER GATE
# ══════════════════════════════════════════════════════════════════════════════
def run_filters(ticker: str, side: str, trade_date: str,
                x_bearer_token:     str  = None,
                block_results_day:  bool = True,
                block_bad_news:     bool = True,
                block_bad_x:        bool = True) -> dict:
    """
    Run all pre-trade risk filters for a single stock.

    Returns:
        {
            "pass":    True / False,
            "reason":  human-readable verdict,
            "details": { results, news_sentiment, x_sentiment }
        }
    """
    details = {}

    # ── Filter 1: Earnings / results day ──────────────────────────────
    res = check_results_day(ticker, trade_date)
    details["results"] = res
    if block_results_day and res["is_results_day"]:
        return {
            "pass":    False,
            "reason":  f"Results Day\nearnings on {res['earnings_date']} ({res['days_away']}d away)",
            "details": details,
        }

    # ── Filter 2: News sentiment ───────────────────────────────────────
    news = get_news_sentiment(ticker)
    details["news"] = news

    if block_bad_news:
        # BUY needs non-negative news; SELL needs non-positive news
        if side == "BUY"  and news["score"] < -SENT_THRESHOLD:
            return {
                "pass":    False,
                "reason":  f"Negative News\nscore {news['score']:+.3f} blocks BUY",
                "details": details,
            }
        if side == "SELL" and news["score"] > SENT_THRESHOLD:
            return {
                "pass":    False,
                "reason":  f"Positive News\nscore {news['score']:+.3f} blocks SELL",
                "details": details,
            }

    # ── Filter 3: X / Twitter sentiment ───────────────────────────────
    x = get_x_sentiment(ticker, bearer_token=x_bearer_token)
    details["x"] = x

    if block_bad_x and x.get("available"):
        if side == "BUY"  and x["score"] < -SENT_THRESHOLD:
            return {
                "pass":    False,
                "reason":  f"Negative X Sentiment\nscore {x['score']:+.3f} blocks BUY",
                "details": details,
            }
        if side == "SELL" and x["score"] > SENT_THRESHOLD:
            return {
                "pass":    False,
                "reason":  f"Positive X Sentiment\nscore {x['score']:+.3f} blocks SELL",
                "details": details,
            }

    return {"pass": True, "reason": "All filters passed", "details": details}


# ══════════════════════════════════════════════════════════════════════════════
#  5. BATCH SENTIMENT SUMMARY  (for UI display)
# ══════════════════════════════════════════════════════════════════════════════
def get_sentiment_summary(tickers: list[str], trade_date: str,
                          x_bearer_token: str = None) -> dict[str, dict]:
    """
    Run full sentiment + results check for a list of tickers.
    Returns dict keyed by base ticker (no .NS).
    """
    results = {}
    for tkr in tickers:
        base = tkr.replace(".NS", "")
        results[base] = {
            "results": check_results_day(base, trade_date),
            "news":    get_news_sentiment(base),
            "x":       get_x_sentiment(base, bearer_token=x_bearer_token),
        }
    return results
