"""
scraper/reddit.py - Scrapes Reddit posts for a stock ticker
Uses Reddit's PUBLIC JSON API — zero API keys required.
Reddit exposes every subreddit as JSON at /search.json, no auth needed.
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

FINANCE_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 SentiFi/1.0 (stock sentiment analyzer)"
}


def _fetch_subreddit(subreddit: str, ticker: str, limit: int = 25) -> list:
    """
    Fetch posts from a subreddit using Reddit's public JSON search endpoint.
    No API key needed — this is publicly available.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": ticker,
        "restrict_sr": "true",   # search within this subreddit only
        "sort": "new",
        "limit": min(limit, 25), # Reddit public API allows max 25 per request
        "t": "week",             # posts from last week
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 429:
            print(f"[Reddit] Rate limited on r/{subreddit}, skipping.")
            return []
        if response.status_code != 200:
            print(f"[Reddit] r/{subreddit} returned {response.status_code}")
            return []
        data = response.json()
        return data.get("data", {}).get("children", [])
    except Exception as e:
        print(f"[Reddit] Error fetching r/{subreddit}: {e}")
        return []


def scrape_reddit(ticker: str, limit: int = 50) -> pd.DataFrame:
    """
    Scrape posts mentioning a ticker from finance subreddits.
    Uses Reddit public JSON — NO API KEY NEEDED.

    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
        limit: Approximate max total posts to collect

    Returns:
        pd.DataFrame with columns: [text, source, upvotes, created_at, url]
    """
    records = []
    per_sub = max(10, limit // len(FINANCE_SUBREDDITS))

    for subreddit_name in FINANCE_SUBREDDITS:
        posts = _fetch_subreddit(subreddit_name, ticker, limit=per_sub)

        for post in posts:
            d = post.get("data", {})
            title = d.get("title", "")
            body = d.get("selftext", "")
            full_text = f"{title} {body}".strip()

            if len(full_text) < 10:
                continue

            records.append({
                "text": full_text,
                "source": f"reddit/{subreddit_name}",
                "upvotes": d.get("score", 0),
                "created_at": datetime.utcfromtimestamp(d.get("created_utc", 0)),
                "url": f"https://reddit.com{d.get('permalink', '')}",
            })

        # Be polite — small delay between subreddit requests
        time.sleep(0.5)

    if not records:
        return pd.DataFrame(columns=["text", "source", "upvotes", "created_at", "url"])

    df = pd.DataFrame(records)

    # --- Pandas cleaning pipeline ---
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)   # remove URLs
    df["text"] = df["text"].str.replace(r"[^\w\s]", " ", regex=True)  # remove special chars
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 15]                                 # drop very short posts
    df = df.drop_duplicates(subset=["text"])                           # deduplicate

    # NumPy: clip upvotes to avoid extreme outliers skewing weights
    df["upvotes"] = np.clip(df["upvotes"].values, 0, 10000)

    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    return df
