"""
scraper/reddit.py - Scrapes Reddit for Indian stock sentiment
Uses Reddit public JSON API - no key needed.
Focuses on Indian finance subreddits + global ones.
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

INDIAN_SUBREDDITS = [
    "IndiaInvestments",
    "DalalStreetTalks", 
    "IndianStockMarket",
    "stocks",
    "investing",
    "wallstreetbets",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 SentiFi/1.0"
}


def _fetch_subreddit(subreddit: str, ticker: str, limit: int = 25) -> list:
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    # For Indian stocks, also search without .NS suffix
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    params = {
        "q": clean_ticker,
        "restrict_sr": "true",
        "sort": "new",
        "limit": min(limit, 25),
        "t": "month",
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if response.status_code == 429:
            print(f"[Reddit] Rate limited on r/{subreddit}")
            return []
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("data", {}).get("children", [])
    except Exception as e:
        print(f"[Reddit] Error r/{subreddit}: {e}")
        return []


def scrape_reddit(ticker: str, limit: int = 50) -> pd.DataFrame:
    records = []
    per_sub = max(10, limit // len(INDIAN_SUBREDDITS))

    for subreddit_name in INDIAN_SUBREDDITS:
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
        time.sleep(0.3)

    if not records:
        return pd.DataFrame(columns=["text", "source", "upvotes", "created_at", "url"])

    df = pd.DataFrame(records)
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)
    df["text"] = df["text"].str.replace(r"[^\w\s]", " ", regex=True)
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 15]
    df = df.drop_duplicates(subset=["text"])
    df["upvotes"] = np.clip(df["upvotes"].values, 0, 10000)
    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    return df
