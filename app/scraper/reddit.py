"""
scraper/reddit.py - Reddit scraper with rotation and retries
Uses multiple user agents to avoid bot detection on Render.
"""

import time
import random
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

# Rotate user agents to avoid bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
]


def _fetch_subreddit(subreddit: str, ticker: str, limit: int = 25) -> list:
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": clean_ticker,
        "restrict_sr": "true",
        "sort": "new",
        "limit": min(limit, 25),
        "t": "month",
    }
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for attempt in range(2):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=12)
            if response.status_code == 200:
                return response.json().get("data", {}).get("children", [])
            elif response.status_code == 429:
                time.sleep(2)
            else:
                break
        except Exception as e:
            print(f"[Reddit] r/{subreddit} attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return []


def scrape_reddit(ticker: str, limit: int = 50) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    records = []

    def fetch(sub):
        return sub, _fetch_subreddit(sub, clean_ticker, limit=15)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch, sub): sub for sub in INDIAN_SUBREDDITS}
        for future in as_completed(futures, timeout=30):
            try:
                sub, posts = future.result()
                for post in posts:
                    d = post.get("data", {})
                    title = d.get("title", "")
                    body = d.get("selftext", "")
                    full_text = f"{title} {body}".strip()
                    if len(full_text) < 10:
                        continue
                    records.append({
                        "text": full_text,
                        "source": f"reddit/{sub}",
                        "upvotes": d.get("score", 0),
                        "created_at": datetime.utcfromtimestamp(d.get("created_utc", 0)),
                        "url": f"https://reddit.com{d.get('permalink', '')}",
                    })
            except Exception as e:
                print(f"[Reddit] Thread error: {e}")

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
