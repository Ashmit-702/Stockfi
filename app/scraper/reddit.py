"""
scraper/reddit.py - Scrapes financial news headlines as Reddit fallback.
Reddit blocks cloud IPs. Instead we scrape MoneyControl/ET headlines
via RSS feeds - no API key, works from any server.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET

NEWS_FEEDS = [
    ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "EconomicTimes"),
    ("https://www.moneycontrol.com/rss/marketreports.xml", "MoneyControl"),
    ("https://feeds.feedburner.com/ndtvprofit-latest", "NDTVProfit"),
]


def _fetch_rss(url: str, source: str, ticker: str) -> list:
    clean = ticker.replace(".NS", "").replace(".BO", "").upper()
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 SentiFi/1.0"
        })
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)
        records = []
        for item in root.iter("item"):
            title = item.findtext("title", "") or ""
            desc = item.findtext("description", "") or ""
            pub_date = item.findtext("pubDate", "") or ""
            text = f"{title} {desc}".strip()

            # Only keep articles mentioning the ticker or company
            if clean.lower() not in text.lower() and ticker.lower() not in text.lower():
                continue

            try:
                created = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
            except:
                created = datetime.utcnow()

            records.append({
                "text": text[:500],
                "source": f"news/{source}",
                "upvotes": 5,
                "created_at": created,
                "url": item.findtext("link", ""),
            })
        return records
    except Exception as e:
        print(f"[News] {source} error: {e}")
        return []


def scrape_reddit(ticker: str, limit: int = 50) -> pd.DataFrame:
    """
    Scrapes Indian financial news RSS feeds instead of Reddit.
    Reddit blocks cloud IPs — RSS feeds are reliable and free.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    records = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_fetch_rss, url, src, ticker) for url, src in NEWS_FEEDS]
        for f in as_completed(futures, timeout=20):
            try:
                records.extend(f.result())
            except:
                pass

    if not records:
        return pd.DataFrame(columns=["text", "source", "upvotes", "created_at", "url"])

    df = pd.DataFrame(records)
    df["text"] = df["text"].str.replace(r"<[^>]+>", " ", regex=True)  # strip HTML
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 15]
    df = df.drop_duplicates(subset=["text"])
    df["upvotes"] = np.clip(df["upvotes"].values, 0, 100)
    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    return df
