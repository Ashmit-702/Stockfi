"""
scraper/reddit.py - Indian financial news via RSS + Google News
Searches by company name for better article matching.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote

# Map tickers to search terms for better matching
TICKER_NAMES = {
    "RELIANCE": "Reliance Industries",
    "TCS": "TCS Tata Consultancy",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "WIPRO": "Wipro",
    "TATAMOTORS": "Tata Motors",
    "ADANIENT": "Adani Enterprises",
    "BAJFINANCE": "Bajaj Finance",
    "SBIN": "SBI State Bank India",
    "ICICIBANK": "ICICI Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "ITC": "ITC Limited",
    "MARUTI": "Maruti Suzuki",
    "HCLTECH": "HCL Technologies",
    "SUNPHARMA": "Sun Pharma",
    "AXISBANK": "Axis Bank",
    "KOTAKBANK": "Kotak Bank",
    "LT": "Larsen Toubro",
    "NTPC": "NTPC",
    "POWERGRID": "Power Grid",
}

NEWS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://feeds.feedburner.com/ndtvprofit-latest",
]


def _get_search_terms(ticker: str) -> list:
    clean = ticker.replace(".NS", "").replace(".BO", "").upper()
    terms = [clean]
    if clean in TICKER_NAMES:
        terms.append(TICKER_NAMES[clean])
    return [t.lower() for t in terms]


def _fetch_google_news(ticker: str) -> list:
    """Google News RSS - searches by company name, very reliable."""
    clean = ticker.replace(".NS", "").replace(".BO", "").upper()
    query = TICKER_NAMES.get(clean, clean) + " stock NSE"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    records = []
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0 SentiFi/1.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        for item in root.iter("item"):
            title = item.findtext("title", "") or ""
            pub_date = item.findtext("pubDate", "") or ""
            if len(title) < 10:
                continue
            try:
                created = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
            except:
                created = datetime.utcnow()
            records.append({
                "text": title,
                "source": "news/GoogleNews",
                "upvotes": 5,
                "created_at": created,
                "url": item.findtext("link", ""),
            })
    except Exception as e:
        print(f"[GoogleNews] Error: {e}")
    return records


def _fetch_rss(url: str, search_terms: list) -> list:
    records = []
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 SentiFi/1.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        source_name = url.split("/")[2].replace("www.", "").split(".")[0].capitalize()
        for item in root.iter("item"):
            title = item.findtext("title", "") or ""
            desc = item.findtext("description", "") or ""
            text = f"{title} {desc}".strip()
            text_lower = text.lower()
            # Check if any search term appears in the article
            if not any(term in text_lower for term in search_terms):
                continue
            pub_date = item.findtext("pubDate", "") or ""
            try:
                created = datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
            except:
                created = datetime.utcnow()
            records.append({
                "text": text[:400],
                "source": f"news/{source_name}",
                "upvotes": 5,
                "created_at": created,
                "url": item.findtext("link", ""),
            })
    except Exception as e:
        print(f"[RSS] {url} error: {e}")
    return records


def scrape_reddit(ticker: str, limit: int = 50) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    search_terms = _get_search_terms(ticker)
    records = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_fetch_google_news, ticker)]
        futures += [executor.submit(_fetch_rss, url, search_terms) for url in NEWS_FEEDS]
        for f in as_completed(futures, timeout=20):
            try:
                records.extend(f.result())
            except:
                pass

    if not records:
        return pd.DataFrame(columns=["text", "source", "upvotes", "created_at", "url"])

    df = pd.DataFrame(records)
    df["text"] = df["text"].str.replace(r"<[^>]+>", " ", regex=True)
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 10]
    df = df.drop_duplicates(subset=["text"])
    df["upvotes"] = np.clip(df["upvotes"].values, 0, 100)
    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    return df.head(limit)
