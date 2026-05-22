"""
scraper/twitter.py - Scrapes tweets for a stock ticker using Tweepy v2
Returns a clean Pandas DataFrame.
"""

import os
import tweepy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_twitter_client() -> tweepy.Client:
    return tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))


def scrape_twitter(ticker: str, limit: int = 50) -> pd.DataFrame:
    """
    Scrape recent tweets mentioning a stock ticker.

    Args:
        ticker: Stock ticker e.g. 'AAPL'
        limit: Max number of tweets (max 100 per request on free tier)

    Returns:
        pd.DataFrame with columns: [text, source, upvotes, created_at]
    """
    client = get_twitter_client()
    query = f"${ticker} OR #{ticker} lang:en -is:retweet -is:reply"

    records = []
    try:
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(limit, 100),
            tweet_fields=["created_at", "public_metrics", "text"],
            start_time=datetime.utcnow() - timedelta(days=7),
        )

        if not tweets.data:
            return pd.DataFrame(columns=["text", "source", "upvotes", "created_at", "url"])

        for tweet in tweets.data:
            likes = tweet.public_metrics.get("like_count", 0) if tweet.public_metrics else 0
            records.append({
                "text": tweet.text,
                "source": "twitter",
                "upvotes": likes,
                "created_at": tweet.created_at,
                "url": f"https://twitter.com/i/web/status/{tweet.id}",
            })

    except Exception as e:
        print(f"[Twitter] Error scraping ${ticker}: {e}")
        return pd.DataFrame(columns=["text", "source", "upvotes", "created_at", "url"])

    if not records:
        return pd.DataFrame(columns=["text", "source", "upvotes", "created_at", "url"])

    df = pd.DataFrame(records)

    # --- Pandas cleaning pipeline ---
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)
    df["text"] = df["text"].str.replace(r"[^\w\s\$\#]", " ", regex=True)
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 10]
    df = df.drop_duplicates(subset=["text"])

    # NumPy: normalize likes (log scale to reduce influence of viral tweets)
    df["upvotes"] = np.log1p(df["upvotes"].values)

    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    return df
