"""
signals/engine.py - Signal generation from sentiment scores
This is the NumPy-heavy module: weighted scoring, normalization, thresholding.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Signal:
    ticker: str
    signal: str             # BUY / SELL / HOLD
    avg_score: float        # simple mean of all scores
    weighted_score: float   # upvote-weighted mean
    confidence: float       # mean confidence from FinBERT
    post_count: int
    reddit_score: float | None
    twitter_score: float | None
    label_distribution: dict  # {"positive": 0.6, "neutral": 0.2, "negative": 0.2}
    reasoning: str


# Thresholds — tune these as needed
BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15
MIN_POSTS = 3


def compute_signal(df: pd.DataFrame, ticker: str) -> Signal:
    """
    Given a sentiment-analyzed DataFrame, compute a trading signal.

    Args:
        df: DataFrame with columns [score, label, confidence, upvotes, source]
        ticker: Stock ticker symbol

    Returns:
        Signal dataclass
    """
    if df.empty or len(df) < MIN_POSTS:
        return Signal(
            ticker=ticker,
            signal="HOLD",
            avg_score=0.0,
            weighted_score=0.0,
            confidence=0.0,
            post_count=len(df),
            reddit_score=None,
            twitter_score=None,
            label_distribution={"positive": 0, "neutral": 0, "negative": 0},
            reasoning=f"Not enough data (only {len(df)} posts found, need {MIN_POSTS}+).",
        )

    scores = df["score"].values.astype(np.float64)
    upvotes = df["upvotes"].values.astype(np.float64)
    confidences = df["confidence"].values.astype(np.float64)

    # --- NumPy: Simple average ---
    avg_score = float(np.mean(scores))

    # --- NumPy: Upvote-weighted average ---
    weights = upvotes + 1.0  # +1 to avoid zero weights
    weights = weights / weights.sum()  # normalize to sum = 1
    weighted_score = float(np.dot(weights, scores))

    # --- NumPy: Confidence-adjusted score ---
    conf_weights = confidences / confidences.sum() if confidences.sum() > 0 else weights
    confidence_adjusted = float(np.dot(conf_weights, scores))

    # --- NumPy: Per-source scores ---
    reddit_mask = df["source"].str.startswith("reddit").values
    twitter_mask = df["source"].str.startswith("twitter").values

    reddit_score = float(np.mean(scores[reddit_mask])) if reddit_mask.any() else None
    twitter_score = float(np.mean(scores[twitter_mask])) if twitter_mask.any() else None

    # --- Label distribution ---
    label_counts = df["label"].value_counts(normalize=True)
    label_dist = {
        "positive": round(float(label_counts.get("positive", 0)), 3),
        "neutral":  round(float(label_counts.get("neutral", 0)), 3),
        "negative": round(float(label_counts.get("negative", 0)), 3),
    }

    # --- NumPy: Clip final score to [-1, 1] ---
    final_score = float(np.clip(weighted_score, -1.0, 1.0))

    # --- Signal thresholding ---
    if final_score >= BUY_THRESHOLD:
        signal = "BUY"
        reasoning = (
            f"{label_dist['positive']*100:.1f}% positive posts, "
            f"weighted sentiment score of {final_score:+.3f} exceeds BUY threshold ({BUY_THRESHOLD})."
        )
    elif final_score <= SELL_THRESHOLD:
        signal = "SELL"
        reasoning = (
            f"{label_dist['negative']*100:.1f}% negative posts, "
            f"weighted sentiment score of {final_score:+.3f} below SELL threshold ({SELL_THRESHOLD})."
        )
    else:
        signal = "HOLD"
        reasoning = (
            f"Mixed sentiment. Score {final_score:+.3f} is between thresholds "
            f"({SELL_THRESHOLD} to {BUY_THRESHOLD})."
        )

    mean_confidence = float(np.mean(confidences))

    return Signal(
        ticker=ticker.upper(),
        signal=signal,
        avg_score=round(avg_score, 4),
        weighted_score=round(final_score, 4),
        confidence=round(mean_confidence, 4),
        post_count=len(df),
        reddit_score=round(reddit_score, 4) if reddit_score is not None else None,
        twitter_score=round(twitter_score, 4) if twitter_score is not None else None,
        label_distribution=label_dist,
        reasoning=reasoning,
    )


def rolling_sentiment(df: pd.DataFrame, window: str = "1D") -> pd.DataFrame:
    """
    Compute rolling average sentiment over time windows.
    Useful for the dashboard trend chart.

    Args:
        df: DataFrame with [score, created_at] columns
        window: Pandas time window string ('1H', '6H', '1D', '7D')

    Returns:
        DataFrame with [created_at, rolling_score, rolling_count]
    """
    if df.empty:
        return pd.DataFrame(columns=["created_at", "rolling_score", "rolling_count"])

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.set_index("created_at").sort_index()

    rolled = df["score"].resample(window).agg(
        rolling_score="mean",
        rolling_count="count"
    ).reset_index()

    # NumPy: fill NaN rolling scores with 0
    rolled["rolling_score"] = np.nan_to_num(rolled["rolling_score"].values, nan=0.0)
    return rolled


def correlation_matrix(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> np.ndarray:
    """
    Compute correlation between daily sentiment scores and price % change.

    Args:
        sentiment_df: DataFrame with [created_at, score]
        price_df: DataFrame with [Date, Close]

    Returns:
        NumPy correlation matrix
    """
    if sentiment_df.empty or price_df.empty:
        return np.array([])

    sent = sentiment_df.copy()
    sent["date"] = pd.to_datetime(sent["created_at"]).dt.date
    daily_sent = sent.groupby("date")["score"].mean().reset_index()
    daily_sent.columns = ["date", "avg_sentiment"]

    price = price_df.copy()
    price["date"] = pd.to_datetime(price["Date"]).dt.date
    price["pct_change"] = price["Close"].pct_change() * 100

    merged = pd.merge(daily_sent, price[["date", "pct_change"]], on="date", how="inner")

    if len(merged) < 2:
        return np.array([])

    # NumPy correlation
    corr = np.corrcoef(
        merged["avg_sentiment"].values,
        merged["pct_change"].values
    )
    return np.round(corr, 4)
