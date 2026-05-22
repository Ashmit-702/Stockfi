"""
tests/test_signals.py - Unit tests for the signal engine
Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
from app.signals.engine import compute_signal, rolling_sentiment, Signal


def make_df(scores: list, sources: list = None, upvotes: list = None) -> pd.DataFrame:
    n = len(scores)
    return pd.DataFrame({
        "score": scores,
        "label": ["positive" if s > 0 else ("negative" if s < 0 else "neutral") for s in scores],
        "confidence": [0.9] * n,
        "source": sources or ["reddit"] * n,
        "upvotes": upvotes or [10] * n,
        "created_at": pd.date_range("2024-01-01", periods=n, freq="h"),
    })


def test_buy_signal():
    df = make_df([0.8, 0.7, 0.9, 0.6, 0.75])
    signal = compute_signal(df, "AAPL")
    assert signal.signal == "BUY"
    assert signal.weighted_score > 0.15


def test_sell_signal():
    df = make_df([-0.8, -0.7, -0.9, -0.6, -0.75])
    signal = compute_signal(df, "AAPL")
    assert signal.signal == "SELL"
    assert signal.weighted_score < -0.15


def test_hold_signal():
    df = make_df([0.1, -0.05, 0.08, -0.1, 0.05])
    signal = compute_signal(df, "AAPL")
    assert signal.signal == "HOLD"


def test_not_enough_data():
    df = make_df([0.9, 0.8])  # only 2 posts
    signal = compute_signal(df, "TSLA")
    assert signal.signal == "HOLD"
    assert "enough data" in signal.reasoning.lower()


def test_empty_dataframe():
    df = pd.DataFrame(columns=["score", "label", "confidence", "source", "upvotes", "created_at"])
    signal = compute_signal(df, "MSFT")
    assert signal.signal == "HOLD"


def test_weighted_score_uses_upvotes():
    # High upvote post is positive — should pull signal positive
    df = make_df(
        scores=[-0.5, -0.5, -0.5, 0.9],
        upvotes=[1, 1, 1, 10000],
    )
    signal = compute_signal(df, "NVDA")
    # Weighted score should be more positive than simple average
    assert signal.weighted_score > signal.avg_score


def test_label_distribution_sums_to_one():
    df = make_df([0.8, -0.5, 0.0, 0.3, -0.2])
    signal = compute_signal(df, "GOOG")
    dist_sum = sum(signal.label_distribution.values())
    assert abs(dist_sum - 1.0) < 0.01


def test_rolling_sentiment():
    df = make_df([0.5, -0.3, 0.2, 0.8, -0.1])
    rolled = rolling_sentiment(df, window="1h")
    assert isinstance(rolled, pd.DataFrame)
    assert "rolling_score" in rolled.columns


def test_per_source_scores():
    df = make_df(
        scores=[0.8, 0.7, -0.9, -0.8],
        sources=["reddit", "reddit", "twitter", "twitter"],
    )
    signal = compute_signal(df, "AMC")
    assert signal.reddit_score is not None
    assert signal.twitter_score is not None
    assert signal.reddit_score > 0
    assert signal.twitter_score < 0
