"""
sentiment/analyzer.py - Lightweight sentiment analysis using VADER.
VADER is rule-based, uses ~5MB RAM (vs FinBERT's 600MB).
Finance-tuned via custom lexicon boosting stock-specific words.
Produces -1 to +1 scores with positive/negative/neutral labels.
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Singleton
_analyzer = None

# Finance-specific word boosts for VADER
FINANCE_LEXICON = {
    "bullish": 3.0, "bearish": -3.0,
    "moon": 2.5, "rocket": 2.0,
    "dump": -2.5, "crash": -3.0, "dip": -1.5,
    "buy": 1.5, "sell": -1.5, "short": -1.0,
    "calls": 1.5, "puts": -1.5,
    "undervalued": 2.0, "overvalued": -2.0,
    "squeeze": 2.0, "bubble": -2.0,
    "beat": 2.0, "miss": -2.0,
    "upgrade": 2.5, "downgrade": -2.5,
    "bankrupt": -3.5, "fraud": -3.5, "lawsuit": -2.0,
    "dividend": 1.5, "buyback": 2.0, "rally": 2.5,
}


def load_model() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        print("[Sentiment] Loading VADER analyzer...")
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_LEXICON)
        print("[Sentiment] Ready.")
    return _analyzer


def _score_text(analyzer, text: str) -> tuple:
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    confidence = abs(compound)
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return round(compound, 4), label, round(confidence, 4)


def analyze_dataframe(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    if df.empty:
        df["score"] = pd.Series(dtype=float)
        df["label"] = pd.Series(dtype=str)
        df["confidence"] = pd.Series(dtype=float)
        return df

    analyzer = load_model()
    scores, labels, confidences = [], [], []

    for text in df["text"].tolist():
        try:
            s, l, c = _score_text(analyzer, str(text))
        except Exception:
            s, l, c = 0.0, "neutral", 0.0
        scores.append(s)
        labels.append(l)
        confidences.append(c)

    df = df.copy()
    df["score"] = np.array(scores, dtype=np.float64)
    df["label"] = labels
    df["confidence"] = np.array(confidences, dtype=np.float64)
    return df
