"""
sentiment/analyzer.py - FinBERT-based sentiment analysis
FinBERT is a finance-specific BERT model, far better than generic models for stock sentiment.
Processes a Pandas DataFrame and returns it enriched with sentiment scores.
"""

import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Singleton pattern — load model once, reuse across requests
_pipeline = None


def load_model():
    global _pipeline
    if _pipeline is None:
        print("[FinBERT] Loading model... (first time only)")
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        _pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
        )
        print("[FinBERT] Model loaded successfully.")
    return _pipeline


LABEL_SCORE_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def _score_from_output(output: list) -> tuple[float, str, float]:
    """
    Convert FinBERT output to (score, label, confidence).
    FinBERT returns: [{'label': 'positive', 'score': 0.9}, ...]
    """
    label_scores = {item["label"]: item["score"] for item in output}
    best_label = max(label_scores, key=label_scores.get)
    confidence = label_scores[best_label]

    # Weighted numeric score: positive contribution - negative contribution
    numeric_score = (
        label_scores.get("positive", 0) * 1.0
        + label_scores.get("neutral", 0) * 0.0
        + label_scores.get("negative", 0) * -1.0
    )
    return round(numeric_score, 4), best_label, round(confidence, 4)


def analyze_dataframe(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """
    Run FinBERT on a DataFrame with a 'text' column.
    Adds: score (-1 to 1), label, confidence columns.

    Args:
        df: DataFrame with 'text' column
        batch_size: texts per batch for inference

    Returns:
        df with added columns: [score, label, confidence]
    """
    if df.empty:
        df["score"] = pd.Series(dtype=float)
        df["label"] = pd.Series(dtype=str)
        df["confidence"] = pd.Series(dtype=float)
        return df

    nlp = load_model()
    texts = df["text"].tolist()

    # Truncate texts to 512 tokens (FinBERT limit)
    texts = [t[:512] for t in texts]

    scores, labels, confidences = [], [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            results = nlp(batch)
            for output in results:
                s, l, c = _score_from_output(output)
                scores.append(s)
                labels.append(l)
                confidences.append(c)
        except Exception as e:
            print(f"[FinBERT] Batch error: {e}")
            for _ in batch:
                scores.append(0.0)
                labels.append("neutral")
                confidences.append(0.0)

    df = df.copy()
    df["score"] = np.array(scores)
    df["label"] = labels
    df["confidence"] = np.array(confidences)

    return df
