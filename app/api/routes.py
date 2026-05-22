"""
api/routes.py - FastAPI routes with parallel scraping for speed
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.db import get_db, SentimentRecord, SignalRecord
from app.scraper.reddit import scrape_reddit
from app.scraper.twitter import scrape_twitter
from app.sentiment.analyzer import analyze_dataframe
from app.signals.engine import compute_signal, rolling_sentiment
from app.stock.fetcher import get_price_history, get_ticker_info, get_summary_stats

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/analyze/{ticker}")
def analyze_ticker(
    ticker: str,
    sources: str = Query(default="reddit,twitter"),
    limit: int = Query(default=40, ge=5, le=100),
    db: Session = Depends(get_db),
):
    ticker = ticker.upper()
    source_list = [s.strip().lower() for s in sources.split(",")]
    frames = []

    # Scrape sources in parallel
    def fetch_reddit():
        return scrape_reddit(ticker, limit=limit)

    def fetch_twitter():
        return scrape_twitter(ticker, limit=limit)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if "reddit" in source_list:
            futures["reddit"] = executor.submit(fetch_reddit)
        if "twitter" in source_list:
            futures["twitter"] = executor.submit(fetch_twitter)

        for name, future in futures.items():
            try:
                df = future.result(timeout=35)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"[API] {name} fetch failed: {e}")

    if not frames:
        # Return neutral HOLD with 0 posts instead of 404
        return {
            "ticker": ticker,
            "signal": "HOLD",
            "avg_score": 0.0,
            "weighted_score": 0.0,
            "confidence": 0.0,
            "post_count": 0,
            "reddit_score": None,
            "twitter_score": None,
            "label_distribution": {"positive": 0, "neutral": 1, "negative": 0},
            "reasoning": f"No posts found for ${ticker}. Try a more popular ticker or check back later.",
        }

    combined_df = pd.concat(frames, ignore_index=True)
    analyzed_df = analyze_dataframe(combined_df)

    # Save to DB (batch)
    try:
        for _, row in analyzed_df.iterrows():
            db.add(SentimentRecord(
                ticker=ticker, source=row["source"],
                text=row["text"][:500], score=row["score"],
                label=row["label"], confidence=row["confidence"],
            ))
        db.commit()
    except Exception as e:
        print(f"[DB] Save error: {e}")

    signal = compute_signal(analyzed_df, ticker)

    try:
        db.add(SignalRecord(
            ticker=ticker, signal=signal.signal,
            avg_score=signal.avg_score, weighted_score=signal.weighted_score,
            post_count=signal.post_count, reddit_score=signal.reddit_score,
            twitter_score=signal.twitter_score,
        ))
        db.commit()
    except Exception as e:
        print(f"[DB] Signal save error: {e}")

    return {
        "ticker": ticker,
        "signal": signal.signal,
        "avg_score": signal.avg_score,
        "weighted_score": signal.weighted_score,
        "confidence": signal.confidence,
        "post_count": signal.post_count,
        "reddit_score": signal.reddit_score,
        "twitter_score": signal.twitter_score,
        "label_distribution": signal.label_distribution,
        "reasoning": signal.reasoning,
    }


@router.get("/stock/{ticker}")
def stock_data(
    ticker: str,
    days: int = Query(default=30, ge=1, le=365),
):
    ticker = ticker.upper()
    df = get_price_history(ticker, days=days)
    info = get_ticker_info(ticker)
    stats = get_summary_stats(df) if not df.empty else {}

    return {
        "ticker": ticker,
        "info": info,
        "stats": stats,
        "prices": df.to_dict(orient="records") if not df.empty else [],
    }


@router.get("/history/{ticker}")
def sentiment_history(
    ticker: str,
    limit: int = Query(default=100),
    db: Session = Depends(get_db),
):
    ticker = ticker.upper()
    records = (
        db.query(SentimentRecord)
        .filter(SentimentRecord.ticker == ticker)
        .order_by(SentimentRecord.created_at.desc())
        .limit(limit).all()
    )
    if not records:
        return {"ticker": ticker, "total_records": 0, "records": []}

    df = pd.DataFrame([{
        "text": r.text, "source": r.source, "score": r.score,
        "label": r.label, "confidence": r.confidence, "created_at": r.created_at,
    } for r in records])

    return {
        "ticker": ticker,
        "total_records": len(records),
        "records": df.head(50).to_dict(orient="records"),
    }


@router.get("/signals/{ticker}")
def signal_history(
    ticker: str,
    limit: int = Query(default=20),
    db: Session = Depends(get_db),
):
    ticker = ticker.upper()
    signals = (
        db.query(SignalRecord)
        .filter(SignalRecord.ticker == ticker)
        .order_by(SignalRecord.created_at.desc())
        .limit(limit).all()
    )
    return {
        "ticker": ticker,
        "signals": [{
            "signal": s.signal, "avg_score": s.avg_score,
            "weighted_score": s.weighted_score, "post_count": s.post_count,
            "created_at": s.created_at,
        } for s in signals]
    }
