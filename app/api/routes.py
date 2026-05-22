"""
api/routes.py - FastAPI REST endpoints for SentiFi
All endpoints are async and production-ready.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime

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
    sources: str = Query(default="reddit,twitter", description="Comma-separated: reddit,twitter"),
    limit: int = Query(default=50, ge=5, le=200),
    db: Session = Depends(get_db),
):
    """
    Main endpoint: scrape + analyze + return signal for a ticker.
    
    Example: GET /analyze/AAPL?sources=reddit,twitter&limit=50
    """
    ticker = ticker.upper()
    source_list = [s.strip().lower() for s in sources.split(",")]

    frames = []

    if "reddit" in source_list:
        reddit_df = scrape_reddit(ticker, limit=limit)
        if not reddit_df.empty:
            frames.append(reddit_df)

    if "twitter" in source_list:
        twitter_df = scrape_twitter(ticker, limit=limit)
        if not twitter_df.empty:
            frames.append(twitter_df)

    if not frames:
        raise HTTPException(status_code=404, detail=f"No posts found for ${ticker}.")

    combined_df = pd.concat(frames, ignore_index=True)
    analyzed_df = analyze_dataframe(combined_df)

    # Save to DB
    for _, row in analyzed_df.iterrows():
        record = SentimentRecord(
            ticker=ticker,
            source=row["source"],
            text=row["text"][:1000],
            score=row["score"],
            label=row["label"],
            confidence=row["confidence"],
        )
        db.add(record)
    db.commit()

    # Compute signal
    signal = compute_signal(analyzed_df, ticker)

    # Save signal
    sig_record = SignalRecord(
        ticker=ticker,
        signal=signal.signal,
        avg_score=signal.avg_score,
        weighted_score=signal.weighted_score,
        post_count=signal.post_count,
        reddit_score=signal.reddit_score,
        twitter_score=signal.twitter_score,
    )
    db.add(sig_record)
    db.commit()

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
    """
    Fetch historical stock price data + summary stats.
    
    Example: GET /stock/AAPL?days=30
    """
    ticker = ticker.upper()
    df = get_price_history(ticker, days=days)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {ticker}.")

    info = get_ticker_info(ticker)
    stats = get_summary_stats(df)

    return {
        "ticker": ticker,
        "info": info,
        "stats": stats,
        "prices": df.to_dict(orient="records"),
    }


@router.get("/history/{ticker}")
def sentiment_history(
    ticker: str,
    limit: int = Query(default=100, ge=10, le=500),
    db: Session = Depends(get_db),
):
    """
    Fetch stored sentiment records for a ticker from DB.
    
    Example: GET /history/AAPL?limit=100
    """
    ticker = ticker.upper()
    records = (
        db.query(SentimentRecord)
        .filter(SentimentRecord.ticker == ticker)
        .order_by(SentimentRecord.created_at.desc())
        .limit(limit)
        .all()
    )

    if not records:
        raise HTTPException(status_code=404, detail=f"No history found for {ticker}.")

    df = pd.DataFrame([{
        "text": r.text,
        "source": r.source,
        "score": r.score,
        "label": r.label,
        "confidence": r.confidence,
        "created_at": r.created_at,
    } for r in records])

    rolling = rolling_sentiment(df, window="6h")

    return {
        "ticker": ticker,
        "total_records": len(records),
        "rolling_sentiment": rolling.to_dict(orient="records"),
        "records": df.head(50).to_dict(orient="records"),
    }


@router.get("/signals/{ticker}")
def signal_history(
    ticker: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Fetch past signals generated for a ticker.
    
    Example: GET /signals/AAPL
    """
    ticker = ticker.upper()
    signals = (
        db.query(SignalRecord)
        .filter(SignalRecord.ticker == ticker)
        .order_by(SignalRecord.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "ticker": ticker,
        "signals": [{
            "signal": s.signal,
            "avg_score": s.avg_score,
            "weighted_score": s.weighted_score,
            "post_count": s.post_count,
            "reddit_score": s.reddit_score,
            "twitter_score": s.twitter_score,
            "created_at": s.created_at,
        } for s in signals]
    }
