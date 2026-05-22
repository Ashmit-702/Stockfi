"""
stock/fetcher.py - Fetches historical stock price data using yFinance
Returns clean Pandas DataFrames for use in the dashboard and correlation engine.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def get_price_history(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker.

    Args:
        ticker: Stock symbol e.g. 'AAPL'
        days: Number of past days to fetch

    Returns:
        pd.DataFrame with columns: [Date, Open, High, Low, Close, Volume, pct_change, direction]
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        print(f"[yFinance] Error fetching {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # --- NumPy: calculate % change and direction ---
    close_vals = df["Close"].values.astype(np.float64)
    pct_changes = np.concatenate([[np.nan], np.diff(close_vals) / close_vals[:-1] * 100])
    df["pct_change"] = np.round(pct_changes, 4)
    df["direction"] = np.where(pct_changes >= 0, "up", "down")

    # Round prices
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = np.round(df[col].values.astype(np.float64), 2)

    df["Volume"] = df["Volume"].astype(np.int64)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    return df


def get_ticker_info(ticker: str) -> dict:
    """
    Fetch basic info about a ticker (name, sector, market cap, etc.)

    Args:
        ticker: Stock symbol

    Returns:
        dict with basic info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", None),
            "current_price": info.get("currentPrice", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "pe_ratio": info.get("trailingPE", None),
        }
    except Exception as e:
        print(f"[yFinance] Info error for {ticker}: {e}")
        return {"name": ticker}


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary stats from price DataFrame using NumPy.

    Returns:
        dict with mean, std, min, max of Close prices
    """
    if df.empty:
        return {}

    closes = df["Close"].values.astype(np.float64)
    return {
        "mean_price": round(float(np.mean(closes)), 2),
        "std_price": round(float(np.std(closes)), 2),
        "min_price": round(float(np.min(closes)), 2),
        "max_price": round(float(np.max(closes)), 2),
        "price_range": round(float(np.ptp(closes)), 2),
        "volatility_pct": round(float(np.std(closes) / np.mean(closes) * 100), 2),
    }
