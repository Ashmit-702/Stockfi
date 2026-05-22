"""
stock/fetcher.py - Stock price data
Uses yFinance with .NS suffix for Indian stocks.
NSE API used only for live quote info.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

INDIAN_TICKERS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO",
    "TATAMOTORS", "ADANIENT", "BAJFINANCE", "SBIN", "ICICIBANK",
    "HINDUNILVR", "ITC", "MARUTI", "HCLTECH", "SUNPHARMA",
    "AXISBANK", "KOTAKBANK", "LT", "NTPC", "POWERGRID",
}


def normalize_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if "." in t:
        return t
    if t in INDIAN_TICKERS:
        return f"{t}.NS"
    return t


def get_price_history(ticker: str, days: int = 30) -> pd.DataFrame:
    yticker = normalize_ticker(ticker)
    end = datetime.today()
    start = end - timedelta(days=days + 5)
    try:
        df = yf.download(yticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        closes = df["Close"].values.astype(np.float64)
        pct = np.concatenate([[np.nan], np.diff(closes) / closes[:-1] * 100])
        df["pct_change"] = np.round(pct, 4)
        df["direction"] = np.where(pct >= 0, "up", "down")
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = np.round(df[col].values.astype(np.float64), 2)
        df["Volume"] = df["Volume"].fillna(0).astype(np.int64)
        df["Date"] = df["Date"].astype(str)
        # Replace NaN with None for JSON serialization
        df = df.where(pd.notnull(df), None)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"[yFinance] Error {yticker}: {e}")
        return pd.DataFrame()


def get_ticker_info(ticker: str) -> dict:
    yticker = normalize_ticker(ticker)
    try:
        info = yf.Ticker(yticker).info
        currency = info.get("currency", "USD")
        sym = "₹" if currency == "INR" else "$"
        mc = info.get("marketCap")
        if mc and currency == "INR":
            mc_display = f"₹{round(mc/1e7):,} Cr"
        elif mc:
            mc_display = f"${mc/1e9:.1f}B"
        else:
            mc_display = "N/A"
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": mc_display,
            "current_price": f"{sym}{info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}",
            "52w_high": f"{sym}{info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52w_low": f"{sym}{info.get('fiftyTwoWeekLow', 'N/A')}",
            "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
            "currency": currency,
            "exchange": info.get("exchange", "NSE"),
        }
    except Exception as e:
        print(f"[Info] Error {yticker}: {e}")
        return {"name": ticker, "currency": "INR" if yticker.endswith(".NS") else "USD"}


def get_summary_stats(df: pd.DataFrame) -> dict:
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
