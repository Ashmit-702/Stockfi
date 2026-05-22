"""
stock/fetcher.py - Fetches Indian & global stock price data using yFinance
Indian stocks use .NS (NSE) or .BO (BSE) suffix.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Popular Indian stocks for quick access
POPULAR_INDIAN_STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "WIPRO": "WIPRO.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "ADANIENT": "ADANIENT.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "SBIN": "SBIN.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "MARUTI": "MARUTI.NS",
    "HCLTECH": "HCLTECH.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
}


def normalize_ticker(ticker: str) -> str:
    """Auto-append .NS for known Indian stocks if no exchange suffix given."""
    t = ticker.upper().strip()
    if "." in t:
        return t  # already has suffix
    if t in POPULAR_INDIAN_STOCKS:
        return POPULAR_INDIAN_STOCKS[t]
    return t


def get_price_history(ticker: str, days: int = 30) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    close_vals = df["Close"].values.astype(np.float64)
    pct_changes = np.concatenate([[np.nan], np.diff(close_vals) / close_vals[:-1] * 100])
    df["pct_change"] = np.round(pct_changes, 4)
    df["direction"] = np.where(pct_changes >= 0, "up", "down")

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = np.round(df[col].values.astype(np.float64), 2)

    df["Volume"] = df["Volume"].fillna(0).astype(np.int64)
    df["Date"] = df["Date"].astype(str)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df


def get_ticker_info(ticker: str) -> dict:
    ticker = normalize_ticker(ticker)
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Format market cap in Indian notation (Cr/Lakh)
        market_cap = info.get("marketCap", None)
        if market_cap and ".NS" in ticker:
            market_cap_cr = round(market_cap / 1e7, 2)
            market_cap_display = f"₹{market_cap_cr:,.0f} Cr"
        elif market_cap:
            market_cap_display = f"${market_cap/1e9:.1f}B"
        else:
            market_cap_display = "N/A"

        currency = info.get("currency", "USD")
        symbol = "₹" if currency == "INR" else "$"

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": market_cap_display,
            "current_price": f"{symbol}{info.get('currentPrice', 'N/A')}",
            "52w_high": f"{symbol}{info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52w_low": f"{symbol}{info.get('fiftyTwoWeekLow', 'N/A')}",
            "pe_ratio": info.get("trailingPE", "N/A"),
            "currency": currency,
            "exchange": info.get("exchange", "N/A"),
        }
    except Exception as e:
        print(f"[yFinance] Info error for {ticker}: {e}")
        return {"name": ticker}


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
