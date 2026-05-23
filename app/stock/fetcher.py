"""
stock/fetcher.py - Stock price data via yFinance
Auto-detects Indian stocks and appends .NS suffix.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Known Indian tickers on NSE
# Any ticker not in this set will still try .NS suffix automatically
INDIAN_TICKERS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO",
    "TATAMOTORS", "ADANIENT", "BAJFINANCE", "SBIN", "ICICIBANK",
    "HINDUNILVR", "ITC", "MARUTI", "HCLTECH", "SUNPHARMA",
    "AXISBANK", "KOTAKBANK", "LT", "NTPC", "POWERGRID",
    "NATIONALUM", "SAIL", "ONGC", "BPCL", "GAIL",
    "TATASTEEL", "HINDALCO", "JSWSTEEL", "COALINDIA", "VEDL",
    "BHARTIARTL", "IDEA", "INFRATEL", "TECHM", "MPHASIS",
    "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON", "AUROPHARMA",
    "TITAN", "ASIANPAINT", "PIDILITIND", "BERGEPAINT", "WHIRLPOOL",
    "ULTRACEMCO", "AMBUJACEM", "ACC", "SHREECEM", "DALMIA",
    "HDFC", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIGI",
    "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "IRCTC",
    "DMART", "TRENT", "JUBLFOOD", "INDIGO", "SPICEJET",
    "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "IDFCFIRSTB",
    "RECLTD", "PFC", "IREDA", "NHPC", "SJVN",
    "HAL", "BEL", "BHEL", "BEML", "COCHINSHIP",
    "MRF", "APOLLOTYRE", "BALKRISIND", "CEATLTD",
    "PIDILITIND", "ASIANPAINT", "BERGERPAINTS",
}

# Ticker name aliases — maps common search terms to NSE symbols
TICKER_ALIASES = {
    "NATIONALALU": "NATIONALUM",
    "NALU": "NATIONALUM",
    "STATEBANKOFIN": "SBIN",
    "STATEBANK": "SBIN",
    "TATASTL": "TATASTEEL",
    "BAJAJFIN": "BAJFINANCE",
    "HDFCBK": "HDFCBANK",
    "ICICI": "ICICIBANK",
}


def normalize_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    # Apply aliases
    t = TICKER_ALIASES.get(t, t)
    if "." in t:
        return t
    if t in INDIAN_TICKERS:
        return f"{t}.NS"
    return t


def get_price_history(ticker: str, days: int = 30) -> pd.DataFrame:
    t = ticker.upper().strip()
    t = TICKER_ALIASES.get(t, t)  # resolve aliases

    # Build list of tickers to try in order
    if "." in t:
        candidates = [t]
    elif t in INDIAN_TICKERS:
        candidates = [f"{t}.NS", f"{t}.BO"]
    else:
        # Unknown ticker — try NSE, BSE, then as-is (for US stocks like AAPL)
        candidates = [f"{t}.NS", f"{t}.BO", t]

    for yticker in candidates:
        df = _download(yticker, days)
        if not df.empty:
            return df

    return pd.DataFrame()


def _download(yticker: str, days: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=days + 5)
    try:
        df = yf.download(yticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in needed:
            if col not in df.columns:
                return pd.DataFrame()
        df = df[needed].copy()
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            return pd.DataFrame()
        closes = df["Close"].values.astype(np.float64)
        pct = np.concatenate([[np.nan], np.diff(closes) / closes[:-1] * 100])
        df["pct_change"] = np.round(pct, 4)
        df["direction"] = np.where(pct >= 0, "up", "down")
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = np.round(df[col].values.astype(np.float64), 2)
        df["Volume"] = df["Volume"].fillna(0).astype(np.int64)
        df["Date"] = df["Date"].astype(str)
        df = df.where(pd.notnull(df), None)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"[yFinance] {yticker}: {e}")
        return pd.DataFrame()


def get_ticker_info(ticker: str) -> dict:
    t = ticker.upper().strip()
    if "." in t:
        yticker = t
    elif t in INDIAN_TICKERS:
        yticker = f"{t}.NS"
    else:
        yticker = f"{t}.NS"  # try NSE first for unknown

    currency = "INR" if yticker.endswith(".NS") or yticker.endswith(".BO") else "USD"
    sym = "₹" if currency == "INR" else "$"

    try:
        stock = yf.Ticker(yticker)
        fi = stock.fast_info
        high52 = getattr(fi, "year_high", None)
        low52 = getattr(fi, "year_low", None)
        price = getattr(fi, "last_price", None)
        mc = getattr(fi, "market_cap", None)

        # If fast_info returns nothing, try US ticker
        if not price and not yticker.endswith((".NS", ".BO")):
            currency = "USD"
            sym = "$"

        if mc and currency == "INR":
            mc_display = f"₹{round(mc/1e7):,} Cr"
        elif mc:
            mc_display = f"${mc/1e9:.1f}B"
        else:
            mc_display = "N/A"

        try:
            info = stock.info
            name = info.get("longName") or info.get("shortName") or ticker
            sector = info.get("sector") or info.get("industry") or "N/A"
            pe = round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A"
        except:
            name = ticker
            sector = "N/A"
            pe = "N/A"

        return {
            "name": name,
            "sector": sector,
            "market_cap": mc_display,
            "current_price": f"{sym}{round(price, 2)}" if price else "N/A",
            "52w_high": f"{sym}{round(high52, 2)}" if high52 else "N/A",
            "52w_low": f"{sym}{round(low52, 2)}" if low52 else "N/A",
            "pe_ratio": pe,
            "currency": currency,
            "exchange": "NSE" if yticker.endswith(".NS") else "BSE" if yticker.endswith(".BO") else "NYSE/NASDAQ",
        }
    except Exception as e:
        print(f"[Info] Error {yticker}: {e}")
        return {"name": ticker, "currency": currency, "exchange": "NSE"}


def get_summary_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    closes = df["Close"].values.astype(np.float64)
    closes = closes[~np.isnan(closes)]
    if len(closes) == 0:
        return {}
    return {
        "mean_price": round(float(np.mean(closes)), 2),
        "std_price": round(float(np.std(closes)), 2),
        "min_price": round(float(np.min(closes)), 2),
        "max_price": round(float(np.max(closes)), 2),
        "price_range": round(float(np.ptp(closes)), 2),
        "volatility_pct": round(float(np.std(closes) / np.mean(closes) * 100), 2),
    }
