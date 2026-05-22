"""
stock/fetcher.py - Indian stock data using NSE India API (free, no key)
Falls back to yFinance for non-Indian stocks.
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

POPULAR_INDIAN_STOCKS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO",
    "TATAMOTORS", "ADANIENT", "BAJFINANCE", "SBIN", "ICICIBANK",
    "HINDUNILVR", "ITC", "MARUTI", "HCLTECH", "SUNPHARMA",
    "AXISBANK", "KOTAKBANK", "LT", "NTPC", "POWERGRID",
}


def is_indian_stock(ticker: str) -> bool:
    t = ticker.upper().replace(".NS", "").replace(".BO", "")
    return t in POPULAR_INDIAN_STOCKS or ticker.endswith(".NS") or ticker.endswith(".BO")


def get_nse_session() -> requests.Session:
    """Create a session with NSE cookies."""
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
    except:
        pass
    return session


def get_price_history_nse(ticker: str, days: int = 30) -> pd.DataFrame:
    """Fetch historical data from NSE India API."""
    clean = ticker.replace(".NS", "").replace(".BO", "").upper()
    end = datetime.now()
    start = end - timedelta(days=days + 10)

    url = "https://www.nseindia.com/api/historical/cm/equity"
    params = {
        "symbol": clean,
        "series": '["EQ"]',
        "from": start.strftime("%d-%m-%Y"),
        "to": end.strftime("%d-%m-%Y"),
        "csv": "true",
    }

    session = get_nse_session()
    try:
        resp = session.get(url, headers=NSE_HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        if df.empty:
            return pd.DataFrame()

        # NSE column names vary — normalize
        df.columns = [c.strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "date" in cl: col_map[c] = "Date"
            elif "open" in cl: col_map[c] = "Open"
            elif "high" in cl: col_map[c] = "High"
            elif "low" in cl: col_map[c] = "Low"
            elif cl in ["close", "ltp", "last"]: col_map[c] = "Close"
            elif "volume" in cl or "traded qty" in cl: col_map[c] = "Volume"
        df = df.rename(columns=col_map)

        needed = ["Date", "Open", "High", "Low", "Close"]
        for col in needed:
            if col not in df.columns:
                return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

        closes = df["Close"].values.astype(np.float64)
        pct = np.concatenate([[np.nan], np.diff(closes) / closes[:-1] * 100])
        df["pct_change"] = np.round(pct, 4)
        df["direction"] = np.where(pct >= 0, "up", "down")
        df["Date"] = df["Date"].astype(str)
        return df

    except Exception as e:
        print(f"[NSE] Error for {clean}: {e}")
        return pd.DataFrame()


def get_price_history_yf(ticker: str, days: int = 30) -> pd.DataFrame:
    """Fallback: yFinance for non-Indian stocks."""
    try:
        import yfinance as yf
        end = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        closes = df["Close"].values.astype(np.float64)
        pct = np.concatenate([[np.nan], np.diff(closes) / closes[:-1] * 100])
        df["pct_change"] = np.round(pct, 4)
        df["direction"] = np.where(pct >= 0, "up", "down")
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = np.round(df[col].values.astype(np.float64), 2)
        df["Volume"] = df["Volume"].fillna(0).astype(np.int64)
        df["Date"] = df["Date"].astype(str)
        return df.dropna(subset=["Close"]).reset_index(drop=True)
    except Exception as e:
        print(f"[yFinance] Error {ticker}: {e}")
        return pd.DataFrame()


def get_price_history(ticker: str, days: int = 30) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    if is_indian_stock(ticker):
        df = get_price_history_nse(ticker, days)
        if not df.empty:
            return df
        # fallback to yfinance with .NS
        clean = ticker.replace(".NS", "").replace(".BO", "")
        return get_price_history_yf(f"{clean}.NS", days)
    return get_price_history_yf(ticker, days)


def get_ticker_info(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    clean = ticker.replace(".NS", "").replace(".BO", "")

    if is_indian_stock(ticker):
        try:
            session = get_nse_session()
            url = f"https://www.nseindia.com/api/quote-equity?symbol={clean}"
            resp = session.get(url, headers=NSE_HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("info", {})
                price_info = data.get("priceInfo", {})
                meta = data.get("metadata", {})
                ltp = price_info.get("lastPrice", "N/A")
                return {
                    "name": info.get("companyName", clean),
                    "sector": meta.get("industry", "N/A"),
                    "market_cap": "N/A",
                    "current_price": f"₹{ltp}",
                    "52w_high": f"₹{price_info.get('weekHighLow', {}).get('max', 'N/A')}",
                    "52w_low": f"₹{price_info.get('weekHighLow', {}).get('min', 'N/A')}",
                    "pe_ratio": meta.get("pdSymbolPe", "N/A"),
                    "currency": "INR",
                    "exchange": "NSE",
                }
        except Exception as e:
            print(f"[NSE Info] Error: {e}")

    # fallback yfinance
    try:
        import yfinance as yf
        yticker = f"{clean}.NS" if is_indian_stock(ticker) else ticker
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
            "current_price": f"{sym}{info.get('currentPrice', 'N/A')}",
            "52w_high": f"{sym}{info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52w_low": f"{sym}{info.get('fiftyTwoWeekLow', 'N/A')}",
            "pe_ratio": info.get("trailingPE", "N/A"),
            "currency": currency,
            "exchange": info.get("exchange", "N/A"),
        }
    except:
        return {"name": ticker, "currency": "INR" if is_indian_stock(ticker) else "USD"}


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
