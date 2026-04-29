"""
Market data engine — Indian (NSE/BSE) + US markets via yfinance.
Tickers: NSE → append .NS  |  BSE → append .BO
Fixes: NSE holiday calendar 2025-2026, price validity flag, NaN-safe returns.
"""

import yfinance as yf
import pandas as pd
from datetime import date, datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── NSE Holiday Calendar 2025 & 2026 ──────────────────────────────────
# Source: NSE India official holiday list
NSE_HOLIDAYS = {
    # 2025
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 21),  # Diwali – Balipratipada
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Mahashivratri
    date(2026, 3, 20),   # Holi
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 19),  # Diwali – Balipratipada (approx)
    date(2026, 11, 24),  # Gurunanak Jayanti
    date(2026, 12, 25),  # Christmas
}


def is_market_open() -> dict:
    """
    Check if NSE is currently open.
    Hours: 9:15 AM – 3:30 PM IST, Monday–Friday, excluding NSE holidays.
    """
    now      = datetime.now(IST)
    today    = now.date()
    weekday  = now.weekday()          # 0 = Mon … 6 = Sun
    mins     = now.hour * 60 + now.minute
    open_min = 9 * 60 + 15            # 9:15
    close_min= 15 * 60 + 30           # 15:30

    is_holiday = today in NSE_HOLIDAYS
    is_weekday = weekday < 5
    in_hours   = open_min <= mins <= close_min
    pre_open   = (9 * 60) <= mins < open_min

    is_open = is_weekday and in_hours and not is_holiday
    pre     = is_weekday and pre_open and not is_holiday

    status = "OPEN" if is_open else ("PRE-OPEN" if pre else
             ("HOLIDAY" if is_holiday else "CLOSED"))
    return {
        "is_open":    is_open,
        "pre_open":   pre,
        "is_holiday": is_holiday,
        "time_ist":   now.strftime("%H:%M:%S IST"),
        "date":       now.strftime("%d %b %Y, %A"),
        "status":     status,
    }


def nse(ticker: str) -> str:
    """Append .NS suffix for NSE."""
    return ticker.upper() + ".NS" if not ticker.upper().endswith(".NS") else ticker.upper()


def get_stock_info(ticker: str) -> dict:
    """
    Comprehensive stock info.
    Uses None (not 0) for fundamental metrics where 0 is a valid value,
    so callers can distinguish 'missing data' from '0%'.
    Adds price_valid flag — callers should skip if False.
    """
    stock = yf.Ticker(ticker)
    info  = stock.info

    price = (info.get("currentPrice")
             or info.get("regularMarketPrice")
             or info.get("previousClose")
             or None)
    prev = info.get("previousClose") or None

    change     = (price - prev) if (price and prev) else 0.0
    change_pct = (change / prev * 100) if prev else 0.0
    currency   = info.get("currency") or "INR"

    # Use None for metrics where 0 ≠ "no data"
    def _num(key, fallback=None):
        v = info.get(key)
        return v if (v is not None and v == v) else fallback  # v==v rejects NaN

    return {
        "ticker":        ticker.upper(),
        "name":          info.get("longName") or info.get("shortName") or ticker,
        "sector":        info.get("sector") or "N/A",
        "industry":      info.get("industry") or "N/A",
        "market_cap":    _num("marketCap", 0),
        "price":         price or 0,
        "price_valid":   price is not None and price > 0,
        "previous_close":prev or 0,
        "change":        round(change, 2),
        "change_pct":    round(change_pct, 2),
        "open":          _num("open", _num("regularMarketOpen", 0)),
        "day_high":      _num("dayHigh", _num("regularMarketDayHigh", 0)),
        "day_low":       _num("dayLow",  _num("regularMarketDayLow",  0)),
        "volume":        _num("volume",  _num("regularMarketVolume",  0)),
        "avg_volume":    _num("averageVolume", 0),
        "52w_high":      _num("fiftyTwoWeekHigh"),
        "52w_low":       _num("fiftyTwoWeekLow"),
        # Fundamentals: None means "no data", 0 means genuinely zero
        "pe_ratio":      _num("trailingPE"),       # None if unavailable / negative PE
        "forward_pe":    _num("forwardPE"),
        "eps":           _num("trailingEps"),
        "dividend_yield":_num("dividendYield"),
        "beta":          _num("beta"),
        "profit_margin": _num("profitMargins"),    # None if unpublished
        "revenue_growth":_num("revenueGrowth"),    # None if not reported
        "target_mean":   _num("targetMeanPrice"),
        "target_high":   _num("targetHighPrice"),
        "target_low":    _num("targetLowPrice"),
        "recommendation":info.get("recommendationKey") or "N/A",
        "analyst_count": _num("numberOfAnalystOpinions", 0),
        "short_ratio":   _num("shortRatio"),
        "currency":      currency,
        "symbol":        "₹" if currency == "INR" else "$",
    }


def get_historical_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df


def get_intraday_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d", interval="5m")
    return df


def get_multiple_quotes(tickers: list) -> list:
    results = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            price = info.get("currentPrice", info.get("regularMarketPrice",
                             info.get("previousClose", 0))) or 0
            prev = info.get("previousClose", 0) or 0
            change = price - prev
            change_pct = (change / prev * 100) if prev else 0
            results.append({
                "ticker": t,
                "name": info.get("shortName", info.get("longName", t)),
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "volume": info.get("volume", info.get("regularMarketVolume", 0)) or 0,
                "currency": info.get("currency", "INR"),
            })
        except Exception:
            results.append({"ticker": t, "name": t, "price": 0,
                            "change": 0, "change_pct": 0, "volume": 0, "currency": "INR"})
    return results


# ── Indian Market Indices ──────────────────────────────────────────────
INDIAN_INDICES = {
    "^NSEI":    "Nifty 50",
    "^BSESN":   "Sensex",
    "^NSEBANK": "Bank Nifty",
    "^CNXIT":   "Nifty IT",
    "^CNXPHARMA": "Nifty Pharma",
    "^CNXFMCG": "Nifty FMCG",
    "INDIA VIX.NS": "India VIX",
}

# US indices for global context
US_INDICES = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^DJI":  "Dow Jones",
}


def get_market_overview() -> dict:
    """Fetch Indian indices + global context."""
    india = []
    for sym, name in INDIAN_INDICES.items():
        try:
            s = yf.Ticker(sym)
            info = s.info
            price = info.get("regularMarketPrice", info.get("previousClose", 0)) or 0
            prev = info.get("previousClose", 0) or 0
            change = price - prev
            pct = (change / prev * 100) if prev else 0
            india.append({"name": name, "symbol": sym,
                          "price": round(price, 2), "change": round(change, 2),
                          "change_pct": round(pct, 2)})
        except Exception:
            india.append({"name": name, "symbol": sym, "price": 0, "change": 0, "change_pct": 0})

    global_ctx = []
    for sym, name in US_INDICES.items():
        try:
            s = yf.Ticker(sym)
            info = s.info
            price = info.get("regularMarketPrice", info.get("previousClose", 0)) or 0
            prev = info.get("previousClose", 0) or 0
            change = price - prev
            pct = (change / prev * 100) if prev else 0
            global_ctx.append({"name": name, "symbol": sym,
                                "price": round(price, 2), "change": round(change, 2),
                                "change_pct": round(pct, 2)})
        except Exception:
            global_ctx.append({"name": name, "symbol": sym, "price": 0,
                                "change": 0, "change_pct": 0})

    return {
        "market_status": is_market_open(),
        "indian_indices": india,
        "global_indices": global_ctx,
    }
