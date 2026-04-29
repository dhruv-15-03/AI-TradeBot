"""
News engine — Indian + global financial news with sentiment analysis.
Sources: Economic Times, MoneyControl, LiveMint, Business Standard,
         Yahoo Finance, Google News, Finviz (for global context).
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from typing import Optional

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Indian news RSS feeds ──────────────────────────────────────────────
INDIAN_RSS = {
    "Economic Times Markets":
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021502.cms",
    "Economic Times Stocks":
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "LiveMint Markets":
        "https://www.livemint.com/rss/markets",
    "Business Standard":
        "https://www.business-standard.com/rss/markets-106.rss",
    "MoneyControl":
        "https://www.moneycontrol.com/rss/marketsnews.xml",
}

# ── Global macro RSS feeds ─────────────────────────────────────────────
GLOBAL_RSS = {
    "Reuters Business":
        "https://feeds.reuters.com/reuters/businessNews",
    "Bloomberg Markets":
        "https://feeds.bloomberg.com/markets/news.rss",
}


def _sentiment(text: str) -> dict:
    blob = TextBlob(text)
    p = blob.sentiment.polarity
    label = "BULLISH" if p > 0.12 else ("BEARISH" if p < -0.12 else "NEUTRAL")
    return {"polarity": round(p, 3), "label": label}


def _parse_feed(source_name: str, url: str, limit: int = 8) -> list:
    try:
        feed = feedparser.parse(url, request_headers=HEADERS)
        articles = []
        for entry in feed.entries[:limit]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            # Strip HTML from summary
            if summary:
                summary = BeautifulSoup(summary, "html.parser").get_text()[:250]
            sentiment = _sentiment(f"{title}. {summary}")
            articles.append({
                "source": source_name,
                "title": title,
                "summary": summary,
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "sentiment": sentiment,
            })
        return articles
    except Exception as e:
        return [{"source": source_name, "title": f"Feed unavailable: {e}",
                 "summary": "", "link": "", "published": "",
                 "sentiment": {"polarity": 0, "label": "NEUTRAL"}}]


def get_indian_market_news(limit: int = 10) -> list:
    """Fetch news from all Indian financial news sources."""
    articles = []
    for name, url in INDIAN_RSS.items():
        articles.extend(_parse_feed(name, url, limit=limit // 2 + 2))
    return articles[:limit * 2]


def get_global_context_news(limit: int = 6) -> list:
    """Fetch global macro news that impacts Indian markets."""
    articles = []
    for name, url in GLOBAL_RSS.items():
        articles.extend(_parse_feed(name, url, limit=limit))
    return articles[:limit]


def get_yahoo_news(ticker: str, limit: int = 8) -> list:
    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
           f"?s={ticker}&region=IN&lang=en-IN")
    return _parse_feed("Yahoo Finance", url, limit)


def get_google_news(query: str, limit: int = 8) -> list:
    url = (f"https://news.google.com/rss/search?q={query}+NSE+stock"
           f"&hl=en-IN&gl=IN&ceid=IN:en")
    return _parse_feed("Google News", url, limit)


def get_finviz_news(ticker: str, limit: int = 6) -> list:
    """Scrape Finviz (useful for US-listed Indian ADRs like INFY, WIT, IBN)."""
    base = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://finviz.com/quote.ashx?t={base}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find(id="news-table")
        if not table:
            return []
        articles = []
        for row in table.find_all("tr")[:limit]:
            a = row.find("a")
            if not a:
                continue
            title = a.get_text(strip=True)
            articles.append({
                "source": "Finviz",
                "title": title,
                "summary": "",
                "link": a.get("href", ""),
                "published": "",
                "sentiment": _sentiment(title),
            })
        return articles
    except Exception:
        return []


def get_all_news(ticker: str, limit_per_source: int = 6) -> dict:
    """
    Aggregate news for a stock ticker from all sources.
    Works for Indian tickers (RELIANCE.NS) and US ADRs.
    """
    # Clean ticker for search
    base = ticker.replace(".NS", "").replace(".BO", "")

    yahoo = get_yahoo_news(ticker, limit_per_source)
    google = get_google_news(base, limit_per_source)
    indian = _parse_feed("Economic Times", INDIAN_RSS["Economic Times Stocks"],
                          limit_per_source)
    # filter indian news relevant to this ticker
    indian = [a for a in indian if base.lower() in a["title"].lower()]

    all_articles = yahoo + google + indian

    # Deduplicate by title
    seen = set()
    unique = []
    for a in all_articles:
        key = a["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    polarities = [a["sentiment"]["polarity"] for a in unique
                  if a.get("sentiment", {}).get("polarity") is not None]
    avg = sum(polarities) / len(polarities) if polarities else 0
    bullish = sum(1 for p in polarities if p > 0.12)
    bearish = sum(1 for p in polarities if p < -0.12)
    neutral = len(polarities) - bullish - bearish
    overall = "BULLISH" if avg > 0.12 else ("BEARISH" if avg < -0.12 else "NEUTRAL")

    return {
        "ticker": ticker.upper(),
        "total_articles": len(unique),
        "articles": unique,
        "sentiment_summary": {
            "average_polarity": round(avg, 3),
            "overall": overall,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
        },
    }


def get_market_sentiment_score() -> dict:
    """
    Overall market sentiment from Indian news headlines.
    Returns a score useful for deciding whether it's a good day to trade.
    """
    articles = get_indian_market_news(limit=20)
    polarities = [a["sentiment"]["polarity"] for a in articles
                  if a.get("sentiment")]
    if not polarities:
        return {"score": 0, "label": "NEUTRAL", "articles": articles}
    avg = sum(polarities) / len(polarities)
    label = "BULLISH" if avg > 0.08 else ("BEARISH" if avg < -0.08 else "NEUTRAL")
    return {
        "score": round(avg, 3),
        "label": label,
        "bullish": sum(1 for p in polarities if p > 0.12),
        "bearish": sum(1 for p in polarities if p < -0.12),
        "total": len(polarities),
        "articles": articles[:15],
    }
