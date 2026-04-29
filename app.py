"""
Flask web server — hardened version.
Fixes: thread watchdog + auto-restart, lock race, api_analyze timeout,
       stale-cache flag, input validation.
"""

import sys, os, threading, time, logging
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, render_template, request
from market_data import get_market_overview, get_multiple_quotes, is_market_open
from news_engine import get_market_sentiment_score, get_all_news, get_indian_market_news
from stock_analyzer import quick_scan, full_stock_analysis, get_best_pick, WATCHLISTS, CAPITAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Cache ──────────────────────────────────────────────────────────────
_cache = {
    "overview": None,
    "signals":  None,
    "news":     None,
    "bestpick": None,
    "last_refresh": None,
    "is_stale": True,   # True until first successful refresh
}
_cache_lock = threading.Lock()

DEFAULT_SCAN = list(dict.fromkeys(
    WATCHLISTS["affordable"]
    + WATCHLISTS["banking"][:5]
    + WATCHLISTS["volatile"][:5]
))[:20]

# ── Background Refresh ─────────────────────────────────────────────────

def _one_refresh():
    """Single refresh cycle. Returns True on success."""
    try:
        log.info("Refreshing cache…")
        overview  = get_market_overview()
        signals   = quick_scan(DEFAULT_SCAN)
        sentiment = get_market_sentiment_score()
        bestpick  = get_best_pick(signals)
        ts        = datetime.now(IST).strftime("%H:%M:%S IST")

        with _cache_lock:
            _cache["overview"]     = overview
            _cache["signals"]      = signals
            _cache["news"]         = sentiment
            _cache["bestpick"]     = bestpick
            _cache["last_refresh"] = ts
            _cache["is_stale"]     = False

        top = signals[0]["ticker"] if signals else "—"
        verdict = signals[0].get("verdict", "—") if signals else "—"
        log.info("Cache refreshed. Top signal: %s — %s", top, verdict)
        return True
    except Exception as exc:
        log.error("Cache refresh failed: %s", exc, exc_info=True)
        # Mark stale so UI shows warning — do NOT wipe existing cache
        with _cache_lock:
            _cache["is_stale"] = True
        return False


def _refresh_loop():
    """Refresh loop — keeps running even after individual errors."""
    while True:
        _one_refresh()
        status     = is_market_open()
        sleep_secs = 5 * 60 if status["is_open"] else 15 * 60
        log.info("Next refresh in %d min.", sleep_secs // 60)
        time.sleep(sleep_secs)


def start_background_refresh():
    """Start daemon refresh thread + a watchdog that restarts it if it dies."""
    def watchdog():
        while True:
            t = threading.Thread(target=_refresh_loop, daemon=True, name="RefreshLoop")
            t.start()
            t.join()                        # blocks until thread dies
            log.warning("Refresh thread died — restarting in 30 s…")
            time.sleep(30)

    w = threading.Thread(target=watchdog, daemon=True, name="RefreshWatchdog")
    w.start()
    log.info("Background refresh watchdog started.")


# ── Input helpers ──────────────────────────────────────────────────────

_TICKER_RE = __import__("re").compile(r"^[A-Z0-9\-\.]{1,20}$")

def _sanitize_ticker(t: str) -> str | None:
    t = t.strip().upper()
    # Prevent double-suffix
    if t.endswith(".NS.NS") or t.endswith(".BO.BO"):
        t = t[:-3]
    if "." not in t:
        t = t + ".NS"
    return t if _TICKER_RE.match(t.replace(".NS", "").replace(".BO", "")) else None


def _get_cached(key: str, fallback_fn):
    """Return cached value; call fallback_fn if cache is empty."""
    with _cache_lock:
        val   = _cache[key]
        stale = _cache["is_stale"]
    if val is None:
        val = fallback_fn()
        with _cache_lock:
            _cache[key] = val
    return val, stale


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _cache_lock:
        last  = _cache["last_refresh"]
        stale = _cache["is_stale"]
    return jsonify({
        "market":       is_market_open(),
        "last_refresh": last or "Loading…",
        "is_stale":     stale,
        "capital":      CAPITAL,
    })


@app.route("/api/overview")
def api_overview():
    data, stale = _get_cached("overview", get_market_overview)
    return jsonify({**(data or {}), "is_stale": stale})


@app.route("/api/signals")
def api_signals():
    data, stale = _get_cached("signals", lambda: quick_scan(DEFAULT_SCAN))
    with _cache_lock:
        last = _cache["last_refresh"]
    return jsonify({
        "signals":      data or [],
        "last_refresh": last or "Loading…",
        "is_stale":     stale,
        "capital":      CAPITAL,
    })


@app.route("/api/news")
def api_news():
    data, stale = _get_cached("news", get_market_sentiment_score)
    return jsonify({**(data or {}), "is_stale": stale})


@app.route("/api/bestpick")
def api_bestpick():
    with _cache_lock:
        data  = _cache["bestpick"]
        stale = _cache["is_stale"]
        last  = _cache["last_refresh"]
    if data is None:
        # Compute on first request if cache hasn't warmed yet
        with _cache_lock:
            signals = _cache["signals"]
        data = get_best_pick(signals or [])
        with _cache_lock:
            _cache["bestpick"] = data
    return jsonify({**(data or {}), "is_stale": stale, "last_refresh": last or "Loading…"})


@app.route("/api/analyze/<ticker>")
def api_analyze(ticker: str):
    clean = _sanitize_ticker(ticker)
    if not clean:
        return jsonify({"error": f"Invalid ticker: {ticker}"}), 400

    # Run with a hard 30-second timeout via a thread
    result = {}
    exc    = {}

    def _run():
        try:
            result["data"] = full_stock_analysis(clean)
        except Exception as e:
            exc["err"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        return jsonify({"error": "Analysis timed out — try again in a moment."}), 504
    if "err" in exc:
        log.error("Analysis error for %s: %s", clean, exc["err"])
        return jsonify({"error": exc["err"]}), 500
    return jsonify(result["data"])


@app.route("/api/scan")
def api_scan():
    watchlist = request.args.get("watchlist", "affordable")
    custom    = request.args.get("tickers", "")

    if custom:
        raw     = [t for t in custom.replace(",", " ").split() if t][:15]  # max 15
        tickers = [_sanitize_ticker(t) for t in raw]
        tickers = [t for t in tickers if t]  # drop invalid
        if not tickers:
            return jsonify({"error": "No valid tickers provided."}), 400
    elif watchlist in WATCHLISTS:
        tickers = WATCHLISTS[watchlist]
    else:
        return jsonify({"error": f"Unknown watchlist: {watchlist}",
                        "valid": list(WATCHLISTS.keys())}), 400

    results = quick_scan(tickers)
    return jsonify({"watchlist": watchlist, "results": results})


@app.route("/api/quote")
def api_quote():
    raw = request.args.get("tickers", "")
    if not raw:
        return jsonify({"error": "Pass ?tickers=SBIN.NS,TCS.NS"}), 400
    tickers = [_sanitize_ticker(t) for t in raw.replace(",", " ").split() if t]
    tickers = [t for t in tickers if t][:20]
    if not tickers:
        return jsonify({"error": "No valid tickers."}), 400
    return jsonify(get_multiple_quotes(tickers))


@app.route("/api/news/<ticker>")
def api_ticker_news(ticker: str):
    clean = _sanitize_ticker(ticker)
    if not clean:
        return jsonify({"error": f"Invalid ticker: {ticker}"}), 400
    return jsonify(get_all_news(clean, limit_per_source=8))


@app.route("/api/watchlists")
def api_watchlists():
    return jsonify({k: v for k, v in WATCHLISTS.items()})


# ── Entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_background_refresh()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
