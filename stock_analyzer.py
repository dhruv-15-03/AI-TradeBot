"""
AI-powered stock scoring and trade recommendation system.
Optimized for Indian markets (NSE). Combines technicals + sentiment + fundamentals.
"""

from market_data import get_stock_info, get_historical_data
from technical_analysis import compute_all_indicators, generate_signals, get_support_resistance
from news_engine import get_all_news
import pandas as pd

# ── Indian NSE Watchlists ──────────────────────────────────────────────
WATCHLISTS = {
    "nifty50": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "WIPRO.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "HCLTECH.NS", "TECHM.NS",
        "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    ],
    "banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
        "BANKBARODA.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "INDUSINDBK.NS",
    ],
    "it_tech": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "OFSS.NS",
    ],
    "volatile": [
        "TATAMOTORS.NS", "TATASTEEL.NS", "BANKBARODA.NS", "HINDALCO.NS", "VEDL.NS",
        "IDEA.NS", "YESBANK.NS", "SUZLON.NS", "IRFC.NS", "NHPC.NS",
    ],
    "affordable": [  # Price typically under ₹500 — good for ₹4K budget
        "SBIN.NS", "BANKBARODA.NS", "IDEA.NS", "NHPC.NS", "IRFC.NS",
        "SUZLON.NS", "NTPC.NS", "ONGC.NS", "SAIL.NS", "NMDC.NS",
    ],
    "midcap": [
        "ADANIPORTS.NS", "ADANIENT.NS", "GODREJCP.NS", "PIIND.NS", "MUTHOOTFIN.NS",
        "AUROPHARMA.NS", "BALKRISIND.NS", "ASTRAL.NS", "ABFRL.NS", "CANBK.NS",
    ],
    "global_adr": [  # Indian companies listed in US
        "INFY", "WIT", "IBN", "HDB", "SIFY",
    ],
}

CAPITAL = 4000       # INR starting capital
MARGIN_FACTOR = 5    # Upstox/Groww intraday margin up to 5x
MAX_BUYING_POWER = CAPITAL * MARGIN_FACTOR  # ₹20,000 effective


def analyze_fundamentals(info: dict) -> dict:
    scores, details = [], []
    data_found = 0   # track how many metrics actually had data

    pe = info.get("pe_ratio")           # None = no data (not 0)
    if pe is not None:
        data_found += 1
        if pe <= 0:
            details.append(f"Negative/zero P/E — company may be loss-making")
        elif pe < 15:
            scores.append(3); details.append(f"Cheap P/E ({pe:.1f}) — undervalued")
        elif pe < 25:
            scores.append(1); details.append(f"Fair P/E ({pe:.1f})")
        elif pe < 40:
            scores.append(-1); details.append(f"High P/E ({pe:.1f}) — expensive")
        else:
            scores.append(-2); details.append(f"Very High P/E ({pe:.1f}) — overvalued")

    margin = info.get("profit_margin")  # None = not reported
    if margin is not None:
        data_found += 1
        if margin > 0.20:
            scores.append(2); details.append(f"Strong margin ({margin*100:.1f}%)")
        elif margin > 0.10:
            scores.append(1); details.append(f"Decent margin ({margin*100:.1f}%)")
        elif margin >= 0:
            scores.append(0); details.append(f"Thin margin ({margin*100:.1f}%)")
        else:
            scores.append(-2); details.append(f"Negative margin ({margin*100:.1f}%) — losing money")

    growth = info.get("revenue_growth")  # None = not reported
    if growth is not None:
        data_found += 1
        if growth > 0.20:
            scores.append(3); details.append(f"High revenue growth ({growth*100:.1f}%)")
        elif growth > 0.08:
            scores.append(2); details.append(f"Solid growth ({growth*100:.1f}%)")
        elif growth >= 0:
            scores.append(1); details.append(f"Slow growth ({growth*100:.1f}%)")
        else:
            scores.append(-2); details.append(f"Revenue declining ({growth*100:.1f}%)")

    price  = info.get("price") or 0
    target = info.get("target_mean")
    if target is not None and price > 0:
        data_found += 1
        upside = (target - price) / price * 100
        if upside > 20:
            scores.append(3); details.append(f"Analyst upside {upside:.1f}% (target ₹{target:.0f})")
        elif upside > 5:
            scores.append(1); details.append(f"Analyst upside {upside:.1f}% (target ₹{target:.0f})")
        elif upside > -5:
            scores.append(0); details.append(f"Analyst sees flat ({upside:.1f}%)")
        else:
            scores.append(-2); details.append(f"Analyst downside {upside:.1f}%")

    h52 = info.get("52w_high"); l52 = info.get("52w_low")
    if h52 and l52 and h52 > l52 and price:
        data_found += 1
        pos = (price - l52) / (h52 - l52) * 100
        if pos < 30:
            scores.append(2); details.append(f"Near 52W low ({pos:.0f}% of range)")
        elif pos > 90:
            scores.append(-1); details.append(f"Near 52W high ({pos:.0f}% of range)")
        else:
            details.append(f"At {pos:.0f}% of 52W range")

    total = sum(scores) if scores else 0
    return {
        "fundamental_score": total,
        "max_score": max(data_found * 3, 1),
        "data_found": data_found,
        "no_data": data_found == 0,
        "details": details,
    }


def calculate_position_size(price: float, stop_loss: float,
                             capital: float = CAPITAL, risk_pct: float = 0.02) -> dict:
    """
    Kelly/fixed-risk position sizer. Uses 2% of capital as maximum risk per trade.
    stop_loss must be BELOW price for BUY, ABOVE for SELL.
    Falls back to 2% of price as risk per share if stop_loss is invalid.
    """
    if not price or price <= 0:
        return {"shares": 0, "investment": 0, "risk_amount": 0, "uses_margin": False,
                "note": "Invalid price"}

    risk_budget   = capital * risk_pct                    # ₹80 for ₹4K capital
    risk_per_share = abs(price - stop_loss) if (stop_loss and stop_loss != price) else 0

    # Guard: if stop is too tight (< 0.3%) or invalid, use 2% of price
    if risk_per_share < price * 0.003:
        risk_per_share = price * 0.02

    shares_by_risk    = int(risk_budget / risk_per_share)
    shares_by_capital = int(MAX_BUYING_POWER / price)
    shares = min(shares_by_risk, shares_by_capital)
    shares = max(shares, 0)    # never force minimum; 0 means "too expensive"

    investment = round(shares * price, 2)
    return {
        "shares":      shares,
        "investment":  investment,
        "risk_amount": round(shares * risk_per_share, 2),
        "uses_margin": investment > capital,
        "note":        "OK" if shares > 0 else "Price exceeds buying power",
    }


def full_stock_analysis(ticker: str) -> dict:
    info = get_stock_info(ticker)
    hist = get_historical_data(ticker, period="1y", interval="1d")
    hist = compute_all_indicators(hist)
    tech_signals = generate_signals(hist)
    sr = get_support_resistance(hist)
    news = get_all_news(ticker, limit_per_source=5)
    fund = analyze_fundamentals(info)

    tech_score = tech_signals.get("net_score", 0)
    sent_pol   = news["sentiment_summary"]["average_polarity"]
    fund_score = fund["fundamental_score"]

    # Rescaled: new tech scoring can reach ±50, normalise to ±10
    tech_norm = max(min(tech_score / 5, 10), -10)
    sent_norm = max(min(sent_pol * 30, 10), -10)
    fund_norm = max(min(fund_score / 2, 10), -10)
    combined  = tech_norm * 0.45 + sent_norm * 0.25 + fund_norm * 0.30

    if combined >= 5:    verdict = "STRONG BUY"
    elif combined >= 2:  verdict = "BUY"
    elif combined >= -2: verdict = "HOLD"
    elif combined >= -5: verdict = "SELL"
    else:                verdict = "STRONG SELL"

    price    = info.get("price", 0)
    atr      = tech_signals.get("indicators", {}).get("atr", 0)
    risk_pct = (atr / price * 100) if price > 0 else 0
    risk_level = "HIGH" if risk_pct > 3 else ("MODERATE" if risk_pct > 1.5 else "LOW")

    tp       = _directional_trade_plan(price, sr, verdict)
    position = calculate_position_size(price, tp["stop_loss"])

    return {
        "ticker": ticker.upper(),
        "name": info.get("name", ticker),
        "price": price,
        "stock_info": info,
        "technical": tech_signals,
        "support_resistance": sr,
        "news": news,
        "fundamentals": fund,
        "combined_score": round(combined, 2),
        "verdict": verdict,
        "risk": {"level": risk_level, "daily_volatility_pct": round(risk_pct, 2)},
        "trade_plan": {**tp, "position": position},
    }


def _directional_trade_plan(price: float, sr: dict, verdict: str) -> dict:
    """
    Entry, stop-loss and target vary by signal direction.
    BUY  → enter near support, stop below next support, target resistance.
    SELL → enter near resistance, stop above next resistance, target support.
    """
    is_buy  = "BUY"  in verdict
    is_sell = "SELL" in verdict

    if is_buy:
        entry    = sr.get("support_1",    price * 0.98)
        stop     = sr.get("support_2",    price * 0.95)
        target_1 = sr.get("resistance_1", price * 1.04)
        target_2 = sr.get("resistance_2", price * 1.08)
    elif is_sell:
        entry    = sr.get("resistance_1", price * 1.02)
        stop     = sr.get("resistance_2", price * 1.05)
        target_1 = sr.get("support_1",   price * 0.96)
        target_2 = sr.get("support_2",   price * 0.92)
    else:  # HOLD — show current levels, no directional trade
        entry    = price
        stop     = sr.get("support_1",    price * 0.97)
        target_1 = sr.get("resistance_1", price * 1.03)
        target_2 = sr.get("resistance_2", price * 1.06)

    # Sanity: entry must be a real number close to current price
    entry    = float(entry)    if isinstance(entry,    (int, float)) else price
    stop     = float(stop)     if isinstance(stop,     (int, float)) else (price * 0.95 if is_buy else price * 1.05)
    target_1 = float(target_1) if isinstance(target_1, (int, float)) else (price * 1.04 if is_buy else price * 0.96)
    target_2 = float(target_2) if isinstance(target_2, (int, float)) else (price * 1.08 if is_buy else price * 0.92)

    # Risk-reward
    if is_buy:
        rr = round((target_1 - entry) / (entry - stop), 2) if (entry - stop) > 0 else 0
    elif is_sell:
        rr = round((entry - target_1) / (stop - entry), 2) if (stop - entry) > 0 else 0
    else:  # HOLD
        rr = 0

    return {
        "entry":    round(entry,    2),
        "stop_loss":round(stop,     2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "risk_reward": rr,
    }


def quick_scan(tickers: list) -> list:
    """
    Fast scan — technicals (all 14 signal families) + basic momentum sentiment.
    Uses 1y data so SMA-200 and Supertrend have enough history.
    Entry/stop/target are direction-aware (BUY vs SELL).
    """
    results = []
    for t in tickers:
        try:
            info  = get_stock_info(t)
            price = info.get("price", 0)
            if not info.get("price_valid", False) or not price:
                continue

            # 1y gives 252 candles — enough for SMA-200 + Supertrend warm-up
            hist = get_historical_data(t, period="1y", interval="1d")
            hist = compute_all_indicators(hist)
            tech = generate_signals(hist)
            sr   = get_support_resistance(hist)

            # Rescale: new scoring can reach ±50+, normalise to ±10
            net = tech.get("net_score", 0)
            tech_norm = max(min(net / 5, 10), -10)

            # Lightweight momentum sentiment: today's price vs 20-day SMA
            sma20_arr = hist["SMA_20"]
            sma20_val = float(sma20_arr.iloc[-1]) if not sma20_arr.empty and pd.notna(sma20_arr.iloc[-1]) else price
            mom_boost = 1.0 if price > sma20_val else -1.0   # small tilt, not a full sentiment fetch

            combined = tech_norm + mom_boost * 0.5

            if combined >= 5:    verdict = "STRONG BUY"
            elif combined >= 2:  verdict = "BUY"
            elif combined >= -2: verdict = "HOLD"
            elif combined >= -5: verdict = "SELL"
            else:                verdict = "STRONG SELL"

            # Direction-aware trade plan
            tp = _directional_trade_plan(price, sr, verdict)

            atr      = tech.get("indicators", {}).get("atr", 0)
            risk_pct = (atr / price * 100) if price > 0 else 0
            position = calculate_position_size(price, tp["stop_loss"])

            conf     = tech.get("confluence", {})
            families = conf.get("buy_families", 0) if "BUY" in verdict else conf.get("sell_families", 0)

            results.append({
                "ticker":       t.upper(),
                "name":         info.get("name", t),
                "price":        round(price, 2),
                "change":       info.get("change", 0),
                "change_pct":   info.get("change_pct", 0),
                "verdict":      verdict,
                "combined_score": round(combined, 2),
                "tech_action":  tech.get("action", "N/A"),
                "confidence":   tech.get("confidence", 0),
                "confluence":   f"{families}/5 families",
                "rsi":          round(tech.get("indicators", {}).get("rsi", 0), 1),
                "supertrend":   tech.get("indicators", {}).get("supertrend", "—"),
                "entry":        tp["entry"],
                "stop_loss":    tp["stop_loss"],
                "target_1":     tp["target_1"],
                "target_2":     tp["target_2"],
                "risk_reward":  tp["risk_reward"],
                "shares":       position["shares"],
                "investment":   position["investment"],
                "uses_margin":  position.get("uses_margin", False),
                "risk":         "HIGH" if risk_pct > 3 else ("MODERATE" if risk_pct > 1.5 else "LOW"),
                "affordable":   price <= MAX_BUYING_POWER,
            })
        except Exception as e:
            results.append({"ticker": t.upper(), "name": t, "error": str(e),
                            "price": 0, "verdict": "N/A", "combined_score": -999})

    # Sort: BUY signals first (by score desc), SELL signals last
    results.sort(key=lambda x: x.get("combined_score", -999), reverse=True)
    return results


def get_best_pick(scan_results: list) -> dict:
    """
    Select ONE clear, actionable recommendation from a quick_scan result list.

    Quality gates (all must pass):
      • verdict is BUY or STRONG BUY
      • confluence ≥ 3/5 indicator families (avoid weak signals)
      • confidence ≥ 25%
      • risk_reward ≥ 1.5  (reward at least 1.5× the risk)
      • RSI not overbought (< 75) — avoid chasing
      • price > 0 and affordable within MAX_BUYING_POWER

    Composite quality score:
      combined_score  ×  (confluence_families / 5)
                      ×  min(risk_reward / 2, 1.5)   # capped so insane RR doesn't dominate
                      ×  rsi_penalty                  # 1.0 normal, 0.7 if RSI 65-75 (hot)
                      ×  supertrend_multiplier         # 1.2 if Supertrend is BULL/BEAR aligned

    If no stock passes all gates → returns a WAIT signal.
    """
    WAIT = {
        "has_pick":    False,
        "verdict":     "WAIT",
        "reason":      "No clear opportunity — market is consolidating or signals are weak.",
    }

    if not scan_results:
        return WAIT

    candidates = []
    for s in scan_results:
        # Skip errored entries
        if s.get("error") or s.get("price", 0) <= 0:
            continue

        verdict = s.get("verdict", "")
        if "BUY" not in verdict:
            continue

        # Parse confluence "X/5 families" string → integer
        conf_str = s.get("confluence", "0/5 families")
        try:
            conf_fam = int(conf_str.split("/")[0])
        except (ValueError, IndexError):
            conf_fam = 0

        confidence  = s.get("confidence", 0)    # 0–100 %
        rr          = s.get("risk_reward", 0)
        rsi         = s.get("rsi", 50)
        price       = s.get("price", 0)
        combined    = s.get("combined_score", -999)
        supertrend  = str(s.get("supertrend", "")).upper()
        shares      = s.get("shares", 0)

        # ── Quality gates ──────────────────────────────────────────────
        if conf_fam < 3:           continue   # weak confluence
        if confidence < 25:        continue   # model not confident
        if rr < 1.5:               continue   # bad risk/reward
        if rsi >= 75:              continue   # overbought — avoid chasing
        if price <= 0:             continue
        if shares == 0:            continue   # can't afford even 1 share

        # ── Composite quality score ────────────────────────────────────
        rsi_penalty = 0.7 if rsi >= 65 else 1.0          # slightly hot
        st_mult     = 1.2 if "BULL" in supertrend else 1.0
        rr_factor   = min(rr / 2.0, 1.5)

        quality = combined * (conf_fam / 5.0) * rr_factor * rsi_penalty * st_mult
        candidates.append({**s, "_quality": quality, "_conf_fam": conf_fam})

    if not candidates:
        return WAIT

    # Pick the best
    best = max(candidates, key=lambda x: x["_quality"])

    # Human-readable action line
    shares = best.get("shares", 0)
    inv    = best.get("investment", 0)
    margin_note = " (uses intraday margin)" if best.get("uses_margin") else ""
    action_line = (
        f"Buy {shares} share{'s' if shares != 1 else ''} of {best['name']} "
        f"@ ₹{best['entry']:.2f} — invest ₹{inv:.0f}{margin_note}"
    )

    return {
        "has_pick":      True,
        "ticker":        best["ticker"],
        "name":          best["name"],
        "price":         best["price"],
        "verdict":       best["verdict"],
        "entry":         best["entry"],
        "stop_loss":     best["stop_loss"],
        "target_1":      best["target_1"],
        "target_2":      best.get("target_2", best["target_1"]),
        "risk_reward":   best["risk_reward"],
        "shares":        shares,
        "investment":    inv,
        "uses_margin":   best.get("uses_margin", False),
        "confidence":    best.get("confidence", 0),
        "confluence":    best.get("confluence", ""),
        "rsi":           best.get("rsi", 0),
        "supertrend":    best.get("supertrend", "—"),
        "combined_score":best.get("combined_score", 0),
        "quality_score": round(best["_quality"], 3),
        "action_line":   action_line,
        "reason":        (
            f"Confluence {best['confluence']} · "
            f"Confidence {best.get('confidence', 0):.0f}% · "
            f"R:R {best['risk_reward']} · "
            f"RSI {best.get('rsi', 0):.1f} · "
            f"Supertrend {best.get('supertrend', '—')}"
        ),
    }
