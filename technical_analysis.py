"""
Technical analysis engine — indicators + signal generation.
Improvements over v1:
  - Supertrend (high-weight, most popular on Indian platforms)
  - RSI divergence detection
  - Directional volume confirmation (bullish vs bearish volume spike)
  - OBV trend slope signal
  - VWAP signal (critical for intraday)
  - EMA-9 crossover signals
  - MACD histogram momentum (growing vs shrinking)
  - Candle pattern detection (hammer, engulfing, doji, shooting star)
  - Pivot calculated on PREVIOUS day's completed candle, not today's partial
  - SMA-200 gap fixed: callers now use 1y data (handled in stock_analyzer)
"""

import pandas as pd
import numpy as np
import ta


# ── Indicator Computation ──────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # Trend
    df["EMA_9"]   = ta.trend.ema_indicator(close, window=9)
    df["EMA_12"]  = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"]  = ta.trend.ema_indicator(close, window=26)
    df["SMA_20"]  = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"]  = ta.trend.sma_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)

    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()

    # ADX
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"]     = adx.adx()
    df["ADX_Pos"] = adx.adx_pos()
    df["ADX_Neg"] = adx.adx_neg()

    # Momentum
    df["RSI"]       = ta.momentum.rsi(close, window=14)
    df["RSI_7"]     = ta.momentum.rsi(close, window=7)
    df["Williams_R"]= ta.momentum.williams_r(high, low, close, lbp=14)
    df["CCI"]       = ta.trend.cci(high, low, close, window=20)
    df["ROC"]       = ta.momentum.roc(close, window=12)

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["ATR"] = atr.average_true_range()

    # Volume — safe_vol replaces zeros with NaN to prevent 0/0 in VWAP
    safe_vol             = volume.replace(0, np.nan)
    df["OBV"]            = ta.volume.on_balance_volume(close, volume)
    df["OBV_SMA_20"]     = df["OBV"].rolling(window=20).mean()
    df["VWAP"]           = (safe_vol * (high + low + close) / 3).cumsum() / safe_vol.cumsum()
    df["Volume_SMA_20"]  = volume.rolling(window=20).mean()
    df["Volume_Ratio"]   = volume / df["Volume_SMA_20"].replace(0, np.nan)

    # Supertrend (factor=3, period=10 — standard Indian retail settings)
    df = _compute_supertrend(df, period=10, multiplier=3.0)

    return df


def _compute_supertrend(df: pd.DataFrame, period: int = 10,
                         multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute Supertrend using numpy arrays (avoids pandas SettingWithCopyWarning
    and correctly handles NaN rows in ATR warm-up period).
    """
    close_arr = df["Close"].values.astype(float)
    high_arr  = df["High"].values.astype(float)
    low_arr   = df["Low"].values.astype(float)
    atr_arr   = df["ATR"].values.astype(float)
    n         = len(df)

    hl2         = (high_arr + low_arr) / 2
    upper_band  = hl2 + multiplier * atr_arr
    lower_band  = hl2 - multiplier * atr_arr
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    supertrend  = np.ones(n, dtype=bool)   # default bullish

    for i in range(1, n):
        # During ATR warm-up period ATR is NaN — propagate previous values
        if np.isnan(atr_arr[i]) or np.isnan(atr_arr[i - 1]):
            fu_prev = final_upper[i - 1] if not np.isnan(final_upper[i - 1]) else upper_band[i]
            fl_prev = final_lower[i - 1] if not np.isnan(final_lower[i - 1]) else lower_band[i]
            final_upper[i] = fu_prev
            final_lower[i] = fl_prev
            supertrend[i]  = supertrend[i - 1]
            continue

        # Upper band: only tighten (move lower), never widen unless price breaks above
        if upper_band[i] < final_upper[i - 1] or close_arr[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band: only tighten (move higher), never widen unless price breaks below
        if lower_band[i] > final_lower[i - 1] or close_arr[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction
        if close_arr[i] > final_upper[i - 1]:
            supertrend[i] = True
        elif close_arr[i] < final_lower[i - 1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i - 1]

    df["Supertrend_Bull"] = supertrend
    df["Supertrend_Line"] = np.where(supertrend, final_lower, final_upper)
    return df


# ── Candle Pattern Detection ───────────────────────────────────────────

def _detect_candle_patterns(df: pd.DataFrame) -> list:
    """Detect last-candle patterns. Returns list of (name, action, weight, detail) tuples."""
    if len(df) < 3:
        return []

    c0 = df.iloc[-1]   # today
    c1 = df.iloc[-2]   # yesterday
    c2 = df.iloc[-3]   # day before

    o0, h0, l0, cl0 = c0["Open"], c0["High"], c0["Low"], c0["Close"]
    o1, h1, l1, cl1 = c1["Open"], c1["High"], c1["Low"], c1["Close"]

    patterns = []
    body0    = abs(cl0 - o0)
    range0   = h0 - l0 if (h0 - l0) > 0 else 0.0001
    body1    = abs(cl1 - o1)

    # Doji (indecision) — body < 10% of range
    if body0 < range0 * 0.10:
        patterns.append(("Doji Candle", "HOLD", 1, "Indecision — wait for next candle direction"))

    # Hammer / Bullish pin bar — after downtrend, long lower wick >= 2× body, small upper wick
    lower_wick0 = min(o0, cl0) - l0
    upper_wick0 = h0 - max(o0, cl0)
    prev_trend_down = cl1 < c2["Close"]  # simple: yesterday lower than 2 days ago
    if (lower_wick0 >= 2 * body0 and upper_wick0 < body0 * 0.5
            and body0 > 0 and prev_trend_down):
        patterns.append(("Hammer (Bullish Pin)", "BUY", 3,
                          "Long lower wick after downtrend — buyers absorbing selling"))

    # Shooting Star — after uptrend, long upper wick >= 2× body
    prev_trend_up = cl1 > c2["Close"]
    if (upper_wick0 >= 2 * body0 and lower_wick0 < body0 * 0.5
            and body0 > 0 and prev_trend_up):
        patterns.append(("Shooting Star (Bearish)", "SELL", 3,
                          "Long upper wick after uptrend — sellers rejecting highs"))

    # Bullish Engulfing — bullish candle fully engulfs previous bearish candle
    if (cl0 > o0              # today bullish
            and cl1 < o1      # yesterday bearish
            and o0 < cl1      # today opens below yesterday close
            and cl0 > o1):    # today closes above yesterday open
        patterns.append(("Bullish Engulfing", "BUY", 4,
                          "Buyers completely overwhelmed sellers — strong reversal signal"))

    # Bearish Engulfing
    if (cl0 < o0
            and cl1 > o1
            and o0 > cl1
            and cl0 < o1):
        patterns.append(("Bearish Engulfing", "SELL", 4,
                          "Sellers completely overwhelmed buyers — strong reversal signal"))

    # Bullish Marubozu (strong momentum candle, almost no wicks)
    if (cl0 > o0
            and lower_wick0 < body0 * 0.1
            and upper_wick0 < body0 * 0.1
            and body0 > range0 * 0.8):
        patterns.append(("Bullish Marubozu", "BUY", 2,
                          "Strong full-bodied up candle — bulls in complete control"))

    # Bearish Marubozu
    if (cl0 < o0
            and lower_wick0 < body0 * 0.1
            and upper_wick0 < body0 * 0.1
            and body0 > range0 * 0.8):
        patterns.append(("Bearish Marubozu", "SELL", 2,
                          "Strong full-bodied down candle — bears in complete control"))

    return patterns


# ── RSI Divergence ─────────────────────────────────────────────────────

def _detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> list:
    """
    Detects bullish and bearish RSI divergence over the last `lookback` candles.
    Bullish: price makes lower low, RSI makes higher low.
    Bearish: price makes higher high, RSI makes lower high.
    """
    if len(df) < lookback + 5:
        return []

    recent = df.tail(lookback)
    close  = recent["Close"].values
    rsi    = recent["RSI"].values
    signals = []

    # Find local lows (price)
    price_lows  = [(i, close[i]) for i in range(2, len(close) - 2)
                   if close[i] < close[i-1] and close[i] < close[i+1]]
    # Find local highs (price)
    price_highs = [(i, close[i]) for i in range(2, len(close) - 2)
                   if close[i] > close[i-1] and close[i] > close[i+1]]

    # Bullish divergence: last two lows — price lower, RSI higher
    if len(price_lows) >= 2:
        i1, p1 = price_lows[-2]
        i2, p2 = price_lows[-1]
        r1, r2 = rsi[i1], rsi[i2]
        if p2 < p1 and r2 > r1 and not np.isnan(r1) and not np.isnan(r2):
            signals.append(("RSI Bullish Divergence", "BUY", 4,
                             f"Price made lower low (₹{p2:.2f} < ₹{p1:.2f}) "
                             f"but RSI made higher low ({r2:.1f} > {r1:.1f}) — reversal likely"))

    # Bearish divergence: last two highs — price higher, RSI lower
    if len(price_highs) >= 2:
        i1, p1 = price_highs[-2]
        i2, p2 = price_highs[-1]
        r1, r2 = rsi[i1], rsi[i2]
        if p2 > p1 and r2 < r1 and not np.isnan(r1) and not np.isnan(r2):
            signals.append(("RSI Bearish Divergence", "SELL", 4,
                             f"Price made higher high (₹{p2:.2f} > ₹{p1:.2f}) "
                             f"but RSI made lower high ({r2:.1f} < {r1:.1f}) — reversal likely"))

    return signals


# ── Main Signal Generator ──────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 50:
        return {"error": "Insufficient data"}

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    prev2  = df.iloc[-3]
    signals = []

    price      = float(latest["Close"])
    open_price = float(latest.get("Open", price))

    # ── 1. RSI ────────────────────────────────────────────────────────
    rsi   = float(latest.get("RSI", 50) or 50)
    rsi7  = float(latest.get("RSI_7", 50) or 50)
    rsi_p = float(prev.get("RSI", 50) or 50)

    if rsi < 30:
        signals.append(("RSI Oversold", "BUY", 3,
                         f"RSI={rsi:.1f} — deeply oversold, bounce probable"))
    elif rsi < 40:
        signals.append(("RSI Low", "BUY", 1, f"RSI={rsi:.1f} — approaching oversold"))
    elif rsi > 70:
        signals.append(("RSI Overbought", "SELL", 3,
                         f"RSI={rsi:.1f} — deeply overbought, pullback probable"))
    elif rsi > 60:
        signals.append(("RSI High", "SELL", 1, f"RSI={rsi:.1f} — approaching overbought"))
    else:
        signals.append(("RSI Neutral", "HOLD", 0, f"RSI={rsi:.1f}"))

    # RSI momentum flip (crossed 50 from below/above)
    if rsi_p < 50 <= rsi:
        signals.append(("RSI Crossed 50 Up", "BUY", 2, "RSI crossed above 50 — momentum turning bullish"))
    elif rsi_p > 50 >= rsi:
        signals.append(("RSI Crossed 50 Down", "SELL", 2, "RSI crossed below 50 — momentum turning bearish"))

    # Both RSI timeframes agree
    if rsi < 35 and rsi7 < 35:
        signals.append(("RSI Multi-TF Oversold", "BUY", 2,
                         f"Both RSI-14={rsi:.1f} and RSI-7={rsi7:.1f} oversold — high confidence"))
    elif rsi > 65 and rsi7 > 65:
        signals.append(("RSI Multi-TF Overbought", "SELL", 2,
                         f"Both RSI-14={rsi:.1f} and RSI-7={rsi7:.1f} overbought"))

    # ── 2. MACD ───────────────────────────────────────────────────────
    macd      = float(latest.get("MACD", 0) or 0)
    macd_sig  = float(latest.get("MACD_Signal", 0) or 0)
    macd_hist = float(latest.get("MACD_Hist", 0) or 0)
    macd_p    = float(prev.get("MACD", 0) or 0)
    macd_sp   = float(prev.get("MACD_Signal", 0) or 0)
    hist_p    = float(prev.get("MACD_Hist", 0) or 0)

    # Crossover (highest quality)
    if macd_p < macd_sp and macd > macd_sig:
        signals.append(("MACD Bullish Crossover", "BUY", 4,
                         "MACD line crossed above signal — trend change to bullish"))
    elif macd_p > macd_sp and macd < macd_sig:
        signals.append(("MACD Bearish Crossover", "SELL", 4,
                         "MACD line crossed below signal — trend change to bearish"))
    elif macd > macd_sig:
        signals.append(("MACD Bullish", "BUY", 1, "MACD above signal line"))
    else:
        signals.append(("MACD Bearish", "SELL", 1, "MACD below signal line"))

    # Histogram momentum: growing means strengthening
    if macd_hist > 0 and macd_hist > hist_p:
        signals.append(("MACD Histogram Growing", "BUY", 2,
                         f"Bullish histogram expanding ({hist_p:.4f} → {macd_hist:.4f})"))
    elif macd_hist < 0 and macd_hist < hist_p:
        signals.append(("MACD Histogram Falling", "SELL", 2,
                         f"Bearish histogram expanding ({hist_p:.4f} → {macd_hist:.4f})"))

    # ── 3. Moving Averages ────────────────────────────────────────────
    sma20  = float(latest.get("SMA_20", price) or price)
    sma50  = float(latest.get("SMA_50", price) or price)
    sma200 = latest.get("SMA_200")
    ema9   = float(latest.get("EMA_9", price) or price)
    ema9_p = float(prev.get("EMA_9", price) or price)
    prev_close = float(prev["Close"])

    # EMA-9 crossover (fast signal for day trading)
    if prev_close < ema9_p and price > ema9:
        signals.append(("EMA-9 Bullish Cross", "BUY", 3,
                         f"Price crossed above EMA-9 (₹{ema9:.2f}) — intraday bullish"))
    elif prev_close > ema9_p and price < ema9:
        signals.append(("EMA-9 Bearish Cross", "SELL", 3,
                         f"Price crossed below EMA-9 (₹{ema9:.2f}) — intraday bearish"))

    # MA alignment
    if price > sma20 > sma50:
        signals.append(("MA Bullish Stack", "BUY", 2, "Price > SMA20 > SMA50"))
    elif price < sma20 < sma50:
        signals.append(("MA Bearish Stack", "SELL", 2, "Price < SMA20 < SMA50"))

    if sma200 is not None and pd.notna(sma200):
        sma200 = float(sma200)
        if price > sma200:
            signals.append(("Above 200 SMA", "BUY", 2,
                             f"Above long-term trend (SMA200=₹{sma200:.2f})"))
        else:
            signals.append(("Below 200 SMA", "SELL", 2,
                             f"Below long-term trend (SMA200=₹{sma200:.2f})"))

    # Golden / Death Cross
    prev_sma50  = prev.get("SMA_50")
    prev_sma200 = prev.get("SMA_200")
    if (sma200 is not None and pd.notna(sma200)
            and prev_sma50 is not None and pd.notna(prev_sma50)
            and prev_sma200 is not None and pd.notna(prev_sma200)):
        p50, p200 = float(latest.get("SMA_50", 0)), sma200
        pp50, pp200 = float(prev_sma50), float(prev_sma200)
        if pp50 < pp200 and p50 > p200:
            signals.append(("GOLDEN CROSS", "STRONG BUY", 5,
                             "SMA-50 crossed above SMA-200 — major long-term bullish"))
        elif pp50 > pp200 and p50 < p200:
            signals.append(("DEATH CROSS", "STRONG SELL", 5,
                             "SMA-50 crossed below SMA-200 — major long-term bearish"))

    # ── 4. Supertrend ────────────────────────────────────────────────
    st_bull  = bool(latest.get("Supertrend_Bull", True))
    st_bull_p= bool(prev.get("Supertrend_Bull", True))
    st_line  = float(latest.get("Supertrend_Line", price) or price)

    if not st_bull_p and st_bull:
        signals.append(("Supertrend Flipped Bullish", "STRONG BUY", 5,
                         f"Supertrend turned green — trend reversal confirmed (line=₹{st_line:.2f})"))
    elif st_bull_p and not st_bull:
        signals.append(("Supertrend Flipped Bearish", "STRONG SELL", 5,
                         f"Supertrend turned red — trend reversal confirmed (line=₹{st_line:.2f})"))
    elif st_bull:
        signals.append(("Supertrend Bullish", "BUY", 3,
                         f"Supertrend green — uptrend intact (line=₹{st_line:.2f})"))
    else:
        signals.append(("Supertrend Bearish", "SELL", 3,
                         f"Supertrend red — downtrend intact (line=₹{st_line:.2f})"))

    # ── 5. Bollinger Bands ────────────────────────────────────────────
    bb_upper = float(latest.get("BB_Upper", price * 1.05) or price * 1.05)
    bb_lower = float(latest.get("BB_Lower", price * 0.95) or price * 0.95)
    bb_mid   = float(latest.get("BB_Middle", price) or price)

    if price <= bb_lower:
        signals.append(("BB Lower Band Touch", "BUY", 3,
                         f"Price at lower Bollinger Band — mean reversion expected"))
    elif price >= bb_upper:
        signals.append(("BB Upper Band Touch", "SELL", 3,
                         f"Price at upper Bollinger Band — mean reversion expected"))

    # BB squeeze (width < 20-period avg → breakout imminent)
    bb_width     = float(latest.get("BB_Width", 0) or 0)
    bb_width_avg = float(df["BB_Width"].rolling(20).mean().iloc[-1] or 0)
    if bb_width_avg > 0 and bb_width < bb_width_avg * 0.6:
        signals.append(("BB Squeeze", "ALERT", 2,
                         "Bollinger Bands tightening — major breakout imminent"))

    # ── 6. Stochastic ─────────────────────────────────────────────────
    stoch_k  = float(latest.get("Stoch_K", 50) or 50)
    stoch_d  = float(latest.get("Stoch_D", 50) or 50)
    stoch_kp = float(prev.get("Stoch_K", 50) or 50)
    stoch_dp = float(prev.get("Stoch_D", 50) or 50)

    if stoch_k < 20 and stoch_d < 20:
        # Bullish K/D cross inside oversold zone — high quality
        if stoch_kp < stoch_dp and stoch_k > stoch_d:
            signals.append(("Stochastic Bullish Cross (Oversold)", "BUY", 4,
                             f"K crossed D inside oversold zone — strong reversal"))
        else:
            signals.append(("Stochastic Oversold", "BUY", 2,
                             f"K={stoch_k:.1f}, D={stoch_d:.1f}"))
    elif stoch_k > 80 and stoch_d > 80:
        if stoch_kp > stoch_dp and stoch_k < stoch_d:
            signals.append(("Stochastic Bearish Cross (Overbought)", "SELL", 4,
                             f"K crossed D inside overbought zone — strong reversal"))
        else:
            signals.append(("Stochastic Overbought", "SELL", 2,
                             f"K={stoch_k:.1f}, D={stoch_d:.1f}"))

    # ── 7. ADX Trend Strength ─────────────────────────────────────────
    adx     = float(latest.get("ADX", 0) or 0)
    adx_pos = float(latest.get("ADX_Pos", 0) or 0)
    adx_neg = float(latest.get("ADX_Neg", 0) or 0)

    if adx > 25:
        if adx_pos > adx_neg:
            signals.append(("Strong Uptrend (ADX)", "BUY", 2,
                             f"ADX={adx:.1f} (>25), +DI={adx_pos:.1f} > -DI={adx_neg:.1f}"))
        else:
            signals.append(("Strong Downtrend (ADX)", "SELL", 2,
                             f"ADX={adx:.1f} (>25), -DI={adx_neg:.1f} > +DI={adx_pos:.1f}"))
    elif adx < 20:
        signals.append(("Weak Trend (ADX)", "HOLD", 0,
                         f"ADX={adx:.1f} — market consolidating, avoid trend trades"))

    # ── 8. VWAP ──────────────────────────────────────────────────────
    vwap = float(latest.get("VWAP", price) or price)
    if price > vwap * 1.005:
        signals.append(("Above VWAP", "BUY", 2,
                         f"Price ₹{price:.2f} above VWAP ₹{vwap:.2f} — intraday bullish"))
    elif price < vwap * 0.995:
        signals.append(("Below VWAP", "SELL", 2,
                         f"Price ₹{price:.2f} below VWAP ₹{vwap:.2f} — intraday bearish"))

    # ── 9. OBV Trend ─────────────────────────────────────────────────
    obv      = float(latest.get("OBV", 0) or 0)
    obv_sma  = float(latest.get("OBV_SMA_20", 0) or 0)
    obv_prev = float(prev.get("OBV", 0) or 0)
    obv_p_sma= float(prev.get("OBV_SMA_20", 0) or 0)

    if obv_sma > 0:
        if obv > obv_sma and obv_prev < obv_p_sma:
            signals.append(("OBV Crossed Above SMA", "BUY", 3,
                             "On Balance Volume turned positive — institutional accumulation"))
        elif obv < obv_sma and obv_prev > obv_p_sma:
            signals.append(("OBV Crossed Below SMA", "SELL", 3,
                             "On Balance Volume turned negative — institutional distribution"))
        elif obv > obv_sma:
            signals.append(("OBV Bullish", "BUY", 1, "OBV above its average — volume supporting uptrend"))
        else:
            signals.append(("OBV Bearish", "SELL", 1, "OBV below its average — volume not supporting"))

    # ── 10. Directional Volume Confirmation ───────────────────────────
    vol_ratio  = float(latest.get("Volume_Ratio", 1) or 1)
    today_bull = float(latest["Close"]) > float(latest.get("Open", latest["Close"]))
    if vol_ratio > 2:
        if today_bull:
            signals.append(("High Volume Up Day", "BUY", 3,
                             f"Volume {vol_ratio:.1f}× average on UP candle — strong buying"))
        else:
            signals.append(("High Volume Down Day", "SELL", 3,
                             f"Volume {vol_ratio:.1f}× average on DOWN candle — strong selling"))
    elif vol_ratio > 1.5:
        if today_bull:
            signals.append(("Elevated Volume Up", "BUY", 1,
                             f"Volume {vol_ratio:.1f}× average on up candle"))
        else:
            signals.append(("Elevated Volume Down", "SELL", 1,
                             f"Volume {vol_ratio:.1f}× average on down candle"))

    # ── 11. Williams %R ──────────────────────────────────────────────
    wr = float(latest.get("Williams_R", -50) or -50)
    if wr > -20:
        signals.append(("Williams %R Overbought", "SELL", 1, f"Williams %R = {wr:.1f}"))
    elif wr < -80:
        signals.append(("Williams %R Oversold", "BUY", 1, f"Williams %R = {wr:.1f}"))

    # ── 12. CCI ──────────────────────────────────────────────────────
    cci = float(latest.get("CCI", 0) or 0)
    if cci > 150:
        signals.append(("CCI Strongly Overbought", "SELL", 2, f"CCI={cci:.1f}"))
    elif cci > 100:
        signals.append(("CCI Overbought", "SELL", 1, f"CCI={cci:.1f}"))
    elif cci < -150:
        signals.append(("CCI Strongly Oversold", "BUY", 2, f"CCI={cci:.1f}"))
    elif cci < -100:
        signals.append(("CCI Oversold", "BUY", 1, f"CCI={cci:.1f}"))

    # ── 13. Candle Patterns ──────────────────────────────────────────
    signals.extend(_detect_candle_patterns(df))

    # ── 14. RSI Divergence ───────────────────────────────────────────
    signals.extend(_detect_rsi_divergence(df))

    # ── Aggregate Score ───────────────────────────────────────────────
    buy_score  = sum(s[2] for s in signals if s[1] in ("BUY", "STRONG BUY"))
    sell_score = sum(s[2] for s in signals if s[1] in ("SELL", "STRONG SELL"))
    net_score  = buy_score - sell_score
    total_pts  = sum(s[2] for s in signals if s[2] > 0 and s[1] != "HOLD")

    # Scale: each STRONG BUY flip is worth 5 pts, total possible ~50+ pts in ideal scenario
    if net_score >= 16:   action = "STRONG BUY"
    elif net_score >= 8:  action = "BUY"
    elif net_score <= -16:action = "STRONG SELL"
    elif net_score <= -8: action = "SELL"
    else:                 action = "HOLD"

    confidence = round(min(abs(net_score) / max(total_pts, 1) * 100, 100), 1)

    # Confluence bonus: how many independent indicator families agree?
    buy_families  = sum([
        any(s[1] in ("BUY","STRONG BUY") for s in signals if "RSI" in s[0]),
        any(s[1] in ("BUY","STRONG BUY") for s in signals if "MACD" in s[0]),
        any(s[1] in ("BUY","STRONG BUY") for s in signals if "Supertrend" in s[0]),
        any(s[1] in ("BUY","STRONG BUY") for s in signals if "MA" in s[0] or "EMA" in s[0]),
        any(s[1] in ("BUY","STRONG BUY") for s in signals if "Volume" in s[0] or "OBV" in s[0]),
    ])
    sell_families = sum([
        any(s[1] in ("SELL","STRONG SELL") for s in signals if "RSI" in s[0]),
        any(s[1] in ("SELL","STRONG SELL") for s in signals if "MACD" in s[0]),
        any(s[1] in ("SELL","STRONG SELL") for s in signals if "Supertrend" in s[0]),
        any(s[1] in ("SELL","STRONG SELL") for s in signals if "MA" in s[0] or "EMA" in s[0]),
        any(s[1] in ("SELL","STRONG SELL") for s in signals if "Volume" in s[0] or "OBV" in s[0]),
    ])

    bb_pos = 50.0
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pos = round((price - bb_lower) / bb_range * 100, 1)

    return {
        "signals": [{"name": s[0], "action": s[1], "weight": s[2], "detail": s[3]}
                     for s in signals],
        "buy_score":    buy_score,
        "sell_score":   sell_score,
        "net_score":    net_score,
        "action":       action,
        "confidence":   confidence,
        "confluence": {
            "buy_families":  buy_families,
            "sell_families": sell_families,
            "note": f"{buy_families}/5 indicator families bullish" if net_score > 0
                    else f"{sell_families}/5 indicator families bearish",
        },
        "indicators": {
            "rsi":          round(rsi, 2),
            "rsi7":         round(rsi7, 2),
            "macd":         round(macd, 4),
            "macd_hist":    round(macd_hist, 4),
            "adx":          round(adx, 2),
            "supertrend":   "BULL" if st_bull else "BEAR",
            "vwap":         round(vwap, 2),
            "bb_position":  bb_pos,
            "stoch_k":      round(stoch_k, 2),
            "cci":          round(cci, 1),
            "atr":          round(float(latest.get("ATR", 0) or 0), 4),
            "volume_ratio": round(vol_ratio, 2),
            "obv_trend":    "UP" if obv > obv_sma else "DOWN",
        },
    }


# ── Support & Resistance ───────────────────────────────────────────────

def get_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Pivot calculated on PREVIOUS COMPLETED day's candle (not today's partial).
    Local S/R from last 60 candles for robustness.
    """
    if df.empty or len(df) < window + 1:
        return {}

    # Use iloc[-2] = yesterday's completed candle for pivot
    prev_day   = df.iloc[-2]
    ph, pl, pc = float(prev_day["High"]), float(prev_day["Low"]), float(prev_day["Close"])
    pivot = (ph + pl + pc) / 3

    r1 = 2 * pivot - pl
    r2 = pivot + (ph - pl)
    r3 = ph + 2 * (pivot - pl)
    s1 = 2 * pivot - ph
    s2 = pivot - (ph - pl)
    s3 = pl - 2 * (ph - pivot)

    # Local extrema over last 60 candles (wider window → more reliable S/R)
    lookback = df.tail(60)
    loc_sup, loc_res = [], []
    highs, lows = lookback["High"], lookback["Low"]
    for i in range(2, len(lookback) - 2):
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            loc_sup.append(round(float(lows.iloc[i]), 2))
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
            loc_res.append(round(float(highs.iloc[i]), 2))

    # Cluster nearby levels within 0.5%
    def cluster(levels, pct=0.005):
        if not levels: return []
        lvls = sorted(set(levels))
        clustered, group = [], [lvls[0]]
        for lv in lvls[1:]:
            if lv <= group[-1] * (1 + pct):
                group.append(lv)
            else:
                clustered.append(round(sum(group)/len(group), 2))
                group = [lv]
        clustered.append(round(sum(group)/len(group), 2))
        return clustered

    return {
        "pivot":        round(pivot, 2),
        "resistance_1": round(r1, 2),
        "resistance_2": round(r2, 2),
        "resistance_3": round(r3, 2),
        "support_1":    round(s1, 2),
        "support_2":    round(s2, 2),
        "support_3":    round(s3, 2),
        "local_supports":    cluster(loc_sup)[-5:],
        "local_resistances": cluster(loc_res)[-5:],
    }
