#!/usr/bin/env python3
"""
technicals_v2.py — Extended Stage 2 indicator computation.

Superset of the original technicals.py. Adds trend, momentum, volatility,
volume, price-structure, pattern, and statistical indicators requested by
the user, computed purely from daily OHLCV (yfinance) — no intraday/order
book data, so microstructure indicators (order flow imbalance, VPIN, book
depth) are structurally out of scope and are not attempted here.

Philosophy carried over from the original script: NUMBERS ONLY. Every
number is deterministic and reproducible from OHLCV. Interpretation,
synthesis-by-category, and the outlook/invalidation clause are written by
the model per references/technical-analysis.md +
technical-analysis-v2-addendum.md. Two exceptions produce genuinely
model-generated content rather than pure OHLCV transforms and are labeled
loudly wherever they appear: ARIMA (an explicit statistical price forecast,
included at the user's request, overriding this skill's normal "no
independent forecasts" rule — treat with real skepticism) and the
peak/trough chart-pattern heuristics (approximate pattern matching, not a
confirmed pattern).

DEPENDENCIES
    pip install yfinance pandas numpy scipy statsmodels arch scikit-learn hmmlearn joblib

USAGE
    python technicals_v2.py AAPL MSFT NVDA --json out.json
"""

import argparse
import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency: pip install yfinance")

SECTOR_ETFS = {
    "technology": "XLK", "energy": "XLE", "financial services": "XLF",
    "healthcare": "XLV", "industrials": "XLI", "utilities": "XLU",
    "consumer defensive": "XLP", "consumer cyclical": "XLY",
    "basic materials": "XLB", "real estate": "XLRE",
    "communication services": "XLC",
}

_cache: dict = {}


def history(ticker, days=750):
    if ticker in _cache:
        return _cache[ticker]
    try:
        df = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=True)
        df = df if not df.empty else None
    except Exception as exc:
        print(f"  ! {ticker}: {exc}", file=sys.stderr)
        df = None
    _cache[ticker] = df
    return df


def safe(fn, *a, **kw):
    """Run an indicator function; return None + note on failure rather than crash the run."""
    try:
        return fn(*a, **kw), None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------- trend ---

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def wma(s, period):
    weights = np.arange(1, period + 1)
    return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_ma(s, period=20):
    half = wma(s, max(period // 2, 1))
    full = wma(s, period)
    raw = 2 * half - full
    return wma(raw, max(int(np.sqrt(period)), 1))


def kama(close, period=10, fast=2, slow=30):
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0)
    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = pd.Series(index=close.index, dtype=float)
    seed = period
    out.iloc[seed] = close.iloc[:seed + 1].mean()
    for i in range(seed + 1, len(close)):
        prev = out.iloc[i - 1] if not pd.isna(out.iloc[i - 1]) else close.iloc[i - 1]
        out.iloc[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)
    return out


def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx_dmi(df, period=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, 1)  # true range, unsmoothed
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di, minus_di, adx


def parabolic_sar(df, af_step=0.02, af_max=0.2):
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)
    ep = np.zeros(n)
    af = np.zeros(n)
    trend[0] = 1 if (n > 1 and close[1] >= close[0]) else -1
    sar[0] = low[0] if trend[0] == 1 else high[0]
    ep[0] = high[0] if trend[0] == 1 else low[0]
    af[0] = af_step
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            s = prev_sar + af[i - 1] * (ep[i - 1] - prev_sar)
            s = min(s, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < s:
                trend[i], sar[i], ep[i], af[i] = -1, ep[i - 1], low[i], af_step
            else:
                trend[i], sar[i] = 1, s
                if high[i] > ep[i - 1]:
                    ep[i], af[i] = high[i], min(af[i - 1] + af_step, af_max)
                else:
                    ep[i], af[i] = ep[i - 1], af[i - 1]
        else:
            s = prev_sar + af[i - 1] * (ep[i - 1] - prev_sar)
            s = max(s, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > s:
                trend[i], sar[i], ep[i], af[i] = 1, ep[i - 1], high[i], af_step
            else:
                trend[i], sar[i] = -1, s
                if low[i] < ep[i - 1]:
                    ep[i], af[i] = low[i], min(af[i - 1] + af_step, af_max)
                else:
                    ep[i], af[i] = ep[i - 1], af[i - 1]
    return sar, trend


def ichimoku(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, senkou_a, senkou_b


def supertrend(df, period=10, mult=3):
    atr_val = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + mult * atr_val
    lowerband = hl2 - mult * atr_val
    close = df["Close"]
    n = len(df)
    final_upper, final_lower = upperband.copy(), lowerband.copy()
    direction = np.ones(n, dtype=int)
    for i in range(1, n):
        final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i - 1]) if close.iloc[i - 1] <= final_upper.iloc[i - 1] else upperband.iloc[i]
        final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i - 1]) if close.iloc[i - 1] >= final_lower.iloc[i - 1] else lowerband.iloc[i]
    for i in range(1, n):
        if close.iloc[i] > final_upper.iloc[i - 1]:
            direction[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    st = np.where(direction == 1, final_lower.values, final_upper.values)
    return st, direction


# ------------------------------------------------------------ momentum ---

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(df, k=14, d=3):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def stoch_rsi(close, period=14):
    r = rsi(close, period)
    low_min, high_max = r.rolling(period).min(), r.rolling(period).max()
    return (r - low_min) / (high_max - low_min) * 100


def williams_r(df, period=14):
    high_max = df["High"].rolling(period).max()
    low_min = df["Low"].rolling(period).min()
    return -100 * (high_max - df["Close"]) / (high_max - low_min)


def cci(df, period=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad)


def roc(close, period=10):
    return (close / close.shift(period) - 1) * 100


def tsi(close, long=25, short=13):
    pc = close.diff()
    dspc = pc.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    dsapc = pc.abs().ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    return 100 * dspc / dsapc


def awesome_oscillator(df):
    mid = (df["High"] + df["Low"]) / 2
    return mid.rolling(5).mean() - mid.rolling(34).mean()


def mfi(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = tp * df["Volume"]
    delta = tp.diff()
    pos_flow = money_flow.where(delta > 0, 0).rolling(period).sum()
    neg_flow = money_flow.where(delta < 0, 0).rolling(period).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


# ----------------------------------------------------------- volatility ---

def bollinger(close, period=20, k=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = mid + k * std, mid - k * std
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return upper, mid, lower, pct_b, bandwidth


def keltner(df, period=20, mult=2):
    mid = ema(df["Close"], period)
    a = atr(df, period)
    return mid + mult * a, mid, mid - mult * a


def donchian(df, period=20):
    return df["High"].rolling(period).max(), df["Low"].rolling(period).min()


# --------------------------------------------------------------- volume ---

def obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def ad_line(df):
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    return (clv * df["Volume"]).fillna(0).cumsum()


def vwap_anchored(df, anchor_idx):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    seg = df.iloc[anchor_idx:]
    tp_seg = tp.iloc[anchor_idx:]
    return float((tp_seg * seg["Volume"]).sum() / seg["Volume"].sum())


def cmf(df, period=20):
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    return (clv * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()


def klinger(df, fast=34, slow=55, signal=13):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    trend = np.sign(tp.diff()).fillna(0)
    dm = df["High"] - df["Low"]
    vf = df["Volume"] * trend * (2 * (dm / dm.rolling(2).mean().replace(0, np.nan)).clip(upper=5)).fillna(1).abs()
    kvo = ema(vf, fast) - ema(vf, slow)
    sig = ema(kvo, signal)
    return kvo, sig


# ------------------------------------------------------------ structure ---

def pivots(prev_high, prev_low, prev_close):
    h, l, c = prev_high, prev_low, prev_close
    rng = h - l
    classic_p = (h + l + c) / 3
    classic = {
        "P": classic_p, "R1": 2 * classic_p - l, "S1": 2 * classic_p - h,
        "R2": classic_p + rng, "S2": classic_p - rng,
        "R3": h + 2 * (classic_p - l), "S3": l - 2 * (h - classic_p),
    }
    fib_p = (h + l + c) / 3
    fib = {
        "P": fib_p, "R1": fib_p + 0.382 * rng, "S1": fib_p - 0.382 * rng,
        "R2": fib_p + 0.618 * rng, "S2": fib_p - 0.618 * rng,
        "R3": fib_p + 1.0 * rng, "S3": fib_p - 1.0 * rng,
    }
    camarilla = {
        "R1": c + rng * 1.1 / 12, "R2": c + rng * 1.1 / 6, "R3": c + rng * 1.1 / 4, "R4": c + rng * 1.1 / 2,
        "S1": c - rng * 1.1 / 12, "S2": c - rng * 1.1 / 6, "S3": c - rng * 1.1 / 4, "S4": c - rng * 1.1 / 2,
    }
    woodie_p = (h + l + 2 * c) / 4
    woodie = {"P": woodie_p, "R1": 2 * woodie_p - l, "S1": 2 * woodie_p - h,
              "R2": woodie_p + rng, "S2": woodie_p - rng}
    return {"classic": classic, "fibonacci": fib, "camarilla": camarilla, "woodie": woodie}


def fib_retracement(swing_high, swing_low):
    rng = swing_high - swing_low
    retr = {f"{p:.3f}": round(swing_high - rng * p, 2) for p in (0.236, 0.382, 0.5, 0.618, 0.786)}
    ext = {f"{p:.3f}": round(swing_high + rng * p, 2) for p in (0.272, 0.618, 1.0)}
    return {"retracements": retr, "extensions": ext}


def zigzag(close, pct=0.05):
    pivots_out = []
    last_price, last_idx, direction = close.iloc[0], 0, 0
    for i in range(1, len(close)):
        change = (close.iloc[i] - last_price) / last_price
        if direction >= 0 and change <= -pct:
            pivots_out.append((last_idx, round(float(last_price), 2), "high"))
            last_price, last_idx, direction = close.iloc[i], i, -1
        elif direction <= 0 and change >= pct:
            pivots_out.append((last_idx, round(float(last_price), 2), "low"))
            last_price, last_idx, direction = close.iloc[i], i, 1
        elif direction >= 0 and close.iloc[i] > last_price:
            last_price, last_idx = close.iloc[i], i
        elif direction <= 0 and close.iloc[i] < last_price:
            last_price, last_idx = close.iloc[i], i
    return pivots_out


# -------------------------------------------------------------- patterns ---

def candlestick_patterns(df, lookback=3):
    out = []
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    for i in range(len(df) - lookback, len(df)):
        if i < 1:
            continue
        flags = []
        if body.iloc[i] / rng.iloc[i] <= 0.1:
            flags.append("doji")
        if lower_wick.iloc[i] >= 2 * body.iloc[i] and upper_wick.iloc[i] <= 0.3 * body.iloc[i]:
            flags.append("hammer" if c.iloc[i] >= o.iloc[i] else "hanging_man")
        if upper_wick.iloc[i] >= 2 * body.iloc[i] and lower_wick.iloc[i] <= 0.3 * body.iloc[i]:
            flags.append("shooting_star")
        if c.iloc[i - 1] < o.iloc[i - 1] and c.iloc[i] > o.iloc[i] and c.iloc[i] > o.iloc[i - 1] and o.iloc[i] < c.iloc[i - 1]:
            flags.append("bullish_engulfing")
        if c.iloc[i - 1] > o.iloc[i - 1] and c.iloc[i] < o.iloc[i] and c.iloc[i] < o.iloc[i - 1] and o.iloc[i] > c.iloc[i - 1]:
            flags.append("bearish_engulfing")
        if flags:
            out.append({"date": str(df.index[i].date()), "patterns": flags})
    return out


def chart_pattern_heuristics(df, lookback=126, order=5):
    from scipy.signal import argrelextrema
    d = df.tail(lookback)
    highs, lows = d["High"].values, d["Low"].values
    hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    notes = []
    if len(hi_idx) >= 3:
        last3 = hi_idx[-3:]
        h3 = highs[last3]
        if h3[1] > h3[0] and h3[1] > h3[2] and abs(h3[0] - h3[2]) / h3[1] < 0.05:
            notes.append("possible head-and-shoulders (bearish) — heuristic peak match, not a confirmed pattern; needs a neckline break to mean anything")
    if len(lo_idx) >= 3:
        last3 = lo_idx[-3:]
        l3 = lows[last3]
        if l3[1] < l3[0] and l3[1] < l3[2] and abs(l3[0] - l3[2]) / l3[1] < 0.05:
            notes.append("possible inverse head-and-shoulders (bullish) — heuristic trough match, not a confirmed pattern; needs a neckline break to mean anything")
    if len(hi_idx) >= 2 and len(lo_idx) >= 2:
        hi_slope = np.polyfit(hi_idx[-3:], highs[hi_idx[-3:]], 1)[0] if len(hi_idx) >= 3 else np.polyfit(hi_idx[-2:], highs[hi_idx[-2:]], 1)[0]
        lo_slope = np.polyfit(lo_idx[-3:], lows[lo_idx[-3:]], 1)[0] if len(lo_idx) >= 3 else np.polyfit(lo_idx[-2:], lows[lo_idx[-2:]], 1)[0]
        if hi_slope < 0 and lo_slope > 0:
            notes.append("possible symmetrical triangle (converging highs/lows)")
        elif abs(hi_slope) < 0.05 * np.mean(highs) / len(d) and lo_slope > 0:
            notes.append("possible ascending triangle")
        elif hi_slope < 0 and abs(lo_slope) < 0.05 * np.mean(lows) / len(d):
            notes.append("possible descending triangle")
    recent = df.tail(15)
    move = (recent["Close"].iloc[-10] / recent["Close"].iloc[0] - 1) if len(recent) >= 10 else 0
    tail_range = (recent["Close"].iloc[-5:].max() - recent["Close"].iloc[-5:].min()) / recent["Close"].iloc[-1]
    if abs(move) > 0.10 and tail_range < 0.04:
        notes.append(f"possible {'bull' if move > 0 else 'bear'} flag/pennant — sharp {move*100:.1f}% move followed by a tight recent consolidation")
    return notes or ["no heuristic chart pattern flagged"]


# ----------------------------------------------------------------- stats ---

def zscore(close, period=20):
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - mean) / std


def hurst_exponent(close, max_lag=100):
    ts = np.asarray(close.values, dtype=float)
    max_lag = min(max_lag, len(ts) // 2)
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    tau = [t if t > 0 else 1e-8 for t in tau]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


def kalman_trend(close):
    x = np.array([close.iloc[0], 0.0])
    P = np.eye(2) * 1.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[max(np.var(np.diff(close.values)), 1e-6)]])
    for z in close.values:
        x = F @ x
        P = F @ P @ F.T + Q
        y = z - (H @ x)[0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T).flatten() / S
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H)) @ P
    return float(x[0]), float(x[1])


def garch_vol_forecast(close, horizon=5):
    from arch import arch_model
    returns = close.pct_change().dropna() * 100
    am = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    daily_vol = np.sqrt(fc.variance.values[-1, :])
    return {
        "next_day_daily_vol_pct": round(float(daily_vol[0]), 3),
        f"next_{horizon}d_avg_daily_vol_pct": round(float(daily_vol.mean()), 3),
        "annualized_next_day_vol_pct": round(float(daily_vol[0] * np.sqrt(252)), 2),
    }


def arima_forecast(close, steps=5, order=(5, 1, 0)):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(close.values, order=order)
    fit = model.fit()
    fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return {
        "order": list(order),
        "step_1_point": round(float(mean[0]), 2),
        "step_1_ci95": [round(float(ci[0][0]), 2), round(float(ci[0][1]), 2)],
        f"step_{steps}_point": round(float(mean[-1]), 2),
        f"step_{steps}_ci95": [round(float(ci[-1][0]), 2), round(float(ci[-1][1]), 2)],
        "disclaimer": "Statistical extrapolation of past price behavior only. Not a prediction, not investment advice, and not something this skill's own methodology would normally include — added at explicit user request, overriding the default 'no independent price forecasts' rule. Confidence intervals widen fast; treat step-5 especially loosely.",
    }


def regime_hmm(close):
    from hmmlearn.hmm import GaussianHMM
    returns = close.pct_change().dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(returns)
    hidden = model.predict(returns)
    means = model.means_.flatten()
    vols = np.sqrt(model.covars_.flatten())
    trending_state = int(np.argmax(np.abs(means)))
    current_state = int(hidden[-1])
    label = "trending" if current_state == trending_state else "choppy/mean-reverting"
    return {
        "current_regime": label,
        "state_means_daily_pct": [round(float(m * 100), 3) for m in means],
        "state_vols_daily_pct": [round(float(v * 100), 3) for v in vols],
        "note": "2-state Gaussian HMM on daily returns, refit each run on this ticker's own history — a lightweight approximation, not a validated regime model.",
    }


def wyckoff_heuristic(df, lookback=60):
    d = df.tail(lookback)
    close, vol = d["Close"], d["Volume"]
    range_pos = (close.iloc[-1] - d["Low"].min()) / (d["High"].max() - d["Low"].min())
    recent_vol_ratio = vol.tail(5).mean() / vol.mean()
    up_day = close.iloc[-1] > close.iloc[-2]
    if range_pos < 0.25 and recent_vol_ratio > 1.3 and up_day:
        phase = "possible Accumulation (spring-type volume pickup near range lows)"
    elif range_pos > 0.75 and recent_vol_ratio > 1.3 and not up_day:
        phase = "possible Distribution (climax-type volume near range highs, fading)"
    elif range_pos > 0.85 and recent_vol_ratio > 1.1 and up_day:
        phase = "possible Markup (breaking toward range highs on rising volume)"
    elif range_pos < 0.15 and recent_vol_ratio > 1.1 and not up_day:
        phase = "possible Markdown (breaking toward range lows on rising volume)"
    else:
        phase = "Unclear / mid-range consolidation"
    return {"phase": phase, "range_position_pct": round(float(range_pos * 100), 1),
            "note": "Heuristic only — a real Wyckoff read needs a human eye on the full schematic (springs, upthrusts, tests), not just range position and volume ratio."}


# ------------------------------------------------------------- assemble ---

def analyze(ticker, asof=None):
    df = history(ticker)
    if df is None or len(df) < 260:
        return {"ticker": ticker, "error": "insufficient price history for the extended indicator set (needs 260+ sessions)"}

    if asof:
        cut = pd.Timestamp(asof)
        if df.index.tz:
            cut = cut.tz_localize(df.index.tz)
        df = df[df.index <= cut]
        if len(df) < 260:
            return {"ticker": ticker, "error": f"insufficient history before {asof}"}

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(close.iloc[-1])
    out = {"ticker": ticker, "asof": str(df.index[-1].date()), "close": round(last, 2), "notes": []}

    # trend
    trend = {}
    trend["ema"] = {p: round(float(ema(close, p).iloc[-1]), 2) for p in (12, 26, 50, 200)}
    hma, err = safe(lambda: hull_ma(close, 20).iloc[-1]); trend["hull_ma_20"] = round(float(hma), 2) if hma is not None else None
    kv, err = safe(lambda: kama(close).iloc[-1]); trend["kama"] = round(float(kv), 2) if kv is not None else None
    (pdi, mdi, adx_v), err = safe(lambda: adx_dmi(df))
    if pdi is not None:
        trend["adx"] = round(float(adx_v.iloc[-1]), 1)
        trend["plus_di"] = round(float(pdi.iloc[-1]), 1)
        trend["minus_di"] = round(float(mdi.iloc[-1]), 1)
    (sar, sar_dir), err = safe(lambda: parabolic_sar(df))
    if sar is not None:
        trend["parabolic_sar"] = round(float(sar[-1]), 2)
        trend["parabolic_sar_dir"] = "up" if sar_dir[-1] == 1 else "down"
    (tenkan, kijun, sa, sb), err = safe(lambda: ichimoku(df))
    if tenkan is not None:
        cloud_top = max(float(sa.iloc[-1]), float(sb.iloc[-1])) if not (pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1])) else None
        cloud_bot = min(float(sa.iloc[-1]), float(sb.iloc[-1])) if cloud_top is not None else None
        trend["ichimoku"] = {
            "tenkan": round(float(tenkan.iloc[-1]), 2), "kijun": round(float(kijun.iloc[-1]), 2),
            "senkou_a": round(float(sa.iloc[-1]), 2) if not pd.isna(sa.iloc[-1]) else None,
            "senkou_b": round(float(sb.iloc[-1]), 2) if not pd.isna(sb.iloc[-1]) else None,
            "price_vs_cloud": ("above" if cloud_top and last > cloud_top else "below" if cloud_bot and last < cloud_bot else "inside") if cloud_top is not None else None,
        }
    (st, st_dir), err = safe(lambda: supertrend(df))
    if st is not None:
        trend["supertrend"] = round(float(st[-1]), 2)
        trend["supertrend_dir"] = "up" if st_dir[-1] == 1 else "down"
    out["trend"] = trend

    # momentum
    mom = {"rsi14": round(float(rsi(close).iloc[-1]), 1)}
    (k, dd), err = safe(lambda: stochastic(df))
    if k is not None:
        mom["stochastic_k"] = round(float(k.iloc[-1]), 1)
        mom["stochastic_d"] = round(float(dd.iloc[-1]), 1)
    sr, err = safe(lambda: stoch_rsi(close).iloc[-1]); mom["stoch_rsi"] = round(float(sr), 1) if sr is not None and not pd.isna(sr) else None
    wr, err = safe(lambda: williams_r(df).iloc[-1]); mom["williams_r"] = round(float(wr), 1) if wr is not None else None
    cc, err = safe(lambda: cci(df).iloc[-1]); mom["cci"] = round(float(cc), 1) if cc is not None else None
    mom["roc_10"] = round(float(roc(close, 10).iloc[-1]), 2)
    mom["roc_20"] = round(float(roc(close, 20).iloc[-1]), 2)
    ts, err = safe(lambda: tsi(close).iloc[-1]); mom["tsi"] = round(float(ts), 2) if ts is not None else None
    ao, err = safe(lambda: awesome_oscillator(df).iloc[-1]); mom["awesome_oscillator"] = round(float(ao), 3) if ao is not None else None
    mf, err = safe(lambda: mfi(df).iloc[-1]); mom["mfi"] = round(float(mf), 1) if mf is not None and not pd.isna(mf) else None
    out["momentum"] = mom

    # volatility
    vola = {}
    (bu, bm, bl, pb, bw), err = safe(lambda: bollinger(close))
    if bu is not None:
        vola["bollinger"] = {"upper": round(float(bu.iloc[-1]), 2), "mid": round(float(bm.iloc[-1]), 2),
                              "lower": round(float(bl.iloc[-1]), 2), "pct_b": round(float(pb.iloc[-1]), 3),
                              "bandwidth": round(float(bw.iloc[-1]), 3)}
    vola["atr_pct"] = round(float(atr(df).iloc[-1]) / last * 100, 2)
    (ku, km, kl), err = safe(lambda: keltner(df))
    if ku is not None:
        vola["keltner"] = {"upper": round(float(ku.iloc[-1]), 2), "mid": round(float(km.iloc[-1]), 2), "lower": round(float(kl.iloc[-1]), 2)}
    (dh, dl), err = safe(lambda: donchian(df))
    if dh is not None:
        vola["donchian"] = {"upper": round(float(dh.iloc[-1]), 2), "lower": round(float(dl.iloc[-1]), 2)}
    out["volatility"] = vola

    # volume
    volu = {}
    ov, err = safe(lambda: obv(close, vol))
    if ov is not None:
        volu["obv"] = int(ov.iloc[-1])
        volu["obv_slope_20"] = "rising" if ov.iloc[-1] > ov.iloc[-20] else "falling"
    ad, err = safe(lambda: ad_line(df))
    if ad is not None:
        volu["ad_line_slope_20"] = "rising" if ad.iloc[-1] > ad.iloc[-20] else "falling"
    vw52, err = safe(lambda: vwap_anchored(df, max(len(df) - 252, 0)))
    volu["vwap_anchored_52w"] = round(float(vw52), 2) if vw52 is not None else None
    cm, err = safe(lambda: cmf(df).iloc[-1]); volu["cmf"] = round(float(cm), 3) if cm is not None else None
    (kvo, ksig), err = safe(lambda: klinger(df))
    if kvo is not None:
        volu["klinger"] = round(float(kvo.iloc[-1]), 0)
        volu["klinger_signal"] = round(float(ksig.iloc[-1]), 0)
        volu["klinger_above_signal"] = bool(kvo.iloc[-1] > ksig.iloc[-1])
    volu["vol_last_vs_avg20"] = round(float(vol.iloc[-1] / vol.tail(20).mean()), 2)
    out["volume"] = volu

    # structure
    struct = {}
    piv, err = safe(lambda: pivots(float(df["High"].iloc[-2]), float(df["Low"].iloc[-2]), float(df["Close"].iloc[-2])))
    if piv is not None:
        struct["pivots_daily_basis"] = {k2: {k3: round(v3, 2) for k3, v3 in v2.items()} for k2, v2 in piv.items()}
    zz, err = safe(lambda: zigzag(close, 0.05))
    if zz is not None and len(zz) >= 2:
        last_two = zz[-2:]
        fib, err2 = safe(lambda: fib_retracement(max(p[1] for p in last_two), min(p[1] for p in last_two)))
        struct["fibonacci"] = fib
        struct["zigzag_last_pivots"] = [{"price": p[1], "type": p[2]} for p in zz[-5:]]
    out["structure"] = struct

    # patterns
    patt = {}
    cs, err = safe(lambda: candlestick_patterns(df))
    patt["candlestick_recent"] = cs if cs else []
    cp, err = safe(lambda: chart_pattern_heuristics(df))
    patt["chart_pattern_heuristic"] = cp if cp else ["no heuristic chart pattern flagged"]
    out["patterns"] = patt

    # stats
    stats = {}
    zs, err = safe(lambda: zscore(close).iloc[-1]); stats["zscore_vs_sma20"] = round(float(zs), 2) if zs is not None else None
    hu, err = safe(lambda: hurst_exponent(close))
    if hu is not None:
        stats["hurst_exponent"] = round(hu, 3)
        stats["hurst_read"] = "trending (>0.5)" if hu > 0.55 else "mean-reverting (<0.5)" if hu < 0.45 else "random-walk-like (~0.5)"
    kt, err = safe(lambda: kalman_trend(close))
    if kt is not None:
        stats["kalman_level"] = round(kt[0], 2)
        stats["kalman_slope_per_day"] = round(kt[1], 3)
    gv, err = safe(lambda: garch_vol_forecast(close))
    stats["garch"] = gv if gv is not None else {"error": err}
    af, err = safe(lambda: arima_forecast(close))
    stats["arima"] = af if af is not None else {"error": err}
    hmm, err = safe(lambda: regime_hmm(close))
    stats["regime"] = hmm if hmm is not None else {"error": err}
    wy, err = safe(lambda: wyckoff_heuristic(df))
    stats["wyckoff"] = wy if wy is not None else {"error": err}
    out["stats"] = stats

    return out


def main():
    ap = argparse.ArgumentParser(description="Extended Stage 2 technical panel")
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--json")
    ap.add_argument("--asof")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    if args.file:
        with open(args.file) as fh:
            tickers += [ln.strip().upper() for ln in fh if ln.strip()]
    if not tickers:
        sys.exit("No tickers given.")

    out = []
    for t in tickers:
        print(f"  computing extended panel for {t}...", file=sys.stderr)
        out.append(analyze(t, args.asof))

    print(json.dumps(out, indent=2, default=str))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nDetail written to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
