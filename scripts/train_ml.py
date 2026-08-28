#!/usr/bin/env python3
"""
train_ml.py — trains a lightweight gradient boosting classifier that scores
P(positive forward 10-trading-day return) from engineered technical features.

INFORMATIONAL ONLY. Nothing in the trade-action pipeline (the $100 buy cap,
the 5% top-up rule, the SELL triggers, the IBKR draft creation) is gated on
this score. It is reported alongside the Stage 2 outlook as one more input
for a human to weigh — the outlook vocabulary (Constructive/Neutral/
Deteriorating/Weak/Inflecting) remains the sole trigger for trade actions.

Trained on a fixed ~40-ticker cross-sector universe, NOT on the disclosure
watch list itself — training on 2-3 tickers' ~500 daily rows would be a
massive overfitting risk. Retrain weekly (or whenever ml_model_meta.json is
missing or >7 days old); daily runs just load the persisted model and score.

Writes ml_model.joblib and ml_model_meta.json to the PARENT of this script's
directory, or to CB_MODEL_DIR if set.

DEPENDENCIES
    pip install yfinance pandas numpy scikit-learn joblib
"""
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

from technicals_v2 import ema, rsi, adx_dmi, stochastic, williams_r, cci, roc, bollinger, atr, obv, cmf, zscore

UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "COST", "AVGO", "ADBE", "CSCO", "CRM", "MCD", "BAC", "TMO",
    "ACN", "LIN", "ABT", "DHR", "TXN", "NKE", "PM", "NEE", "UPS", "LOW",
]

FEATURE_COLS = [
    "ema12_dist", "ema26_dist", "ema50_dist", "ema200_dist", "rsi14", "macd_hist",
    "adx", "plus_di", "minus_di", "stoch_k", "stoch_d", "williams_r", "cci",
    "roc10", "roc20", "bb_pct_b", "bb_bandwidth", "atr_pct", "obv_slope20", "cmf",
    "zscore20", "vol_ratio20",
]


def features_for_df(df):
    close, vol = df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["ema12_dist"] = close / ema(close, 12) - 1
    f["ema26_dist"] = close / ema(close, 26) - 1
    f["ema50_dist"] = close / ema(close, 50) - 1
    f["ema200_dist"] = close / ema(close, 200) - 1
    f["rsi14"] = rsi(close)
    macd_line = ema(close, 12) - ema(close, 26)
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = macd_line - macd_sig
    pdi, mdi, adx_v = adx_dmi(df)
    f["adx"], f["plus_di"], f["minus_di"] = adx_v, pdi, mdi
    k, d = stochastic(df)
    f["stoch_k"], f["stoch_d"] = k, d
    f["williams_r"] = williams_r(df)
    f["cci"] = cci(df)
    f["roc10"], f["roc20"] = roc(close, 10), roc(close, 20)
    bu, bm, bl, pb, bw = bollinger(close)
    f["bb_pct_b"], f["bb_bandwidth"] = pb, bw
    f["atr_pct"] = atr(df) / close * 100
    ov = obv(close, vol)
    f["obv_slope20"] = ov.pct_change(20)
    f["cmf"] = cmf(df)
    f["zscore20"] = zscore(close)
    f["vol_ratio20"] = vol / vol.rolling(20).mean()
    return f


def build_dataset(horizon=10):
    print(f"Downloading {len(UNIVERSE)} tickers, 2y history...", file=sys.stderr)
    data = yf.download(UNIVERSE, period="2y", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    rows = []
    for tkr in UNIVERSE:
        try:
            df = data[tkr].dropna()
        except Exception:
            continue
        if len(df) < 260:
            continue
        f = features_for_df(df)
        fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1
        label = (fwd_ret > 0).astype(int)
        block = f.copy()
        block["label"] = label
        block["ticker"] = tkr
        rows.append(block)
    full = pd.concat(rows).dropna()
    return full


def main():
    full = build_dataset()
    X, y = full[FEATURE_COLS], full["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, proba > 0.5)

    out_dir = os.environ.get("CB_MODEL_DIR") or (os.path.dirname(os.path.abspath(__file__)) + "/..")
    joblib.dump(clf, os.path.join(out_dir, "ml_model.joblib"))
    meta = {
        "trained_date": datetime.now().date().isoformat(),
        "universe_n": len(UNIVERSE),
        "n_samples": int(len(full)),
        "holdout_auc": round(float(auc), 3),
        "holdout_accuracy": round(float(acc), 3),
        "horizon_days": 10,
        "feature_cols": FEATURE_COLS,
        "note": ("Informational only. Not used to trigger BUY/SELL trade actions — those stay "
                 "keyed to the five-term outlook vocabulary. Time-based holdout on a fixed "
                 "40-ticker universe (not the disclosure watch list); read as a sanity check, "
                 "not a live track record. Retrain weekly."),
    }
    with open(os.path.join(out_dir, "ml_model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
