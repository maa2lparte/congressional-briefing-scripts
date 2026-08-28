#!/usr/bin/env python3
"""
score_ml.py — scores tickers against the persisted ML model.

INFORMATIONAL ONLY — see train_ml.py docstring. This never gates a trade
action; it's reported context alongside the outlook.

Usage: python score_ml.py TICKER [TICKER ...]
If ml_model.joblib / ml_model_meta.json don't exist yet or are >7 days old,
prints a note instead of a score — the daily task should call train_ml.py
first in that case (weekly cadence, not every run).

NOTE ON PATHS: MODEL_PATH/META_PATH resolve to the PARENT of this script's
directory. Keep the layout <workdir>/scripts/score_ml.py with the model at
<workdir>/ml_model.joblib, or set CB_MODEL_DIR to override.
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import yfinance as yf

from train_ml import FEATURE_COLS, features_for_df

BASE = os.environ.get("CB_MODEL_DIR") or (os.path.dirname(os.path.abspath(__file__)) + "/..")
MODEL_PATH = os.path.join(BASE, "ml_model.joblib")
META_PATH = os.path.join(BASE, "ml_model_meta.json")


def score(ticker, clf):
    df = yf.Ticker(ticker).history(period="420d", auto_adjust=True)
    if df is None or len(df) < 210:
        return {"ticker": ticker, "error": "insufficient history"}
    f = features_for_df(df)
    row = f[FEATURE_COLS].iloc[[-1]]
    if row.isna().any(axis=1).iloc[0]:
        return {"ticker": ticker, "error": "missing feature values (insufficient warm-up history)"}
    proba_up = float(clf.predict_proba(row)[:, 1][0])
    return {"ticker": ticker, "p_positive_10d_forward_return": round(proba_up, 3)}


def main():
    tickers = sys.argv[1:]
    if not tickers:
        sys.exit("No tickers given.")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        print(json.dumps({"status": "no_model", "note": "Run train_ml.py first."}))
        return

    meta = json.load(open(META_PATH))
    trained = datetime.fromisoformat(meta["trained_date"]).date()
    stale = (datetime.now().date() - trained) > timedelta(days=7)

    clf = joblib.load(MODEL_PATH)
    out = {
        "model_meta": meta,
        "stale": stale,
        "stale_note": "Model is >7 days old — retrain due (train_ml.py), scores below still usable but aging." if stale else None,
        "scores": [score(t, clf) for t in tickers],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
