#!/usr/bin/env python3
"""supplement_batch.py -- drop-in replacement for the one-ticker-per-invocation
supplement.py loop.

WHY THIS EXISTS (2026-09-26):
  supplement.py fetches SPY + 11 sector ETFs (12 yfinance history calls) at the
  top of EVERY invocation, then 3 calls per ticker. Because it aborted the whole
  batch on the first bad ticker, the pipeline ran it once per ticker -- so a
  60-ticker run made 60 x (12 + 3) = ~900 yfinance calls, ~720 of them redundant
  benchmark refetches. That is the real cause of the ~170-call rate-limit trap.

  This version:
    * fetches the 12 benchmarks ONCE,
    * wraps each ticker in try/except so one short-history symbol cannot abort
      the batch (failures are recorded, never silently dropped),
    * fetches tickers concurrently with a small, capped thread pool (I/O-bound),
    * writes FLAT output {TICKER: {...}} -- the shape short_screen.py's
      `gate --supp` and every other reader expects. (The old per-ticker loop +
      merge produced {TICKER: {TICKER: {...}}}, which silently nulled the
      relative-strength and SMA-slope terms in the short-side outlook.)

  Field-for-field the per-ticker record is identical to supplement.py's.

Usage:
  YF_DISABLE_CURL_CFFI=1 python3 supplement_batch.py --out supp.json [--workers 4] T1 T2 ...
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

SECTOR_ETFS = {
    "technology": "XLK", "energy": "XLE", "financial services": "XLF",
    "healthcare": "XLV", "industrials": "XLI", "utilities": "XLU",
    "consumer defensive": "XLP", "consumer cyclical": "XLY",
    "basic materials": "XLB", "real estate": "XLRE",
    "communication services": "XLC",
}


def pct(s, n):
    if s is None or len(s) <= n:
        return None
    return round((float(s.iloc[-1]) / float(s.iloc[-1 - n]) - 1) * 100, 2)


def history_with_retry(sym, period="2y", tries=3):
    last = None
    for i in range(tries):
        try:
            h = yf.Ticker(sym).history(period=period)
            if h is not None:
                # drop NaN-close bars (Yahoo partial/late bar) -- see technicals_v2 note
                h = h.dropna(subset=["Close"])
            if h is not None and len(h):
                return h
        except Exception as e:  # rate limit / transient network
            last = e
        time.sleep(1.5 * (i + 1))
    if last:
        raise last
    return None


def one(t, bench):
    tk = yf.Ticker(t)
    h = history_with_retry(t)
    # Same tolerance as supplement.py: short histories still produce a record
    # (SMA200 comes out NaN); only fail when the 20-day slope can't be computed.
    if h is None or len(h) < 22:
        raise ValueError(f"insufficient history ({0 if h is None else len(h)} rows)")
    c = h["Close"]
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        pass
    sec = (info.get("sector") or "").lower()
    etf = SECTOR_ETFS.get(sec)

    rel = {}
    for label, n in [("1M", 21), ("3M", 63), ("6M", 126)]:
        r = pct(c, n)
        spy = pct(bench.get("SPY"), n)
        sect = pct(bench.get(etf), n) if etf else None
        rel[label] = {
            "ticker_pct": r,
            "spy_pct": spy,
            "excess_vs_spy": (round(r - spy, 2) if (r is not None and spy is not None) else None),
            "sector_etf": etf,
            "excess_vs_sector": (round(r - sect, 2) if (r is not None and sect is not None) else None),
        }

    w = h.tail(126)
    hi, lo = [], []
    for i in range(2, len(w) - 2):
        hh = w["High"].iloc[i]
        ll = w["Low"].iloc[i]
        if hh == w["High"].iloc[i - 2:i + 3].max():
            hi.append(round(float(hh), 2))
        if ll == w["Low"].iloc[i - 2:i + 3].min():
            lo.append(round(float(ll), 2))
    px = float(c.iloc[-1])

    earn = None
    try:
        cal = tk.calendar or {}
        e = cal.get("Earnings Date")
        if isinstance(e, list) and e:
            earn = str(e[0])
        elif e:
            earn = str(e)
    except Exception:
        pass

    return {
        "name": info.get("longName") or info.get("shortName") or t,
        "sector": info.get("sector"),
        "sector_etf": etf,
        "close": round(px, 2),
        "last_bar_date": str(h.index[-1].date()),
        "sma20": round(float(c.rolling(20).mean().iloc[-1]), 2),
        "sma50": round(float(c.rolling(50).mean().iloc[-1]), 2),
        "sma200": round(float(c.rolling(200).mean().iloc[-1]), 2),
        "sma50_slope_20d": round(float(c.rolling(50).mean().iloc[-1] - c.rolling(50).mean().iloc[-21]), 2),
        "sma200_slope_20d": round(float(c.rolling(200).mean().iloc[-1] - c.rolling(200).mean().iloc[-21]), 2),
        "relative": rel,
        "swing_resistance": sorted({x for x in hi if x > px})[:4],
        "swing_support": sorted({x for x in lo if x < px}, reverse=True)[:4],
        "high_52w": round(float(h["High"].tail(252).max()), 2),
        "low_52w": round(float(h["Low"].tail(252).min()), 2),
        "next_earnings": earn,
        "analyst_target_median": info.get("targetMedianPrice"),
        "analyst_target_low": info.get("targetLowPrice"),
        "analyst_target_high": info.get("targetHighPrice"),
        "analyst_n": info.get("numberOfAnalystOpinions"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("tickers", nargs="+")
    a = ap.parse_args()
    tickers = sorted({t.upper() for t in a.tickers})

    t0 = time.time()
    bench = {}
    for b in ["SPY"] + sorted(set(SECTOR_ETFS.values())):
        try:
            bench[b] = history_with_retry(b)["Close"]
        except Exception as e:
            print(f"  benchmark {b} FAILED: {e}", file=sys.stderr)

    out, failed = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(one, t, bench): t for t in tickers}
        for f in as_completed(futs):
            t = futs[f]
            try:
                out[t] = f.result()
            except Exception as e:
                failed[t] = str(e)[:200]

    json.dump(out, open(a.out, "w"), indent=1)
    meta = {"n_requested": len(tickers), "n_ok": len(out), "failed": failed,
            "benchmarks_ok": sorted(bench), "workers": a.workers,
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(a.out.replace(".json", "_meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
