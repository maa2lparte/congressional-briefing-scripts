#!/usr/bin/env python3
"""Supplements technicals_v2.py with the base SMA trend read, relative
performance vs SPY + sector ETF, swing levels, company name and next earnings
date — the fields the addendum's block format needs that v2 doesn't emit."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

WORKDIR = __import__("os").environ.get("CB_WORKDIR", "/tmp/cb")
SECTOR_ETFS = {
    "technology": "XLK", "energy": "XLE", "financial services": "XLF",
    "healthcare": "XLV", "industrials": "XLI", "utilities": "XLU",
    "consumer defensive": "XLP", "consumer cyclical": "XLY",
    "basic materials": "XLB", "real estate": "XLRE",
    "communication services": "XLC",
}


def pct(s, n):
    if len(s) <= n:
        return None
    return round((float(s.iloc[-1]) / float(s.iloc[-1 - n]) - 1) * 100, 2)


def main():
    tickers = [t.upper() for t in sys.argv[1:]]
    bench = {}
    for b in ["SPY"] + sorted(set(SECTOR_ETFS.values())):
        bench[b] = yf.Ticker(b).history(period="2y")["Close"]

    out = {}
    for t in tickers:
        tk = yf.Ticker(t)
        h = tk.history(period="2y")
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
            rel[label] = {
                "ticker_pct": r,
                "spy_pct": pct(bench["SPY"], n),
                "excess_vs_spy": (round(r - pct(bench["SPY"], n), 2) if r is not None else None),
                "sector_etf": etf,
                "excess_vs_sector": (round(r - pct(bench[etf], n), 2)
                                     if (r is not None and etf) else None),
            }

        # swing highs/lows on a 5-bar fractal, last 6 months
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

        out[t] = {
            "name": info.get("longName") or info.get("shortName") or t,
            "sector": info.get("sector"),
            "sector_etf": etf,
            "close": round(px, 2),
            "sma20": round(float(c.rolling(20).mean().iloc[-1]), 2),
            "sma50": round(float(c.rolling(50).mean().iloc[-1]), 2),
            "sma200": round(float(c.rolling(200).mean().iloc[-1]), 2),
            "sma50_slope_20d": round(float(c.rolling(50).mean().iloc[-1]
                                           - c.rolling(50).mean().iloc[-21]), 2),
            "sma200_slope_20d": round(float(c.rolling(200).mean().iloc[-1]
                                            - c.rolling(200).mean().iloc[-21]), 2),
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
        print(f"  {t} done", file=sys.stderr)

    json.dump(out, open(f"{WORKDIR}/supplement_today.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
