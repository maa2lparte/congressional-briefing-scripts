#!/usr/bin/env python3
"""
capture_ratio.py — Section 6 of the biweekly review: for each held position, compare
the achievable return (from the insider's own transaction date to today) against the
captured return (from our entry-fill date to today).

BACKGROUND (2026-08-31 biweekly review): this section was reported "blocked" for two
consecutive reviews before it turned out the REST insider endpoint just needed to be
tried instead of assuming the QuiverQuant MCP tool was the only source. This script
formalizes the methodology used the first time it actually ran, so the next biweekly
review runs it instead of re-deriving the same logic inline (the exact token-wastage
complaint that prompted moving these scripts to GitHub in the first place).

METHOD
    For each held ticker, find the most recent insider Form-4 open-market purchase
    (TransactionCode "P") on or before our own entry-fill date — the signal that
    plausibly drove the trade. Compare:
        achievable % = (current_price / insider_signal_price - 1) * 100
        captured   % = (current_price / our_entry_price       - 1) * 100
        pp_lost      = achievable - captured
    Report mean/median pp_lost and mean/median lag (days between the insider's
    transaction and our fill) across all tickers with an identifiable insider signal.
    Tickers with no insider P row on/before entry are excluded (not insider-sourced —
    congressional or another source drove those trades), not forced into the calc.

INPUT
    --insiders-json   raw beta/live/insiders?... response, or a saved copy (list of
                       dicts with Ticker/Date/TransactionCode/PricePerShare)
    --positions-json  a JSON list of {"ticker","entry_date","entry_price","current_price"}
                       — build this from IBKR get_account_positions (current_price,
                       average_price) + get_account_trades (first BUY fill date per
                       ticker = entry_date). Not fetched by this script directly since
                       IBKR access is via MCP tools, not a plain REST call this script
                       can make — the caller (the review run) assembles this file first.

USAGE
    python3 capture_ratio.py --insiders-json raw_insiders.json --positions-json positions.json

DEPENDENCIES: standard library only.
"""
import argparse
import json
import statistics as st
from datetime import date


def most_recent_signal(ticker, entry_date, insider_rows):
    candidates = [
        r for r in insider_rows
        if r.get("Ticker") == ticker
        and r.get("TransactionCode") == "P"
        and r.get("Date", "")[:10] <= entry_date
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["Date"])
    best = candidates[-1]
    return {"signal_date": best["Date"][:10], "signal_price": best["PricePerShare"]}


def main():
    ap = argparse.ArgumentParser(description="Compute insider-signal-to-fill capture ratio for held positions.")
    ap.add_argument("--insiders-json", required=True)
    ap.add_argument("--positions-json", required=True)
    args = ap.parse_args()

    insider_rows = json.load(open(args.insiders_json))
    positions = json.load(open(args.positions_json))

    rows = []
    excluded = []
    for p in positions:
        sig = most_recent_signal(p["ticker"], p["entry_date"], insider_rows)
        if sig is None:
            excluded.append(p["ticker"])
            continue
        achievable = (p["current_price"] / sig["signal_price"] - 1) * 100
        captured = (p["current_price"] / p["entry_price"] - 1) * 100
        lag_days = (date.fromisoformat(p["entry_date"]) - date.fromisoformat(sig["signal_date"])).days
        rows.append({
            "ticker": p["ticker"],
            "signal_date": sig["signal_date"],
            "signal_price": sig["signal_price"],
            "entry_date": p["entry_date"],
            "lag_days": lag_days,
            "achievable_pct": round(achievable, 2),
            "captured_pct": round(captured, 2),
            "pp_lost": round(achievable - captured, 2),
        })

    pp_lost = [r["pp_lost"] for r in rows]
    lags = [r["lag_days"] for r in rows]

    out = {
        "n": len(rows),
        "excluded_not_insider_sourced": excluded,
        "rows": sorted(rows, key=lambda r: -r["pp_lost"]),
        "summary": {
            "mean_pp_lost": round(st.mean(pp_lost), 2) if pp_lost else None,
            "median_pp_lost": round(st.median(pp_lost), 2) if pp_lost else None,
            "mean_lag_days": round(st.mean(lags), 1) if lags else None,
            "median_lag_days": st.median(lags) if lags else None,
        },
        "note": (
            "n below ~10 should be reported as such, not ranked or trusted as a "
            "pipeline-wide baseline — see the small-n discipline rule in policy."
        ),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
