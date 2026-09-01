#!/usr/bin/env python3
"""
altsignal_daily_refresh.py -- rebuild $CB_WORKDIR/altsignal.json for the DAILY pipeline,
in the EXACT flat schema stage1a_v2.py's load_altsignal() reads:
    {"TICKER": {"short_squeeze": 0-100, "gov_contract_momentum": 0-100, "news_momentum": 0-100}, ...}
(top-level keys ARE tickers -- no "_meta"/"components" wrapper, no positional arrays.)

BACKGROUND (2026-09-01): the 2026-08-31 population of altsignal.json used a DIFFERENT,
incompatible shape ({"_meta":..., "components": {TICKER: [a,b,c]}}), which stage1a_v2.py's
load_altsignal() silently failed to read -- every altdata.get(tkr, {}) lookup missed, so
every AltSignal component scored 0 for every ticker, every run, even though the file
"existed" (when it was even copied into WORKDIR at all -- see the second bug below).
That earlier file has been reshaped in Drive; this script is the go-forward daily
refresh mechanism, kept separate from biweekly/altsignal_refresh.py (which intentionally
uses a different, human-auditable shape for the biweekly review's own reporting, not for
feeding stage1a_v2.py directly).

SECOND BUG FIXED BY THIS SCRIPT'S EXISTENCE: even a correctly-shaped altsignal.json sitting
in Drive does nothing for stage1a_v2.py unless something actually copies it to
$CB_WORKDIR/altsignal.json before Stage 1A runs. No such step existed anywhere in the
daily pipeline (policy prose said "refresh every run" without saying HOW). Running this
script as an explicit Step 0-adjacent step writes directly to CB_WORKDIR, closing that gap.

VERIFIED WORKING ENDPOINTS (2026-08-31), curl + Bearer quiver_api_key:
  short_squeeze         beta/live/wallstreetbets            (no ticker param -- bulk, n~300)
  gov_contract_momentum beta/live/govcontractsall            (no ticker param -- bulk, n~20000 rows)
  news_momentum         beta/live/quivernews?ticker=TICKER   (PER-TICKER ONLY)

USAGE
    python3 altsignal_daily_refresh.py --api-key-file config.json \
        --out "$CB_WORKDIR/altsignal.json" --gov-window-days 90 \
        --candidates TICKER1 TICKER2 ...
    (--candidates should be that day's actual watch list: held positions + congressional
    survivors + insider survivors -- not an attempt at the full ~565-name universe, which
    would need that many sequential news_momentum calls.)

    If run with no --candidates (e.g. a quick refresh of only short_squeeze/gov_contract
    coverage), news_momentum is left at 0 for every ticker (missing_data_convention: 0 is
    correct for "not computed", not "measured zero").

DEPENDENCIES: standard library only (urllib).
"""
import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

WSB_URL = "https://api.quiverquant.com/beta/live/wallstreetbets"
GOVCONTRACTS_URL = "https://api.quiverquant.com/beta/live/govcontractsall"
NEWS_URL_TMPL = "https://api.quiverquant.com/beta/live/quivernews?ticker={ticker}"


def _get(url, api_key, timeout=60):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_short_squeeze(api_key):
    rows = _get(WSB_URL, api_key)
    rows_sorted = sorted(rows, key=lambda r: r["Count"])
    n = len(rows_sorted)
    return {r["Ticker"]: round((i + 1) / n * 100, 1) for i, r in enumerate(rows_sorted)}, n


def fetch_gov_contract_momentum(api_key, window_days=90):
    rows = _get(GOVCONTRACTS_URL, api_key)
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    totals = defaultdict(float)
    for r in rows:
        if r.get("Date", "") >= cutoff:
            totals[r["Ticker"]] += r.get("Amount") or 0.0
    ordered = sorted(totals.items(), key=lambda kv: kv[1])
    n = len(ordered)
    return {t: round((i + 1) / n * 100, 1) for i, (t, _) in enumerate(ordered)}, n


def fetch_news_momentum_one(ticker, api_key, lookback_days=14):
    data = _get(NEWS_URL_TMPL.format(ticker=ticker), api_key)
    rows = data.get("data", [])
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    n_recent = sum(1 for r in rows if r.get("time", "")[:10] >= cutoff)
    return n_recent


def percentile_rank_within(values_by_ticker):
    """values_by_ticker: {ticker: raw_count}. Returns {ticker: percentile 0-100, 1dp}
    ranked within just this sample (small-n; only meaningful relative to itself)."""
    ordered = sorted(values_by_ticker.items(), key=lambda kv: kv[1])
    n = len(ordered)
    if n == 0:
        return {}
    return {t: round((i + 1) / n * 100, 1) for i, (t, _) in enumerate(ordered)}


def main():
    ap = argparse.ArgumentParser(description="Refresh $CB_WORKDIR/altsignal.json for the daily pipeline, in stage1a_v2.py's expected flat schema.")
    ap.add_argument("--api-key-file", required=True, help="Path to config.json containing quiver_api_key")
    ap.add_argument("--out", required=True, help="Should be $CB_WORKDIR/altsignal.json")
    ap.add_argument("--gov-window-days", type=int, default=90)
    ap.add_argument("--candidates", nargs="*", default=[], help="That day's actual watch list for news_momentum (held + congressional + insider survivors). Keep this short -- one HTTP call per ticker.")
    ap.add_argument("--news-lookback-days", type=int, default=14)
    args = ap.parse_args()

    cfg = json.load(open(args.api_key_file))
    api_key = cfg["quiver_api_key"]

    print("Fetching short_squeeze (beta/live/wallstreetbets)...", file=sys.stderr)
    squeeze, n_squeeze = fetch_short_squeeze(api_key)
    print(f"  {n_squeeze} tickers scored", file=sys.stderr)

    print(f"Fetching gov_contract_momentum (beta/live/govcontractsall, {args.gov_window_days}d window)...", file=sys.stderr)
    govc, n_govc = fetch_gov_contract_momentum(api_key, args.gov_window_days)
    print(f"  {n_govc} tickers scored", file=sys.stderr)

    news_raw = {}
    for t in args.candidates:
        print(f"Fetching news_momentum for {t}...", file=sys.stderr)
        news_raw[t] = fetch_news_momentum_one(t, api_key, args.news_lookback_days)
    news_pct = percentile_rank_within(news_raw) if news_raw else {}

    all_tickers = sorted(set(squeeze) | set(govc) | set(news_pct))
    out = {
        "_meta": {
            "_schema": "altsignal-components-flat-v2 (matches stage1a_v2.py's load_altsignal() exactly -- top-level keys ARE tickers)",
            "_generated_by": "scripts/altsignal_daily_refresh.py",
            "_generated": date.today().isoformat(),
            "_status": "COLLECTING -- see policy conviction.altsignal.altsignal_weighting_status. Effective conviction weight stays 0 until that status is explicitly flipped to LIVE.",
            "_gov_window_days": args.gov_window_days,
            "_news_lookback_days": args.news_lookback_days,
            "_news_momentum_scope": f"news_momentum computed only for --candidates passed this run ({len(args.candidates)} of them), percentile-ranked WITHIN that sample only (small-n, not comparable across runs) -- NOT a universe-wide population.",
        },
    }
    for t in all_tickers:
        out[t] = {
            "short_squeeze": squeeze.get(t, 0),
            "gov_contract_momentum": govc.get(t, 0),
            "news_momentum": news_pct.get(t, 0),
        }

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"Wrote {args.out}: {len(all_tickers)} tickers (flat schema, ready for stage1a_v2.py)", file=sys.stderr)


if __name__ == "__main__":
    main()
