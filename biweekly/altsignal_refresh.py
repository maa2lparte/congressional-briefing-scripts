#!/usr/bin/env python3
"""
altsignal_refresh.py — rebuild altsignal.json's short_squeeze and gov_contract_momentum
components from live QuiverQuant REST endpoints. news_momentum is NOT bulk-populated here
(see fetch_news_momentum below and the docstring note) — call that function per-ticker,
for the actual candidate list a run has that day, not for a static universe.

BACKGROUND (2026-08-31 biweekly review): altsignal.json had never been populated since
AltSignal went live 2026-08-17. The first pass wrongly concluded no REST source existed
for any of the three components by testing endpoints that turned out to need query
parameters not tried initially. This script encodes what was actually verified working,
so it doesn't need re-deriving by an LLM in-context every run (see the biweekly review's
own complaint about token cost this script exists to fix).

VERIFIED WORKING ENDPOINTS (2026-08-31), curl + Bearer quiver_api_key:
  short_squeeze         beta/live/wallstreetbets            (no ticker param — bulk, n~300)
  gov_contract_momentum beta/live/govcontractsall            (no ticker param — bulk, n~20000 rows)
  news_momentum         beta/live/quivernews?ticker=TICKER   (PER-TICKER ONLY — the bare
                         endpoint without ?ticker= returns an undifferentiated 30-item
                         digest with no Ticker field and is NOT usable for scoring)
NOT usable / not confirmed:
  beta/live/dailydarkpool and several guessed news-endpoint names all returned 404.
  Do not assume an endpoint lacks a per-ticker mode without trying ?ticker=SYMBOL first —
  that is exactly the mistake that caused the first pass to under-deliver on news_momentum.

USAGE
    python3 altsignal_refresh.py --api-key-file config.json --out altsignal.json \
        --gov-window-days 90 --held HELD1 HELD2 ...

    (config.json is the same file the daily pipeline already reads; expects a
    "quiver_api_key" field.)

The output schema matches what stage1a_v2.py expects: components[TICKER] = a dict with
short_squeeze / gov_contract_momentum / news_momentum keys, each 0-100. This differs
slightly from the compact [a,b,c] array format used in the 2026-08-31 one-off population —
prefer this dict form going forward since stage1a_v2.py's altsignal_score() reads by key,
not by position, so either works, but the dict form is more robust to future component
additions.

DEPENDENCIES: standard library only (urllib), so it runs anywhere the daily/biweekly
pipeline already runs without an extra pip install.
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
    """Percentile rank (0-100, one decimal) of WSB mention Count across the whole
    beta/live/wallstreetbets universe returned that day. A retail-attention proxy,
    NOT short-interest-based — no float-short/days-to-cover source has been found."""
    rows = _get(WSB_URL, api_key)
    rows_sorted = sorted(rows, key=lambda r: r["Count"])
    n = len(rows_sorted)
    return {r["Ticker"]: round((i + 1) / n * 100, 1) for i, r in enumerate(rows_sorted)}, n


def fetch_gov_contract_momentum(api_key, window_days=90):
    """Percentile rank (0-100, one decimal) of trailing-window government contract
    award dollar total, from beta/live/govcontractsall. Ticker absent from the feed
    in the window = no score (caller should apply the missing_data_convention: 0,
    not neutral 50)."""
    rows = _get(GOVCONTRACTS_URL, api_key)
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    totals = defaultdict(float)
    for r in rows:
        if r.get("Date", "") >= cutoff:
            totals[r["Ticker"]] += r.get("Amount") or 0.0
    ordered = sorted(totals.items(), key=lambda kv: kv[1])
    n = len(ordered)
    return {t: round((i + 1) / n * 100, 1) for i, (t, _) in enumerate(ordered)}, n, dict(totals)


def fetch_news_momentum_one(ticker, api_key, lookback_days=14):
    """Call PER TICKER for that day's actual candidate/survivor list — never in a loop
    over hundreds of tickers; that is hundreds of sequential HTTP calls for no benefit,
    since the daily/biweekly pipeline only ever needs this for its own short list.
    Returns (item_count_in_lookback_window, newest_item_date_or_None, category_counts)."""
    data = _get(NEWS_URL_TMPL.format(ticker=ticker), api_key)
    rows = data.get("data", [])
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    n_recent = sum(1 for r in rows if r.get("time", "")[:10] >= cutoff)
    newest = max((r["time"][:10] for r in rows), default=None)
    cats = defaultdict(int)
    for r in rows:
        cats[r.get("category", "unknown")] += 1
    return n_recent, newest, dict(cats)


def main():
    ap = argparse.ArgumentParser(description="Refresh altsignal.json's short_squeeze and gov_contract_momentum from live REST endpoints.")
    ap.add_argument("--api-key-file", required=True, help="Path to config.json containing quiver_api_key")
    ap.add_argument("--out", default="altsignal.json")
    ap.add_argument("--gov-window-days", type=int, default=90)
    ap.add_argument("--held", nargs="*", default=[], help="Tickers to also fetch news_momentum for (per-ticker calls, keep this list short)")
    ap.add_argument("--news-lookback-days", type=int, default=14)
    args = ap.parse_args()

    cfg = json.load(open(args.api_key_file))
    api_key = cfg["quiver_api_key"]

    print("Fetching short_squeeze (beta/live/wallstreetbets)...", file=sys.stderr)
    squeeze, n_squeeze = fetch_short_squeeze(api_key)
    print(f"  {n_squeeze} tickers scored", file=sys.stderr)

    print(f"Fetching gov_contract_momentum (beta/live/govcontractsall, {args.gov_window_days}d window)...", file=sys.stderr)
    govc, n_govc, govc_raw = fetch_gov_contract_momentum(api_key, args.gov_window_days)
    print(f"  {n_govc} tickers scored", file=sys.stderr)

    news = {}
    for t in args.held:
        print(f"Fetching news_momentum for {t}...", file=sys.stderr)
        n_recent, newest, cats = fetch_news_momentum_one(t, api_key, args.news_lookback_days)
        news[t] = {"n_recent": n_recent, "newest_item": newest, "categories": cats}

    all_tickers = sorted(set(squeeze) | set(govc) | set(news))
    components = {}
    for t in all_tickers:
        components[t] = {
            "short_squeeze": squeeze.get(t, 0),
            "gov_contract_momentum": govc.get(t, 0),
            "news_momentum_recent_count": news.get(t, {}).get("n_recent", 0),
        }

    out = {
        "_meta": {
            "_schema": "altsignal-components",
            "_generated_by": "biweekly/altsignal_refresh.py",
            "_status": "COLLECTING — see policy conviction.altsignal.altsignal_weighting_status. Effective conviction weight stays 0 until that status is explicitly flipped to LIVE.",
            "_gov_window_days": args.gov_window_days,
            "_news_lookback_days": args.news_lookback_days,
            "_news_momentum_scope": f"Only computed for --held tickers passed this run ({len(args.held)} of them) — NOT a universe-wide population. Extend by passing more tickers, not by looping this script over hundreds of names.",
        },
        "components": components,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"Wrote {args.out}: {len(components)} tickers", file=sys.stderr)


if __name__ == "__main__":
    main()
