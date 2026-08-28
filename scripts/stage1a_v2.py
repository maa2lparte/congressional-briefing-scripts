#!/usr/bin/env python3
"""Stage 1A v2: filter congressional disclosures, score conviction, emit survivor CSV.

CHANGES FROM v1 (2026-08-23):
  1. AltSignal (the 0.20 slot) is now actually computed, from an OPTIONAL
     altsignal.json in WORKDIR: {"TICKER": {"short_squeeze": 0-100,
     "gov_contract_momentum": 0-100, "news_momentum": 0-100}}.
     AltSignal = short_squeeze*0.40 + gov_contract_momentum*0.35 + news_momentum*0.25.
     Missing-data convention: an absent component, or an absent ticker, or an
     absent file entirely, scores 0 -- NOT a neutral 50. If altsignal.json is
     absent, altsignal_available=False is set at the top level and a loud
     warning is printed; every ticker's altsignal is 0 in that case (same
     numeric effect as v1's committee=0, but now explicit and auditable
     rather than silently hard-coded).
  2. Filer weighting replaces binary bulk-filer exclusion. For each filer f,
     max_tickers_on_any_report_date(f) = the largest number of distinct
     tickers f disclosed on any single ReportDate anywhere in the feed.
     filer_weight(f) = min(1.0, K / max_tickers_on_any_report_date(f)), K=8
     by default (the 75th percentile of the observed 3-345 tickers/filer/date
     distribution as of 2026-08-22). A filer who never appears bulk-filing
     gets weight 1.0; a filer who dumps hundreds of tickers on one date is
     discounted continuously instead of being all-or-nothing excluded. This
     is applied to band (weighted average, not max), cluster breadth
     (weighted filer count) and bipartisan (weighted party mix) -- so no
     single filer's inclusion/exclusion can single-handedly flip whether a
     ticker survives.
  3. conviction_unweighted (filer_weight forced to 1.0 for everyone) is
     reported alongside conviction (real weights applied) for every
     survivor, plus weighting_delta = conviction - conviction_unweighted,
     so the effect of weighting is auditable every run.

Conviction v2 = Band*0.35 + Cluster*0.30 + AltSignal*0.20 + Bipartisan*0.15
Solo filings: bipartisan is n/a, its 0.15 redistributed proportionally
              across the other three (divide by 0.85).
"""
import csv, json, sys, os
from collections import defaultdict
from datetime import date, timedelta

WORKDIR = os.environ.get("CB_WORKDIR", "/tmp/cb")
TODAY = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
WINDOW_START = TODAY - timedelta(days=45)
K = float(os.environ.get("CB_FILER_K", "8"))

BAND_LADDER = [
    ("$1,001 - $15,000", 0),
    ("$15,001 - $50,000", 15),
    ("$50,001 - $100,000", 30),
    ("$100,001 - $250,000", 45),
    ("$250,001 - $500,000", 60),
    ("$500,001 - $1,000,000", 75),
    ("$1,000,001 - $5,000,000", 90),
    ("$5,000,001 - $25,000,000", 100),
    ("$25,000,001 - $50,000,000", 100),
    ("Over $50,000,000", 100),
]
BAND_W = dict(BAND_LADDER)
BAND_RANK = {b: i for i, (b, _) in enumerate(BAND_LADDER)}
BIG_BAND_MIN_RANK = BAND_RANK["$50,001 - $100,000"]

ALT_WEIGHTS = {"short_squeeze": 0.40, "gov_contract_momentum": 0.35, "news_momentum": 0.25}


def band_weight(b):
    return BAND_W.get(b, 0)


def band_rank(b):
    return BAND_RANK.get(b, -1)


def cluster_breadth(n):
    """0 for solo/pair, 40 at three (weighted) members, 100 at eight or more."""
    if n < 3:
        return 0.0
    if n >= 8:
        return 100.0
    return 40.0 + (n - 3) * 12.0


def load_altsignal():
    path = f"{WORKDIR}/altsignal.json"
    if not os.path.exists(path):
        print("WARNING: altsignal.json not found -- AltSignal is 0 for every ticker this run "
              "(missing-data convention, not a neutral 50). altsignal_available=False.",
              file=sys.stderr)
        return {}, False
    try:
        data = json.load(open(path))
        return data, True
    except Exception as e:
        print(f"WARNING: altsignal.json failed to parse ({e}) -- treating as absent.", file=sys.stderr)
        return {}, False


def altsignal_score(tkr, altdata):
    comp = altdata.get(tkr, {})
    total = 0.0
    for k, w in ALT_WEIGHTS.items():
        total += comp.get(k, 0) * w
    return round(total, 1)


def compute_filer_weights(rows, k):
    """max_tickers_on_any_report_date(filer) -> filer_weight, over the WHOLE feed."""
    by_filer_date = defaultdict(set)
    for r in rows:
        f = r.get("Representative")
        rd = r.get("ReportDate")
        tkr = r.get("Ticker")
        if f and rd and tkr:
            by_filer_date[(f, rd)].add(tkr)
    max_by_filer = defaultdict(int)
    for (f, rd), tkrs in by_filer_date.items():
        max_by_filer[f] = max(max_by_filer[f], len(tkrs))
    weights = {}
    for f, mx in max_by_filer.items():
        weights[f] = round(min(1.0, k / mx), 4) if mx > 0 else 1.0
    return weights, dict(max_by_filer)


def score_ticker(tkr, rs, altdata, filer_weights, force_unweighted=False):
    filers = sorted({r["Representative"] for r in rs})
    parties_by_filer = {}
    for r in rs:
        if r.get("Party"):
            parties_by_filer[r["Representative"]] = r["Party"]

    def w(f):
        return 1.0 if force_unweighted else filer_weights.get(f, 1.0)

    weighted_n_filers = sum(w(f) for f in filers)

    # band: weighted average across events (weighted by the filer's weight)
    wsum = sum(w(r["Representative"]) for r in rs)
    if wsum > 0:
        bw = sum(w(r["Representative"]) * band_weight(r["Range"]) for r in rs) / wsum
    else:
        bw = 0.0
    top_band = max((r["Range"] for r in rs), key=band_rank)

    cb = cluster_breadth(weighted_n_filers)

    alt = altsignal_score(tkr, altdata)

    # bipartisan: weighted party mix. bp is None (solo) only if exactly one filer total.
    weighted_by_party = defaultdict(float)
    for f in filers:
        p = parties_by_filer.get(f)
        if p:
            weighted_by_party[p] += w(f)
    if len(filers) == 1:
        bp = None
    elif len(weighted_by_party) < 2:
        bp = 0.0
    else:
        total_w = sum(weighted_by_party.values())
        minority_w = sorted(weighted_by_party.values())[:-1]
        bp = round(100.0 * (sum(minority_w) / total_w), 1) if total_w > 0 else 0.0

    if bp is None:
        conv = (bw * 0.35 + cb * 0.30 + alt * 0.20) / 0.85
    else:
        conv = bw * 0.35 + cb * 0.30 + alt * 0.20 + bp * 0.15

    buys = [r for r in rs if r["Transaction"] == "Purchase"]
    sells = [r for r in rs if r["Transaction"].startswith("Sale")]

    return {
        "conviction": round(conv, 1),
        "band_weight": round(bw, 1),
        "cluster_breadth": round(cb, 1),
        "altsignal": alt,
        "bipartisan": bp,
        "weighted_n_filers": round(weighted_n_filers, 3),
        "n_filers": len(filers),
        "filers": filers,
        "filer_weights_applied": {f: w(f) for f in filers},
        "n_events": len(rs),
        "n_buys": len(buys),
        "n_sells": len(sells),
        "net_direction": "buy" if len(buys) > len(sells) else ("sell" if len(sells) > len(buys) else "mixed"),
        "top_band": top_band,
        "max_disclosure_date": max(r["ReportDate"] for r in rs),
    }


def main():
    rows = json.load(open(f"{WORKDIR}/raw_quiver.json"))
    altdata, altsignal_available = load_altsignal()
    filer_weights, max_tickers_by_filer = compute_filer_weights(rows, K)

    win = []
    for r in rows:
        try:
            rd = date.fromisoformat(r["ReportDate"])
        except Exception:
            continue
        if WINDOW_START <= rd <= TODAY and r.get("Ticker"):
            win.append(r)

    by_tkr = defaultdict(list)
    for r in win:
        by_tkr[r["Ticker"]].append(r)

    qualified = {}
    for tkr, rs in by_tkr.items():
        big_band = any(band_rank(r["Range"]) >= BIG_BAND_MIN_RANK for r in rs)
        ds = sorted((date.fromisoformat(r["ReportDate"]), r["Representative"]) for r in rs)
        cluster = False
        for i, (d0, _) in enumerate(ds):
            members = {m for d, m in ds if d0 <= d <= d0 + timedelta(days=14)}
            if len(members) >= 3:
                cluster = True
                break
        alt_fires = altsignal_score(tkr, altdata) > 0
        if big_band or cluster or alt_fires:
            qualified[tkr] = rs

    scores = {}
    discounted_filers = {}
    for tkr, rs in qualified.items():
        weighted = score_ticker(tkr, rs, altdata, filer_weights, force_unweighted=False)
        unweighted = score_ticker(tkr, rs, altdata, filer_weights, force_unweighted=True)
        weighted["conviction_unweighted"] = unweighted["conviction"]
        weighted["weighting_delta"] = round(weighted["conviction"] - unweighted["conviction"], 1)
        scores[tkr] = weighted
        for f in weighted["filers"]:
            if filer_weights.get(f, 1.0) < 1.0:
                discounted_filers[f] = {
                    "weight": filer_weights[f],
                    "max_tickers_on_any_report_date": max_tickers_by_filer.get(f),
                }

    survivors = {t: s for t, s in scores.items() if s["conviction"] >= 40}

    json.dump(scores, open(f"{WORKDIR}/stage1_scores.json", "w"), indent=1)

    with open(f"{WORKDIR}/filings_survivors.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["ticker", "transaction_date", "disclosure_date", "filer", "type", "band", "party"])
        for tkr in survivors:
            for r in sorted(qualified[tkr], key=lambda x: x["ReportDate"]):
                cw.writerow([tkr, r["TransactionDate"], r["ReportDate"],
                             r["Representative"], r["Transaction"], r["Range"], r.get("Party", "")])

    out = {
        "today": TODAY.isoformat(),
        "window_start": WINDOW_START.isoformat(),
        "rows_total": len(rows),
        "rows_in_window": len(win),
        "tickers_in_window": len(by_tkr),
        "tickers_qualified": len(qualified),
        "survivors": {t: survivors[t] for t in sorted(survivors, key=lambda x: -survivors[x]["conviction"])},
        "max_report_date_in_feed": max(r["ReportDate"] for r in rows),
        "altsignal_available": altsignal_available,
        "filer_k": K,
        "filer_weights_all": filer_weights,
        "discounted_filers": discounted_filers,
        "script_version": "stage1a.py v2 (2026-08-23) -- AltSignal computed, filer weighting live",
    }
    json.dump(out, open(f"{WORKDIR}/stage1a_today.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
