#!/usr/bin/env python3
"""
insiders_short.py -- Stage 1S-insider: discretionary insider SELLING (Form 4)
as a SHORT-side research signal. Sibling of insiders.py (which scores BUYING).

ADDED 2026-09-21 (policy v14, SHORT_RESEARCH). RESEARCH-ONLY: this script's
output feeds short *suggestions* in the daily report. Nothing here creates an
order.

WHY SELLING NEEDS DIFFERENT RULES THAN BUYING
Insiders buy for one reason (they think the stock will go up). They sell for
many: diversification, taxes, house purchases, scheduled 10b5-1 plans, option
exercise-and-sell compensation mechanics, fund distributions. Most insider
selling is uninformative. So this scorer is deliberately stricter than the
buy side and leans on the one feature that separates a decision from routine
liquidation: HOW MUCH OF THEIR STAKE the insider sold.

HARD FILTERS (before scoring)
  1. Open-market sales only: TransactionCode 'S', AcquiredDisposed 'D',
     positive price.
  2. EXERCISE-AND-SELL excluded: if the same owner has an option exercise or
     conversion (M, X, C acquired) in the same ticker within +/-1 day of the
     sale, the sale is treated as compensation liquidation and dropped.
  3. BULK-SUSPECT SELLERS excluded: 10% owners that are not also officers or
     directors, and entities whose name looks institutional (LP, LLC, Fund,
     Capital, Partners, Holdings, Management, Trust, Ventures, Inc, Ltd, plc).
     Funds rebalancing, PE exits and VC distributions are not a view on the
     company from inside it. Mirrors the long side's bulk-suspect exclusion.
  4. Sellers who are neither officers nor directors (no role title, no flag)
     are excluded -- they have no inside view in the relevant sense.
  5. Junk tickers dropped ([NONE], N/A, none, null, empty).
  6. Rows with file date before txn date (impossible) dropped as corrupt.

SCORING
  ShortConviction = Role x 0.30 + Cluster x 0.25 + Value x 0.20 + StakeSold x 0.25

  Role       highest-ranking qualifying seller (same tiers as insiders.py).
  Cluster    distinct EFFECTIVE sellers after same-date/same-price collapse
             (same rule as the buy side, so a coordinated block sale counts
             once). Ladder starts at 20 for one seller -- a lone sale is weaker
             evidence than a lone purchase.
  Value      aggregate USD of qualifying sales (same ladder as buy side).
  StakeSold  largest single seller's sale as a share of their pre-sale DIRECT
             stake (direct_shares_sold / (direct_owned_after + direct_shares_sold)).
             Indirect lines (trusts, LLCs) are ignored because a fully-sold
             indirect line shows a spurious 100%; indirect-only sellers score 0. <10% scores 0
             (routine trimming), 10-25% 40, 25-50% 70, >=50% 100. This is the
             component that does the most work separating a decision from
             routine liquidation.

CONFLICT: if the same ticker ALSO has qualifying open-market insider PURCHASES
in the window, it is flagged conflict=True. The screener treats that as
disqualifying for a short.

KNOWN, UNFIXABLE GAP: no 10b5-1 plan flag in the feed. Pre-scheduled plan
sales cannot be separated from discretionary ones, and plan sales are the
single largest category of insider selling. State this every run.

USAGE
    python insiders_short.py --from-file raw_insiders.json [YYYY-MM-DD]
Writes $CB_WORKDIR/insiders_short_today.json and prints the same JSON.
Standard library only.
"""
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta

WORKDIR = os.environ.get("CB_WORKDIR", "/tmp/cb")
WINDOW_DAYS = int(os.environ.get("CB_INSIDER_WINDOW_DAYS", "7"))
SURVIVOR_THRESHOLD = float(os.environ.get("CB_SHORT_INSIDER_FLOOR", "60"))
SCRIPT_VERSION = "insiders_short.py v1 (2026-09-21)"

args = sys.argv[1:]
FROM_FILE = None
if "--from-file" in args:
    i = args.index("--from-file")
    FROM_FILE = args[i + 1]
    del args[i:i + 2]
TODAY = date.fromisoformat(args[0]) if args else date.today()
WINDOW_START = TODAY - timedelta(days=WINDOW_DAYS)

FIELD_ALIASES = {
    "ticker": ["ticker", "Ticker", "Symbol", "symbol"],
    "txn_date": ["date", "Date", "TransactionDate", "transaction_date"],
    "file_date": ["file_date", "FileDate", "fileDate", "ReportDate", "report_date"],
    "owner": ["owner", "Owner", "Name", "name", "ReportingName", "InsiderName"],
    "txn_code": ["transaction_type", "TransactionCode", "transaction_code", "Code"],
    "acq_disp": ["acquired_disposed", "AcquiredDisposedCode", "AcquiredDisposed"],
    "shares": ["shares", "Shares", "TransactionShares"],
    "price": ["price", "Price", "PricePerShare", "price_per_share"],
    "owned_after": ["shares_owned_after", "SharesOwnedFollowing", "SharesOwnedAfter"],
    "title": ["officer_title", "officerTitle", "OfficerTitle", "Title", "title"],
    "is_officer": ["is_officer", "isOfficer", "IsOfficer"],
    "is_director": ["is_director", "isDirector", "IsDirector"],
    "is_ten_pct": ["isTenPercentOwner", "is_ten_percent_owner", "IsTenPercentOwner"],
}
JUNK_TICKERS = {"[NONE]", "N/A", "NONE", "NULL", ""}
INSTITUTIONAL = re.compile(
    r"\b(L\.?P\.?|LLC|L\.L\.C\.|FUND|CAPITAL|PARTNERS|HOLDINGS|MANAGEMENT|TRUST|"
    r"VENTURES|INC\.?|LTD\.?|PLC|CORP\.?|CORPORATION|ADVISORS|INVESTMENTS?|GROUP)\b",
    re.IGNORECASE)

ROLE_TIERS = [
    (r"chief executive|\bCEO\b", 100, "CEO"),
    (r"chief financial|\bCFO\b", 90, "CFO"),
    (r"exec(utive)?\s+vice\s+president|\bEVP\b", 65, "Other C-suite/EVP"),
    (r"senior\s+vice\s+president|\bSVP\b|vice\s+president|\bVP\b", 50, "Officer/VP"),
    (r"\bpresident\b|chief operating|\bCOO\b", 75, "President/COO"),
    (r"\bchief\b|\bC[A-Z]O\b", 65, "Other C-suite/EVP"),
    (r"\bofficer\b", 50, "Officer/VP"),
    (r"\bdirector\b|\bchairman\b|\bboard\b", 40, "Director"),
]
VALUE_LADDER = [(50_000, 0.0), (100_000, 25.0), (250_000, 40.0), (500_000, 55.0),
                (1_000_000, 70.0), (5_000_000, 85.0)]


def pick(row, field):
    for k in FIELD_ALIASES[field]:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def truthy(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t") if v is not None else False


def fnum(v):
    try:
        return float(str(v).replace("$", "").replace(",", "")) if v is not None else None
    except ValueError:
        return None


def pdate(v):
    try:
        return date.fromisoformat(str(v)[:10]) if v else None
    except ValueError:
        return None


def role_weight(title, is_officer, is_director):
    if title:
        for pat, w, lab in ROLE_TIERS:
            if re.search(pat, str(title), re.IGNORECASE):
                return w, lab
    if is_officer:
        return 50, "Officer (flag)"
    if is_director:
        return 40, "Director (flag)"
    return 35, "unknown"


def cluster_weight(n):
    return {0: 0.0, 1: 20.0, 2: 45.0, 3: 65.0, 4: 85.0}.get(n, 100.0)


def value_weight(usd):
    for ceiling, w in VALUE_LADDER:
        if usd < ceiling:
            return w
    return 100.0


def stake_weight(pct):
    if pct is None:
        return 0.0
    if pct >= 0.50:
        return 100.0
    if pct >= 0.25:
        return 70.0
    if pct >= 0.10:
        return 40.0
    return 0.0


def load_rows():
    if not FROM_FILE:
        return None, "no --from-file given"
    try:
        payload = json.load(open(FROM_FILE))
    except Exception as exc:
        return None, f"could not read {FROM_FILE}: {exc}"
    if isinstance(payload, dict):
        for k in ("results", "rows", "data", "insiders"):
            if isinstance(payload.get(k), list):
                payload = payload[k]
                break
    if not isinstance(payload, list):
        return None, "input is not a list of rows"
    return payload, f"loaded {len(payload)} rows"


def main():
    rows, note = load_rows()
    if rows is None:
        out = {"status": "no_input", "detail": note, "survivors": {}, "script_version": SCRIPT_VERSION}
        json.dump(out, open(f"{WORKDIR}/insiders_short_today.json", "w"), indent=1)
        print(json.dumps(out, indent=1))
        return

    counts = defaultdict(int)
    exercises = defaultdict(list)     # (ticker, owner) -> [dates]
    purchases = defaultdict(int)      # ticker -> qualifying P rows in window
    sales = []
    for r in rows:
        tkr = str(pick(r, "ticker") or "").strip().upper()
        if tkr in JUNK_TICKERS:
            counts["junk_ticker"] += 1
            continue
        code = str(pick(r, "txn_code") or "").strip().upper()
        ad = str(pick(r, "acq_disp") or "").strip().upper()
        fd, td = pdate(pick(r, "file_date")), pdate(pick(r, "txn_date"))
        if fd is None or not (WINDOW_START <= fd <= TODAY):
            continue
        owner = str(pick(r, "owner") or "unknown")
        if code in ("M", "X", "C") and ad == "A" and td:
            exercises[(tkr, owner)].append(td)
        if code == "P" and ad == "A" and (fnum(pick(r, "price")) or 0) > 0:
            purchases[tkr] += 1
        if code == "S" and ad == "D":
            if td and fd < td:
                counts["corrupt_date"] += 1
                continue
            px = fnum(pick(r, "price"))
            sh = fnum(pick(r, "shares"))
            if not px or px <= 0 or not sh or sh <= 0:
                counts["zero_or_missing_price"] += 1
                continue
            sales.append((tkr, owner, td, fd, px, sh, r))

    by_tkr = defaultdict(list)
    for tkr, owner, td, fd, px, sh, r in sales:
        # exercise-and-sell exclusion
        if td and any(abs((td - d).days) <= 1 for d in exercises.get((tkr, owner), [])):
            counts["excluded:exercise_and_sell"] += 1
            continue
        is_off = truthy(pick(r, "is_officer"))
        is_dir = truthy(pick(r, "is_director"))
        ten = truthy(pick(r, "is_ten_pct"))
        if (ten and not is_off and not is_dir) or (INSTITUTIONAL.search(owner) and not is_off):
            counts["excluded:bulk_suspect_seller"] += 1
            continue
        if not is_off and not is_dir and not role_weight(pick(r, "title"), False, False)[0] > 35:
            counts["excluded:not_officer_or_director"] += 1
            continue
        by_tkr[tkr].append((owner, td, fd, px, sh, r))

    scores = {}
    for tkr, rs in by_tkr.items():
        owners = {}
        for owner, td, fd, px, sh, r in rs:
            o = owners.setdefault(owner, {"usd": 0.0, "shares": 0.0, "owned_after": None,
                                          "title": pick(r, "title"),
                                          "is_off": truthy(pick(r, "is_officer")),
                                          "is_dir": truthy(pick(r, "is_director"))})
            o["usd"] += px * sh
            o["shares"] += sh
            oa = fnum(pick(r, "owned_after"))
            direct = str(r.get("directOrIndirectOwnership") or "D").strip().upper() == "D"
            if direct:
                o["direct_shares"] = o.get("direct_shares", 0.0) + sh
            if oa is not None and direct:
                # the smallest owned_after is the post-final-sale holding
                o["owned_after"] = oa if o["owned_after"] is None else min(o["owned_after"], oa)
        roles = {ow: role_weight(v["title"], v["is_off"], v["is_dir"]) for ow, v in owners.items()}
        top_owner = max(roles, key=lambda k: roles[k][0])
        rw, rlabel = roles[top_owner]

        groups = defaultdict(set)
        for owner, td, fd, px, sh, r in rs:
            groups[(str(td), round(px, 4))].add(owner)
        parent = {o: o for o in owners}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        coordinated = []
        for g in groups.values():
            if len(g) >= 2:
                g = sorted(g)
                coordinated.append(g)
                for o in g[1:]:
                    parent[find(o)] = find(g[0])
        n_distinct = len(owners)
        n_effective = len({find(o) for o in owners})

        stake = {}
        for ow, v in owners.items():
            if v["owned_after"] is not None:
                ds = v.get("direct_shares", 0.0)
                pre = v["owned_after"] + ds
                stake[ow] = ds / pre if pre > 0 and ds > 0 else None
        max_stake_owner = max(stake, key=lambda k: stake[k] or 0) if stake else None
        max_stake = stake.get(max_stake_owner) if max_stake_owner else None

        total_usd = sum(v["usd"] for v in owners.values())
        cw = cluster_weight(n_effective)
        cw_un = cluster_weight(n_distinct)
        vw = value_weight(total_usd)
        sw = stake_weight(max_stake)
        conv = rw * 0.30 + cw * 0.25 + vw * 0.20 + sw * 0.25
        conv_un = rw * 0.30 + cw_un * 0.25 + vw * 0.20 + sw * 0.25
        scores[tkr] = {
            "short_conviction": round(conv, 1),
            "short_conviction_unadjusted": round(conv_un, 1),
            "role_weight": rw, "top_role": rlabel, "top_role_owner": top_owner,
            "cluster_weight": cw, "value_weight": vw, "stake_weight": sw,
            "max_stake_sold_pct": round(max_stake * 100, 1) if max_stake is not None else None,
            "max_stake_seller": max_stake_owner,
            "n_distinct_sellers": n_distinct, "n_effective_sellers": n_effective,
            "cluster_collapsed": n_effective < n_distinct, "coordinated_groups": coordinated,
            "sellers": sorted(owners), "total_sale_usd": round(total_usd, 2),
            "avg_sale_price": round(total_usd / sum(v["shares"] for v in owners.values()), 4),
            "n_sale_rows": len(rs),
            "conflict_insider_buying": purchases.get(tkr, 0) > 0,
            "n_purchase_rows_same_window": purchases.get(tkr, 0),
            "max_file_date": max(str(fd) for _, _, fd, _, _, _ in rs),
        }

    survivors = {t: s for t, s in scores.items() if s["short_conviction"] >= SURVIVOR_THRESHOLD}
    out = {
        "status": "ok", "script_version": SCRIPT_VERSION, "today": TODAY.isoformat(),
        "window_start": WINDOW_START.isoformat(), "window_days": WINDOW_DAYS, "source": note,
        "floor": SURVIVOR_THRESHOLD, "n_sale_rows_in_window": len(sales),
        "tickers_with_qualifying_sales": len(scores), "excluded_row_counts": dict(counts),
        "rule_10b5_1_flag_available": False,
        "rule_10b5_1_caveat": "No 10b5-1 plan flag in the feed. Pre-scheduled plan sales -- the largest single category of insider selling -- cannot be separated from discretionary ones. Every insider-sell score is inflated by an unknown amount. State this every run.",
        "survivors": dict(sorted(survivors.items(), key=lambda kv: -kv[1]["short_conviction"])),
        "all_scored_count": len(scores),
    }
    json.dump(out, open(f"{WORKDIR}/insiders_short_today.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
