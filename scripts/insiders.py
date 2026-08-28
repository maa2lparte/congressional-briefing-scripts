#!/usr/bin/env python3
"""
insiders.py — Stage 1A-insider: corporate insider (SEC Form 4) selection.

Sibling to stage1a.py. Same job, different source: Quiver's insider-trading
dataset (Form 4 transactions by officers, directors and 10% owners).

## HOW THE DATA GETS HERE (changed 2026-08-03 — read this before editing)

The Quiver REST endpoint /beta/live/insiders returns HTTP 403 on this
subscription. The dataset IS available, but only through the **QuiverQuant
MCP server** (tool: `get_insider_trading`).

**A Python script in the sandbox cannot call an MCP tool.** MCP tools are
invoked by the agent, not by subprocesses. So the fetch and the scoring are
deliberately split:

    1. The task runner (Claude) calls get_insider_trading, paginating with
       transaction_code="P", and writes the raw rows to a JSON file.
    2. This script reads that file with --from-file and does all the
       filtering, scoring and auditing.

That keeps every scoring rule in version-controlled, testable code rather than
in a prompt, which is the whole point of having a script. The legacy curl path
is retained behind --try-rest purely so the script still works if REST access
is ever granted; it is not used by the pipeline.

## WINDOW: 7 DAYS, NOT 45

Deliberately different from stage1a.py. Congress discloses on a 30-45 day lag,
so a 45-day window is required there. **Form 4 must be filed within 2 business
days of the transaction**, so insider data has almost no lag and a 7-day window
captures essentially all fresh signal. A 45-day insider window would just drag
in stale filings and inflate cluster counts. Override with
CB_INSIDER_WINDOW_DAYS if you disagree.

## SCORING (role + cluster weighted) — deliberately NOT the congressional formula

Form 4 discloses exact share counts and prices, not dollar bands, and has no
committee or party dimension.

    Conviction = Role x 0.35 + Cluster x 0.30 + Value x 0.35

  Role     highest-ranking insider among the qualifying buyers (CEO > CFO >
           President/COO > other C-suite > officer > director > 10% owner).
           When officer_title is null the feed's is_officer / is_director
           booleans are used as a fallback — roughly 40% of real rows have a
           null title but a populated flag, so skipping this would misclassify
           a large share of genuine director purchases as "unknown".
  Cluster  count of DISTINCT insiders making qualifying purchases in the window.
           Unlike Congress, a single CEO purchase is genuinely informative, so
           the ladder starts at 30 for one buyer rather than at zero.
  Value    aggregate USD of qualifying open-market purchases.

Survivor threshold is 40, matching stage1a.py.

HARD FILTER, APPLIED BEFORE ANY SCORING: only open-market purchases count
(transaction code 'P', acquired). Option exercises (M, X), grants and awards
(A), tax withholding (F), gifts (G), conversions (C) and dispositions to the
issuer (D) are all excluded — they are compensation mechanics or scheduled
events, not decisions to buy. This is the insider equivalent of the
congressional non-discretionary exclusion, applied up front rather than as an
audit adjustment.

DEPENDENCIES: standard library only.

USAGE
    python insiders.py --from-file raw_insiders.json [YYYY-MM-DD]
    CB_WORKDIR=/path CB_INSIDER_WINDOW_DAYS=7 python insiders.py --from-file ...
"""
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date, timedelta

WORKDIR = os.environ.get("CB_WORKDIR", "/tmp/cb")
ENDPOINT = "https://api.quiverquant.com/beta/live/insiders"
WINDOW_DAYS = int(os.environ.get("CB_INSIDER_WINDOW_DAYS", "7"))
SURVIVOR_THRESHOLD = 40.0

args = [a for a in sys.argv[1:]]
FROM_FILE = None
TRY_REST = False
if "--from-file" in args:
    i = args.index("--from-file")
    FROM_FILE = args[i + 1]
    del args[i:i + 2]
if "--try-rest" in args:
    TRY_REST = True
    args.remove("--try-rest")

TODAY = date.fromisoformat(args[0]) if args else date.today()
WINDOW_START = TODAY - timedelta(days=WINDOW_DAYS)

# ---------------------------------------------------------------- schema ---
# Confirmed against live MCP rows 2026-08-03. Aliases retained so the script
# also tolerates the REST spelling if that path is ever enabled.
FIELD_ALIASES = {
    "ticker":      ["ticker", "Ticker", "Symbol", "symbol"],
    "txn_date":    ["date", "Date", "TransactionDate", "transaction_date"],
    "file_date":   ["file_date", "FileDate", "fileDate", "ReportDate", "report_date"],
    "owner":       ["owner", "Owner", "Name", "name", "ReportingName", "reporting_name",
                    "InsiderName", "insider_name"],
    "txn_code":    ["transaction_type", "TransactionCode", "transaction_code",
                    "TransactionType", "Transaction", "Code", "code"],
    "acq_disp":    ["acquired_disposed", "AcquiredDisposed", "AcquiredDisposedCode",
                    "acquired_disposed_code", "AcquiredOrDisposed"],
    "shares":      ["shares", "Shares", "TransactionShares", "transaction_shares"],
    "price":       ["price", "Price", "PricePerShare", "price_per_share",
                    "TransactionPricePerShare"],
    "owned_after": ["shares_owned_after", "SharesOwnedFollowing", "SharesOwnedAfter",
                    "shares_owned_following"],
    "title":       ["officer_title", "OfficerTitle", "Title", "title", "Position",
                    "position", "Relationship", "relationship"],
    "is_officer":  ["is_officer", "isOfficer", "IsOfficer"],
    "is_director": ["is_director", "isDirector", "IsDirector"],
    "plan_flag":   ["Rule10b5_1", "rule_10b5_1", "is10b51", "IsPlan", "PlanTransaction",
                    "rule10b5_1"],
}
KNOWN_KEYS = {k for v in FIELD_ALIASES.values() for k in v}

NON_DISCRETIONARY_CODES = {
    "A": "grant/award",
    "M": "option exercise",
    "X": "in-the-money option exercise",
    "F": "tax withholding",
    "G": "gift",
    "C": "conversion",
    "D": "disposition to issuer",
    "I": "discretionary transaction (non-open-market)",
    "J": "other",
}
SALE_CODES = {"S"}
PURCHASE_CODES = {"P"}


def pick(row, field):
    for key in FIELD_ALIASES[field]:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def truthy(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def parse_date(v):
    if not v:
        return None
    s = str(v)[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


# ------------------------------------------------------------- role tier ---
# ORDER IS LOAD-BEARING. The vice-president tiers must be tested BEFORE the
# plain "president" tier, otherwise "Vice President" matches \bpresident\b and
# is wrongly promoted to the 75 tier. CEO/CFO are tested before EVP so that
# "Executive Vice President and CFO" scores as CFO, which is correct.
ROLE_TIERS = [
    (r"chief executive|\bCEO\b",                                  100, "CEO"),
    (r"chief financial|\bCFO\b",                                   90, "CFO"),
    (r"exec(utive)?\s+vice\s+president|\bEVP\b",                   65, "Other C-suite/EVP"),
    (r"senior\s+vice\s+president|\bSVP\b|vice\s+president|\bVP\b", 50, "Officer/VP"),
    (r"\bpresident\b|chief operating|\bCOO\b",                     75, "President/COO"),
    (r"\bchief\b|\bC[A-Z]O\b",                                     65, "Other C-suite/EVP"),
    (r"\bofficer\b",                                               50, "Officer/VP"),
    (r"\bdirector\b|\bchairman\b|\bboard\b",                       40, "Director"),
    (r"10\s*%|ten percent|beneficial owner",                       30, "10% owner"),
]


def role_weight(title, is_officer=False, is_director=False):
    """Return (weight, label).

    Title text wins when it matches a tier. When the title is null — common in
    real Form 4 data — fall back to the feed's boolean flags before giving up
    and calling it unknown."""
    if title:
        t = str(title)
        for pattern, weight, label in ROLE_TIERS:
            if re.search(pattern, t, re.IGNORECASE):
                return weight, label
    if is_officer:
        return 50, "Officer (flag)"
    if is_director:
        return 40, "Director (flag)"
    return 35, "unknown"


def cluster_weight(n):
    """Distinct qualifying buyers. A single insider buy is informative, unlike
    a single congressional filing, so this starts at 30 rather than 0."""
    ladder = {1: 30.0, 2: 55.0, 3: 75.0, 4: 90.0}
    return ladder.get(n, 100.0) if n >= 1 else 0.0


VALUE_LADDER = [
    (50_000, 0.0), (100_000, 25.0), (250_000, 40.0), (500_000, 55.0),
    (1_000_000, 70.0), (5_000_000, 85.0),
]


def value_weight(usd):
    for ceiling, w in VALUE_LADDER:
        if usd < ceiling:
            return w
    return 100.0


def effective_buyers(buys):
    """Collapse buyers who transacted at the SAME PRICE on the SAME DATE.

    ADDED 2026-08-13. Ten investors subscribing to one financing at $18.00, or
    an employee purchase plan executing for thirty officers at an identical
    $73.31, is ONE decision reached by one process - not ten or thirty
    independent insider judgements. Scoring that as a broad cluster is the
    insider-side analogue of counting congressional spin-off receipts as
    discretionary purchases, and it was badly inflating the top of the
    survivor table: on 2026-08-12 the four highest-conviction names
    (PNAQ 100.0, BRVE 100.0, BLSM 96.5, LTGO 65.5) plus FEMY 79.0 and
    TSM 79.0 were every one of them a syndicate or a plan.

    A (date, price) group holding two or more distinct owners is treated as
    one coordinated event and contributes ONE effective buyer instead of N.
    Row counts and total USD are untouched - only the CLUSTER component of
    the conviction score changes. Both the adjusted and the unadjusted score
    are emitted so the size of the adjustment is always visible.

    Returns (effective_count, coordinated_groups).
    """
    groups = defaultdict(set)
    for r in buys:
        px = to_float(pick(r, "price"))
        if px is None:
            continue
        day = str(pick(r, "txn_date") or "")[:10]
        groups[(day, round(px, 4))].add(str(pick(r, "owner") or "unknown"))

    distinct = {str(pick(r, "owner") or "unknown") for r in buys}
    coordinated = [{"date": k[0], "price": k[1], "n_owners": len(v),
                    "owners": sorted(v)[:12]}
                   for k, v in groups.items() if len(v) >= 2]
    collapsed = sum(g["n_owners"] - 1 for g in coordinated)
    return max(1, len(distinct) - collapsed), sorted(
        coordinated, key=lambda g: -g["n_owners"])


# ---------------------------------------------------------------- input ---

def load_rows():
    """Returns (status, rows_or_none, detail). Never raises."""
    if FROM_FILE:
        try:
            with open(FROM_FILE) as f:
                payload = json.load(f)
        except Exception as exc:
            return "error", None, f"could not read {FROM_FILE}: {exc}"
        # Accept a bare list, or a wrapper like {"results": [...]}.
        if isinstance(payload, dict):
            for key in ("results", "rows", "data", "insiders"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            return "error", None, (
                f"{FROM_FILE} did not contain a list of rows "
                f"(got {type(payload).__name__})")
        return "ok", payload, f"loaded {len(payload)} rows from {os.path.basename(FROM_FILE)}"

    if not TRY_REST:
        return "no_input", None, (
            "No --from-file given and --try-rest not set. The MCP is the only "
            "working source for this dataset; the task runner must fetch via "
            "get_insider_trading and pass the rows in with --from-file.")

    # Legacy REST path. Retained for the day the plan covers /beta/live/insiders.
    try:
        cfg = json.load(open(os.path.join(WORKDIR, "config.json")))
        api_key = cfg["quiver_api_key"]
    except Exception as exc:
        return "error", None, f"could not read quiver_api_key: {exc}"
    try:
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "60", "-w", "\n%{http_code}",
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "User-Agent: Mozilla/5.0", ENDPOINT],
            capture_output=True, text=True, timeout=90)
    except Exception as exc:
        return "error", None, f"curl failed: {exc}"
    raw = proc.stdout or ""
    if "\n" not in raw:
        return "error", None, "no HTTP status returned by curl"
    body, _, code = raw.rpartition("\n")
    code = code.strip()
    if code == "403":
        return "unavailable", None, (
            "HTTP 403 — /beta/live/insiders is not in this Quiver plan. Use the "
            "QuiverQuant MCP get_insider_trading tool instead.")
    if code != "200":
        return "error", None, f"HTTP {code}"
    try:
        return "ok", json.loads(body), "HTTP 200"
    except json.JSONDecodeError as exc:
        return "error", None, f"HTTP 200 but body is not JSON: {exc}"


def write_out(out):
    json.dump(out, open(f"{WORKDIR}/insiders_today.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- main ---

def main():
    status, rows, detail = load_rows()

    if status != "ok":
        write_out({
            "status": status, "today": TODAY.isoformat(), "detail": detail,
            "survivors": {}, "schema_validated": False,
            "note": ("Insider stage did not contribute this run. This is a clean "
                     "no-op, not a failure of the congressional pipeline."),
        })
        return

    if not rows:
        write_out({"status": "empty", "today": TODAY.isoformat(),
                   "detail": "no rows supplied", "survivors": {},
                   "schema_validated": False})
        return

    # ---- schema capture ----
    seen_keys = sorted({k for r in rows[:500] if isinstance(r, dict) for k in r})
    unmapped = [k for k in seen_keys if k not in KNOWN_KEYS]
    probe = next((r for r in rows if isinstance(r, dict)), {})
    mapped_ok = all(pick(probe, f) is not None
                    for f in ("ticker", "txn_date", "owner", "txn_code"))

    # ---- window filter, on file_date (when it became public) ----
    win, undated, corrupt_dates = [], 0, []
    for r in rows:
        if not isinstance(r, dict):
            continue
        fd = parse_date(pick(r, "file_date"))
        td = parse_date(pick(r, "txn_date"))
        # A transaction cannot occur after it was filed. Real corruption exists
        # in this feed (observed: HPK date 2033-11-18 filed 2022-11-18, RMCF
        # date 2027-07-27 filed 2023-07-28). Record and exclude.
        if fd and td and td > fd:
            corrupt_dates.append({"ticker": pick(r, "ticker"),
                                  "txn_date": str(pick(r, "txn_date"))[:10],
                                  "file_date": str(pick(r, "file_date"))[:10]})
            continue
        eff = fd or td
        if eff is None:
            undated += 1
            continue
        if WINDOW_START <= eff <= TODAY and pick(r, "ticker"):
            win.append(r)

    # ---- hard filter: open-market purchases only ----
    def classify(r):
        code = str(pick(r, "txn_code") or "").strip()
        acq = str(pick(r, "acq_disp") or "").strip().upper()
        head = code[:1].upper()
        low = code.lower()
        if head in PURCHASE_CODES or "purchase" in low or low == "buy":
            return "purchase" if acq in ("", "A") else "excluded:acquired_disposed=" + acq
        if head in SALE_CODES or "sale" in low or "sell" in low:
            return "sale"
        if head in NON_DISCRETIONARY_CODES:
            return "excluded:" + NON_DISCRETIONARY_CODES[head]
        return "excluded:unrecognised_code=" + (code or "blank")

    by_tkr = defaultdict(list)
    for r in win:
        by_tkr[pick(r, "ticker")].append(r)

    excluded_reasons = defaultdict(int)
    scores, qualifying_rows = {}, {}

    for tkr, rs in by_tkr.items():
        buys, sells, bad_price = [], [], 0
        for r in rs:
            kind = classify(r)
            if kind == "purchase":
                sh = to_float(pick(r, "shares"))
                px = to_float(pick(r, "price"))
                if not sh or not px or sh <= 0 or px <= 0:
                    bad_price += 1
                    excluded_reasons["excluded:zero_or_missing_price"] += 1
                    continue
                r = dict(r)
                r["_usd"] = sh * px
                buys.append(r)
            elif kind == "sale":
                sells.append(r)
            else:
                excluded_reasons[kind] += 1

        if not buys:
            continue

        owners = {}
        for r in buys:
            o = str(pick(r, "owner") or "unknown")
            slot = owners.setdefault(o, {"usd": 0.0, "title": None,
                                         "off": False, "dir": False, "n": 0})
            slot["usd"] += r["_usd"]
            slot["n"] += 1
            slot["title"] = slot["title"] or pick(r, "title")
            slot["off"] = slot["off"] or truthy(pick(r, "is_officer"))
            slot["dir"] = slot["dir"] or truthy(pick(r, "is_director"))

        role_scored = [(role_weight(v["title"], v["off"], v["dir"]), o, v)
                       for o, v in owners.items()]
        best = max(role_scored, key=lambda x: x[0][0])
        rw, rlabel = best[0]
        n_owners = len(owners)
        total_usd = sum(v["usd"] for v in owners.values())

        eff_owners, coord_groups = effective_buyers(buys)
        cw_raw = cluster_weight(n_owners)
        cw = cluster_weight(eff_owners)
        vw = value_weight(total_usd)
        conviction = rw * 0.35 + cw * 0.30 + vw * 0.35
        conviction_unadjusted = rw * 0.35 + cw_raw * 0.30 + vw * 0.35

        max_fd = max((parse_date(pick(r, "file_date")) or parse_date(pick(r, "txn_date"))
                      for r in buys), default=None)
        scores[tkr] = {
            "conviction": round(conviction, 1),
            "role_weight": rw, "top_role": rlabel, "top_role_owner": best[1],
            "cluster_weight": cw, "value_weight": vw,
            "cluster_weight_unadjusted": cw_raw,
            "conviction_unadjusted": round(conviction_unadjusted, 1),
            "n_distinct_buyers": n_owners,
            "n_effective_buyers": eff_owners,
            "cluster_collapsed": eff_owners < n_owners,
            "coordinated_groups": coord_groups[:5],
            "buyers": sorted(owners),
            "titles": {o: (v["title"] or
                           ("officer(flag)" if v["off"] else
                            "director(flag)" if v["dir"] else "unknown"))
                       for o, v in owners.items()},
            "total_purchase_usd": round(total_usd, 2),
            "n_buy_rows": len(buys), "n_sell_rows": len(sells),
            "net_direction": ("buy" if len(buys) > len(sells)
                              else "sell" if len(sells) > len(buys) else "mixed"),
            "rows_dropped_bad_price": bad_price,
            "max_file_date": max_fd.isoformat() if max_fd else None,
        }
        qualifying_rows[tkr] = buys

    survivors = {t: s for t, s in scores.items()
                 if s["conviction"] >= SURVIVOR_THRESHOLD}

    # ---- audit: concentration, mirroring the congressional bulk-filer rule ----
    owner_rowcount, owner_tickers = defaultdict(int), defaultdict(set)
    for r in win:
        o = str(pick(r, "owner") or "unknown")
        owner_rowcount[o] += 1
        owner_tickers[o].add(pick(r, "ticker"))
    concentration = [
        {"owner": o, "rows": n, "distinct_tickers": len(owner_tickers[o]),
         "share_of_window_rows": round(n / max(len(win), 1) * 100, 1)}
        for o, n in sorted(owner_rowcount.items(), key=lambda kv: -kv[1])[:5]
    ]
    bulk_suspects = [c for c in concentration
                     if c["distinct_tickers"] >= 20 and c["share_of_window_rows"] >= 10]

    ex_bulk = {}
    if bulk_suspects:
        drop = {c["owner"] for c in bulk_suspects}
        for tkr in survivors:
            keep = [r for r in qualifying_rows[tkr]
                    if str(pick(r, "owner") or "unknown") not in drop]
            if not keep:
                ex_bulk[tkr] = None
                continue
            owners = {}
            for r in keep:
                o = str(pick(r, "owner") or "unknown")
                slot = owners.setdefault(o, {"usd": 0.0, "title": None,
                                             "off": False, "dir": False})
                slot["usd"] += r["_usd"]
                slot["title"] = slot["title"] or pick(r, "title")
                slot["off"] = slot["off"] or truthy(pick(r, "is_officer"))
                slot["dir"] = slot["dir"] or truthy(pick(r, "is_director"))
            rw2 = max(role_weight(v["title"], v["off"], v["dir"])[0]
                      for v in owners.values())
            ex_bulk[tkr] = round(rw2 * 0.35 + cluster_weight(len(owners)) * 0.30
                                 + value_weight(sum(v["usd"] for v in owners.values())) * 0.35, 1)

    plan_flag_present = any(pick(r, "plan_flag") is not None for r in win[:500])

    with open(f"{WORKDIR}/insiders_survivors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "file_date", "txn_date", "owner", "title",
                    "code", "shares", "price", "usd"])
        for tkr in survivors:
            for r in sorted(qualifying_rows[tkr],
                            key=lambda x: str(pick(x, "file_date") or "")):
                w.writerow([tkr, pick(r, "file_date"), pick(r, "txn_date"),
                            pick(r, "owner"), pick(r, "title"), pick(r, "txn_code"),
                            pick(r, "shares"), pick(r, "price"), round(r["_usd"], 2)])

    write_out({
        "status": "ok",
        "today": TODAY.isoformat(),
        "window_start": WINDOW_START.isoformat(),
        "window_days": WINDOW_DAYS,
        "source": detail,
        "rows_total": len(rows),
        "rows_in_window": len(win),
        "rows_undated_dropped": undated,
        "tickers_in_window": len(by_tkr),
        "tickers_with_qualifying_purchases": len(scores),
        "survivors": {t: survivors[t] for t in
                      sorted(survivors, key=lambda x: -survivors[x]["conviction"])},
        "excluded_row_counts": dict(excluded_reasons),
        "audit": {
            "owner_concentration_top5": concentration,
            "bulk_suspects": bulk_suspects,
            "conviction_ex_bulk": ex_bulk,
            "corrupt_date_rows": corrupt_dates[:25],
            "n_corrupt_date_rows": len(corrupt_dates),
            "corrupt_date_note": (
                "Rows whose transaction date falls AFTER their filing date. That "
                "is impossible under Form 4 and indicates source corruption. "
                "Excluded before scoring. Report the count every run — a sudden "
                "jump means the feed has degraded."),
            "rule_10b5_1_flag_available": plan_flag_present,
            "rule_10b5_1_caveat": (
                "The feed exposes no 10b5-1 plan flag, so pre-scheduled plan "
                "purchases CANNOT be distinguished from discretionary ones. This is "
                "the insider-side equivalent of the missing committee data: a known, "
                "unfixable gap in the source that inflates apparent conviction. State "
                "it every run." if not plan_flag_present else
                "A 10b5-1 plan flag is present in the feed — use it to exclude "
                "pre-scheduled purchases before scoring."),
        },
        "schema_seen": seen_keys,
        "unmapped_keys": unmapped,
        "schema_validated": bool(mapped_ok),
        "schema_note": ("Field mapping confirmed against live data." if mapped_ok else
                        "FIELD MAPPING FAILED — required fields (ticker, txn_date, "
                        "owner, txn_code) could not be resolved. Scores are unreliable. "
                        "Compare schema_seen against FIELD_ALIASES and extend it."),
        "cluster_independence": ("ADDED 2026-08-13. Buyers transacting at the SAME PRICE "
                                 "on the SAME DATE are collapsed to ONE effective buyer before "
                                 "the cluster weight is computed - a financing syndicate or an "
                                 "employee purchase plan is one decision, not N. The field "
                                 "conviction_unadjusted shows what the score would have been "
                                 "without the adjustment; cluster_collapsed flags the affected names."),
        "scoring": ("Conviction = Role*0.35 + Cluster*0.30 + Value*0.35, survivors >= 40. "
                    "Open-market purchases only (code P, acquired). Option exercises, "
                    "awards, gifts, tax withholding and conversions excluded before scoring."),
    })


if __name__ == "__main__":
    main()
