#!/usr/bin/env python3
"""db_sync.py -- move pipeline state between the run's working directory and
Supabase (Postgres via its REST API), so state never passes through the model's
context and never needs re-typing.

Reads supabase_url / supabase_key from $CB_WORKDIR/config.json. The key is never
printed, logged or written anywhere else.

Commands
  check                      connectivity + all 7 tables reachable
  pull  --date D             write prev_state.json: latest run before D (watch list,
                             latest signal per ticker/source, positions, open drafts)
  push  --date D [--label L] upsert this run from the files the run already writes:
                               journal-D.json      -> journal
                               trades-D.json       -> runs (account/health), positions, drafts
                               filing_tracker-D.json -> signals
                               history-D.json      -> watchlist
                               fills.json (optional, IBKR get_account_trades output) -> fills
  push ... --dry-run         build every payload and print row counts, send nothing

Every write is an idempotent upsert on the table's primary key, so re-running a
push for the same date/label is safe (same-day re-dispatch uses --label run2).
Exit codes: 0 ok, 2 config/connectivity problem, 3 a table write failed.
"""
import argparse
import json
import os
import sys
from datetime import date

import requests

WORKDIR = os.environ.get("CB_WORKDIR", "/tmp/cb")
TABLES = ["runs", "watchlist", "signals", "journal", "positions", "drafts", "fills"]
TIMEOUT = 30


def wp(*p):
    return os.path.join(WORKDIR, *p)


def load(name, default=None):
    try:
        return json.load(open(wp(name)))
    except Exception:
        return default


class DB:
    def __init__(self):
        cfg = load("config.json", {}) or {}
        self.url = (cfg.get("supabase_url") or "").strip().rstrip("/")
        if self.url.endswith("/rest/v1"):          # accept the Data API URL as copied from the dashboard
            self.url = self.url[: -len("/rest/v1")]
        key = cfg.get("supabase_key") or ""
        if not self.url.startswith("https://") or not key or "PASTE" in key:
            sys.stderr.write("config.json is missing supabase_url / supabase_key\n")
            sys.exit(2)
        self.h = {"apikey": key, "Authorization": f"Bearer {key}",
                  "Content-Type": "application/json"}

    def get(self, table, params):
        r = requests.get(f"{self.url}/rest/v1/{table}", headers=self.h, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def upsert(self, table, rows, on_conflict=None):
        if not rows:
            return 0
        h = dict(self.h, Prefer="resolution=merge-duplicates,return=minimal")
        params = {"on_conflict": on_conflict} if on_conflict else {}
        for i in range(0, len(rows), 500):
            r = requests.post(f"{self.url}/rest/v1/{table}", headers=h, params=params,
                              data=json.dumps(rows[i:i + 500], default=str), timeout=TIMEOUT)
            if r.status_code >= 300:
                raise RuntimeError(f"{table}: HTTP {r.status_code} {r.text[:300]}")
        return len(rows)


# ------------------------------------------------------------------ builders
def num(x):
    try:
        return None if x is None or x == "" else float(x)
    except Exception:
        return None


JOURNAL_COLS = ["held", "source", "close", "outlook", "outlook_score", "pct_b", "z20", "rsi14",
                "atr_pct", "above_sma20", "above_sma50", "rel_1m", "rel_3m", "ml_p10d",
                "signal_age_sessions", "first_failing_gate", "disposition", "stop",
                "stop_floor_fired", "rank_score"]


def build(d, label):
    j = load(f"journal-{d}.json", {}) or {}
    tr = load(f"trades-{d}.json", {}) or {}
    ft = load(f"filing_tracker-{d}.json", {}) or {}
    hi = load(f"history-{d}.json", {}) or {}
    fl = load("fills.json", None)
    out = {}

    meta = j.get("meta", {})
    out["runs"] = [{"run_date": d, "run_label": label,
                    "policy_version": meta.get("policy") or tr.get("policy_version_mirrored_from") or "unknown",
                    "health": meta or None, "account": tr.get("account")}]

    jr = []
    for r in j.get("rows", []):
        row = {"run_date": d, "run_label": label, "side": r["side"], "ticker": r["ticker"]}
        for c in JOURNAL_COLS:
            row[c] = r.get(c)
        row["stop"] = r.get("stop") if r.get("stop") is not None else r.get("buy_stop")
        row["detail"] = {k: v for k, v in r.items() if k not in JOURNAL_COLS + ["date", "side", "ticker"]}
        jr.append(row)
    # one row per (side, ticker): last write wins inside a run
    out["journal"] = list({(x["side"], x["ticker"]): x for x in jr}.values())

    # positions.json = IBKR get_account_positions snapshot the run already writes for short_screen.py
    ib = (load("positions.json", {}) or {}).get("positions", [])
    stops = {t: s for t, s in (tr.get("positions_status") or {}).items()}
    shorts = tr.get("open_shorts") or {}
    jl = {x["ticker"]: x for x in out["journal"] if x["side"] in ("long", "short_open")}
    pos = []
    for p in ib:
        q = num(p.get("position"))
        if not q:
            continue                                   # flat rows (e.g. a just-closed name)
        t = p["contract_description"]
        st = stops.get(t) or {}
        sh = shorts.get(t) or {}
        pos.append({"run_date": d, "run_label": label, "ticker": t, "qty": q,
                    "avg_price": num(p.get("average_price")),
                    "close": jl.get(t, {}).get("close"),
                    "market_value": num(p.get("market_value")),
                    "stop": st.get("stop") if q > 0 else sh.get("buy_stop"),
                    "floor_fired": st.get("floor_fired") if q > 0 else None,
                    "entered": sh.get("entered")})
    out["positions"] = pos

    dr = []
    for kind, lst in (("long", tr.get("drafts") or []), ("short", tr.get("short_drafts") or [])):
        for x in lst:
            dr.append({"run_date": d, "ticker": x.get("ticker"), "side": x.get("side"),
                       "kind": x.get("kind") or ("short_open" if kind == "short" else "exit" if x.get("side") == "SELL" else "new_position"),
                       "trigger": x.get("trigger"), "qty": x.get("qty"), "limit_price": x.get("limit"),
                       "stage2_close": x.get("stage2_close"), "instruction_id": str(x.get("instruction_id") or "")})
    out["drafts"] = dr

    sig = []
    for src_key, src in (("congress", "congress"), ("insider_held", "insider"), ("insider_new_candidates", "insider")):
        for t, v in (ft.get(src_key) or {}).items():
            sig.append({"run_date": d, "run_label": label, "ticker": t, "source": src,
                        "conviction": num(v[0]), "max_date": v[1] if len(v) > 1 else None,
                        "fresh": t in (ft.get("fresh_today") or []),
                        "bulk_suspect": None, "detail": None})
    out["signals"] = list({(x["ticker"], x["source"]): x for x in sig}.values())

    tags = hi.get("tags") or {}
    out["watchlist"] = [{"ticker": t, "first_seen": d, "last_tag": tags.get(t)} for t in hi.get("tickers", [])]

    fills = []
    for x in ((fl or {}).get("trades") if isinstance(fl, dict) else (fl or [])) or []:
        fills.append({"trade_id": x["trade_id"], "order_id": str(x.get("order_id")), "ticker": x["symbol"],
                      "side": x["side"], "size": x["size"], "price": x["price"], "order_type": x.get("order_type"),
                      "exchange": x.get("exchange"), "commission": x.get("commission"),
                      "realized_pnl": x.get("realized_pnl"), "trade_time": x["trade_time"]})
    out["fills"] = fills
    return out


CONFLICT = {"runs": "run_date,run_label", "watchlist": "ticker", "signals": "run_date,run_label,ticker,source",
            "journal": "run_date,run_label,side,ticker", "positions": "run_date,run_label,ticker",
            "drafts": None, "fills": "trade_id"}


# ------------------------------------------------------------------ commands
def cmd_check(a):
    db = DB()
    bad = {}
    for t in TABLES:
        try:
            db.get(t, {"select": "*", "limit": "1"})
        except Exception as e:
            bad[t] = str(e)[:200]
    print(json.dumps({"ok": not bad, "tables_ok": [t for t in TABLES if t not in bad], "failed": bad}))
    return 0 if not bad else 2


def cmd_push(a):
    payload = build(a.date, a.label)
    counts = {k: len(v) for k, v in payload.items()}
    if a.dry_run:
        print(json.dumps({"dry_run": True, "rows": counts}))
        return 0
    db = DB()
    # watchlist: never overwrite first_seen or drop sources -- only add new tickers, refresh tag
    existing = {r["ticker"] for r in db.get("watchlist", {"select": "ticker"})}
    new = [w for w in payload["watchlist"] if w["ticker"] not in existing]
    old = [{"ticker": w["ticker"], "last_tag": w["last_tag"]} for w in payload["watchlist"]
           if w["ticker"] in existing and w["last_tag"]]
    payload["watchlist"] = new
    errs, written = {}, {}
    for t in ["runs", "watchlist", "signals", "journal", "positions", "drafts", "fills"]:
        try:
            written[t] = db.upsert(t, payload[t], CONFLICT[t])
        except Exception as e:
            errs[t] = str(e)
    for w in old:
        try:
            requests.patch(f"{db.url}/rest/v1/watchlist", headers=db.h,
                           params={"ticker": f"eq.{w['ticker']}"}, data=json.dumps({"last_tag": w["last_tag"]}),
                           timeout=TIMEOUT)
        except Exception as e:
            errs.setdefault("watchlist_tags", str(e)[:200])
    print(json.dumps({"written": written, "errors": errs}))
    return 3 if errs else 0


def cmd_pull(a):
    db = DB()
    runs = db.get("runs", {"select": "run_date,run_label,policy_version,account",
                           "run_date": f"lt.{a.date}", "order": "run_date.desc,run_label.desc", "limit": "1"})
    prev = runs[0] if runs else None
    state = {"as_of": a.date, "previous_run": prev,
             "watchlist": db.get("watchlist", {"select": "ticker,first_seen,last_tag", "order": "first_seen"}),
             "signals_latest": {}, "positions": [], "drafts_open": []}
    # latest signal per (ticker, source) strictly before today -> freshness baseline
    for r in db.get("signals", {"select": "ticker,source,conviction,max_date,run_date",
                                "run_date": f"lt.{a.date}", "order": "run_date.desc"}):
        state["signals_latest"].setdefault(f"{r['ticker']}|{r['source']}", r)
    if prev:
        state["positions"] = db.get("positions", {"select": "*", "run_date": f"eq.{prev['run_date']}",
                                                  "run_label": f"eq.{prev['run_label']}"})
    state["drafts_open"] = db.get("drafts", {"select": "*", "status": "eq.drafted", "order": "run_date.desc"})
    json.dump(state, open(wp("prev_state.json"), "w"), indent=1)
    print(json.dumps({"previous_run": prev and prev["run_date"], "watchlist": len(state["watchlist"]),
                      "signals": len(state["signals_latest"]), "positions": len(state["positions"]),
                      "drafts_open": len(state["drafts_open"])}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("push")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--label", default="run1")
    p.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("pull")
    q.add_argument("--date", default=date.today().isoformat())
    a = ap.parse_args()
    sys.exit({"check": cmd_check, "push": cmd_push, "pull": cmd_pull}[a.cmd](a))


if __name__ == "__main__":
    main()
