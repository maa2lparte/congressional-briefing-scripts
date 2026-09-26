#!/usr/bin/env python3
"""parallel_runner.py -- deterministic, read-only parallel data gathering for the
congressional-briefing pipeline (proposed in policy v17, run_mechanics).

WHAT IT DOES
  Phase A (network, concurrent): the four bulk Quiver REST pulls at once
      bulk/congresstrading -> bulk.json          (~50MB)
      live/insiders        -> raw_insiders.json
      live/wallstreetbets  -> wsb.json
      live/govcontractsall -> govcontracts.json
  Phase B (after the run's ticker list is known, concurrent):
      technicals_v2.py     chunked across CPUs (BLAS pinned to 1 thread/proc)
      supplement_batch.py  one invocation, benchmarks fetched once, FLAT output
      dark pool            beta/historical/offexchange/{T} -> dp/{T}.json
                           (REST-to-disk, so rows never enter model context)
  Validation: fails loudly (exit 2) on NaN closes, missing tickers, nested
  supplement shape, or empty bulk files -- the three silent-corruption modes
  seen so far.

WHAT IT DOES NOT DO
  No scoring decisions, no ranking, no IBKR calls, no Drive writes. It only
  fetches and computes, then writes run_manifest.json with timings and
  failures. Everything that needs judgment stays with the model/human.

Usage
  CB_WORKDIR=/tmp/cb YF_DISABLE_CURL_CFFI=1 \
    python3 parallel_runner.py fetch                    # Phase A
  CB_WORKDIR=/tmp/cb YF_DISABLE_CURL_CFFI=1 \
    python3 parallel_runner.py enrich --tickers-file combined_list.txt [--no-darkpool]
  python3 parallel_runner.py validate --tickers-file combined_list.txt

  The API key is read from $CB_WORKDIR/config.json ("quiver_api_key"),
  exactly like insiders.py.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKDIR = os.environ.get("CB_WORKDIR", "/tmp/cb")
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
BASE = "https://api.quiverquant.com/beta"
BULK = {
    "bulk.json": f"{BASE}/bulk/congresstrading",
    "raw_insiders.json": f"{BASE}/live/insiders",
    "wsb.json": f"{BASE}/live/wallstreetbets",
    "govcontracts.json": f"{BASE}/live/govcontractsall",
}
DP_WORKERS = 6        # Quiver REST concurrency cap (be polite; tune down on 429)
CURL_TIMEOUT = 180


def wp(*p):
    return os.path.join(WORKDIR, *p)


def api_key():
    return json.load(open(wp("config.json")))["quiver_api_key"]


def curl_to_disk(url, dest, key, tries=3):
    """curl straight to disk; returns (ok, http_code, bytes, seconds)."""
    t0 = time.time()
    code = "000"
    for i in range(tries):
        tmp = dest + ".part"
        p = subprocess.run(
            ["curl", "-s", "--max-time", str(CURL_TIMEOUT), "-o", tmp,
             "-w", "%{http_code}", "-H", f"Authorization: Bearer {key}",
             "-H", "Accept: application/json", "-H", "User-Agent: Mozilla/5.0", url],
            capture_output=True, text=True, timeout=CURL_TIMEOUT + 30)
        code = (p.stdout or "000").strip()
        if code == "200":
            os.replace(tmp, dest)
            return True, code, os.path.getsize(dest), round(time.time() - t0, 1)
        if code in ("401", "403", "404"):
            break                           # not retryable
        time.sleep(2 * (i + 1))             # 429 / 5xx / timeout
    return False, code, 0, round(time.time() - t0, 1)


def read_tickers(path):
    raw = open(path if os.path.isabs(path) else wp(path)).read().replace(",", " ").split()
    # combined_list.txt starts with a count line ("30"); skip pure numbers.
    return sorted({t.strip().upper() for t in raw if t.strip() and not t.strip().isdigit()})


# ---------------------------------------------------------------- Phase A
def cmd_fetch(a):
    key = api_key()
    res = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(BULK)) as ex:
        futs = {ex.submit(curl_to_disk, url, wp(fn), key): fn for fn, url in BULK.items()}
        for f in as_completed(futs):
            ok, code, n, s = f.result()
            res[futs[f]] = {"ok": ok, "http": code, "bytes": n, "seconds": s}
    manifest("fetch", {"files": res, "wall_seconds": round(time.time() - t0, 1)})
    # insiders 403 is an expected plan limitation, not a runner failure --
    # insiders.py then falls back to the MCP rows passed with --from-file.
    hard = [k for k, v in res.items() if not v["ok"] and not (k == "raw_insiders.json" and v["http"] == "403")]
    return 2 if hard else 0


# ---------------------------------------------------------------- Phase B
def run_technicals(tickers, nproc):
    chunks = [tickers[i::nproc] for i in range(nproc)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs, outs = [], []
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        out = wp(f"v2_part{i}.json")
        outs.append(out)
        log = open(wp(f"technicals_part{i}.log"), "w")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(SCRIPTS, "technicals_v2.py"), *ch, "--json", out],
            stdout=log, stderr=subprocess.STDOUT, env=env))
    rcs = [p.wait() for p in procs]
    merged = []
    for o in outs:
        try:
            merged.extend(json.load(open(o)))
        except Exception:
            pass
    json.dump(merged, open(wp("v2_today.json"), "w"), indent=1)
    return {"chunks": len(outs), "returncodes": rcs, "n_out": len(merged)}


def run_supplement(tickers, workers):
    p = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "supplement_batch.py"),
         "--out", wp("supplement_today.json"), "--workers", str(workers), *tickers],
        capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": (p.stderr or "")[-500:], "returncode": p.returncode}


def run_darkpool(tickers):
    key = api_key()
    os.makedirs(wp("dp"), exist_ok=True)
    res = {}
    with ThreadPoolExecutor(max_workers=DP_WORKERS) as ex:
        futs = {ex.submit(curl_to_disk, f"{BASE}/historical/offexchange/{t}", wp("dp", f"{t}.json"), key): t
                for t in tickers}
        for f in as_completed(futs):
            ok, code, n, s = f.result()
            res[futs[f]] = {"ok": ok, "http": code}
    return {"n_ok": sum(v["ok"] for v in res.values()),
            "failed": {t: v["http"] for t, v in res.items() if not v["ok"]}}


def cmd_enrich(a):
    tickers = read_tickers(a.tickers_file)
    nproc = a.procs or max(1, os.cpu_count() or 1)
    t0 = time.time()
    timings, out = {}, {}

    def timed(name, fn, *args):
        s = time.time()
        r = fn(*args)
        timings[name] = round(time.time() - s, 1)
        return name, r

    jobs = [("technicals", run_technicals, tickers, nproc),
            ("supplement", run_supplement, tickers, a.supp_workers)]
    if not a.no_darkpool:
        jobs.append(("darkpool", run_darkpool, tickers))
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for f in as_completed([ex.submit(timed, *j) for j in jobs]):
            name, r = f.result()
            out[name] = r
    out["timings_seconds"] = timings
    out["wall_seconds"] = round(time.time() - t0, 1)
    out["n_tickers"] = len(tickers)
    v = validate(tickers, check_dp=not a.no_darkpool)
    out["validation"] = v
    manifest("enrich", out)
    print(json.dumps(out, indent=1))
    return 0 if v["ok"] else 2


# ---------------------------------------------------------------- validation
def _nan(x):
    return isinstance(x, float) and math.isnan(x)


def validate(tickers, check_dp=True):
    problems = []
    try:
        supp = json.load(open(wp("supplement_today.json")))
    except Exception as e:
        supp = {}
        problems.append(f"supplement unreadable: {e}")
    for t, d in supp.items():
        if isinstance(d, dict) and set(d) == {t}:
            problems.append(f"supplement NESTED for {t} ({{T:{{T:...}}}}) -- flatten before short_screen gate")
            break
    try:
        v2 = {d.get("ticker"): d for d in json.load(open(wp("v2_today.json")))}
    except Exception as e:
        v2 = {}
        problems.append(f"v2_today unreadable: {e}")
    missing_s = [t for t in tickers if t not in supp]
    missing_t = [t for t in tickers if t not in v2]
    nan_s = [t for t, d in supp.items() if isinstance(d, dict) and _nan(d.get("close"))]
    nan_t = [t for t, d in v2.items() if _nan(d.get("close"))]
    stale = sorted({d.get("last_bar_date") for d in supp.values() if isinstance(d, dict)} - {None})
    if nan_s or nan_t:
        problems.append(f"NaN close: supplement={nan_s} technicals={nan_t}")
    warnings = []
    if len(stale) > 1:
        # Usually Yahoo publishing a NaN latest bar for some symbols (dropped by
        # the fetchers). Not fatal, but the facilitator must name the lagging
        # tickers, because their relative-strength terms are a day behind.
        newest = stale[-1]
        lag = sorted(t for t, d in supp.items() if isinstance(d, dict) and d.get("last_bar_date") != newest)
        warnings.append(f"{len(lag)} tickers one+ bar behind {newest}: {lag}")
    dp_missing = []
    if check_dp:
        dp_missing = [t for t in tickers if not os.path.exists(wp("dp", f"{t}.json"))]
    for fn in BULK:
        if os.path.exists(wp(fn)) and os.path.getsize(wp(fn)) < 3:
            problems.append(f"{fn} is empty")
    # Missing tickers are reported, not fatal: some symbols legitimately lack history.
    return {"ok": not problems, "problems": problems, "warnings": warnings,
            "missing_supplement": missing_s, "missing_technicals": missing_t,
            "missing_darkpool": dp_missing, "last_bar_dates": stale}


def cmd_validate(a):
    v = validate(read_tickers(a.tickers_file), check_dp=not a.no_darkpool)
    print(json.dumps(v, indent=1))
    return 0 if v["ok"] else 2


def manifest(stage, data):
    path = wp("run_manifest.json")
    m = {}
    if os.path.exists(path):
        try:
            m = json.load(open(path))
        except Exception:
            m = {}
    m[stage] = data
    json.dump(m, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    e = sub.add_parser("enrich")
    e.add_argument("--tickers-file", required=True)
    e.add_argument("--procs", type=int, default=0)
    e.add_argument("--supp-workers", type=int, default=4)
    e.add_argument("--no-darkpool", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--tickers-file", required=True)
    v.add_argument("--no-darkpool", action="store_true")
    a = ap.parse_args()
    sys.exit({"fetch": cmd_fetch, "enrich": cmd_enrich, "validate": cmd_validate}[a.cmd](a))


if __name__ == "__main__":
    main()
