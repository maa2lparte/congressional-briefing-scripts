#!/usr/bin/env python3
"""
short_screen.py -- Stage 3S: SHORT RESEARCH screen. RESEARCH-ONLY.

ADDED 2026-09-21 (policy v14, SHORT_RESEARCH). Produces short *suggestions*
for the daily report. It never creates, drafts or submits an order, and the
pipeline must not call create_order_instruction for anything this script
outputs unless a future policy version explicitly changes
SHORT_RESEARCH.mode from RESEARCH_ONLY.

Two sub-commands, run in this order by the daily pipeline:

  1) python short_screen.py candidates
       Reads  insiders_short_today.json  (from insiders_short.py)
              stage1_scores.json         (from stage1a_v2.py)
              insiders_today.json        (from insiders.py -- buy-side conflicts)
              positions.json             ({"positions":[...]} from IBKR; held longs)
       Writes short_candidates.json and prints the ticker list that needs a
       Stage 2 technicals + supplement pass (tickers the run does not already
       have panels for should be run through technicals_v2.py / supplement.py).

  2) python short_screen.py gate --tech <v2.json> [--tech <v2b.json> ...]
                                 --supp <supplement_merged.json> [--supp ...]
       Applies the blocking short gates (S1..S11, see GATES below), fetches
       short-interest and 90d dollar volume from yfinance (set
       YF_DISABLE_CURL_CFFI=1), and writes short_research_today.json with a
       reference plan per passing name.

GATES (all blocking unless marked FLAG)
  S1  data_quality       full technicals panel present
  S2  conviction         insider-sell >= floor AND max stake sold >= 10%
                         OR congressional net-sell, conviction ex-AltSignal >= 40
  S3  no_buy_conflict    no open-market insider BUYING in the same window
  S4  not_held_long      ticker is not a current long position
  S5  outlook            Weak or Deteriorating (mechanical outlook synthesis,
                         same method as the long side)
  S6  not_oversold       RSI14 >= 30 AND pct_b >= 0.0 AND z20 >= -2.5
                         (mirror of the long-side overextension gate)
  S7  price_floor        Stage 2 close >= $5
  S8  liquidity          avg 90d dollar volume >= $5,000,000 (10x the long floor:
                         you must be able to cover in a hurry)
  S9  squeeze_risk       short % of float <= 20% AND days-to-cover <= 7.
                         Unknown data = FLAG (not block) -- verify before acting.
  S10 earnings           no earnings within 10 calendar days. Unknown = FLAG.
  S11 price_sanity       insider feed avg sale price vs close: >10x apart BLOCK,
                         2x-10x FLAG.
  Borrow/locate availability and borrow fee cannot be queried through the
  current IBKR connector -- ALWAYS reported as "verify in IBKR before acting".

REFERENCE PLAN (not an order): entry_ref = close*(1 - limit_buffer_pct);
buy-stop = entry_ref*(1 + clamp(2*ATR14%, 5%, 12%)), moved up to
nearest_swing_resistance*1.005 if the raw stop sits below that resistance
(mirror of the long-side structural floor); size_ref = whole shares only
(IBKR does not short fractional shares) nearest to the $250 unit without
exceeding it; max loss at stop in dollars.
"""
import json
import math
import os
import sys
from datetime import date, datetime, timedelta

WORKDIR = os.environ.get("CB_WORKDIR", "/tmp/cb")
SCRIPT_VERSION = "short_screen.py v1 (2026-09-21)"
INSIDER_FLOOR = float(os.environ.get("CB_SHORT_INSIDER_FLOOR", "60"))
CONGRESS_FLOOR = 40.0
MIN_STAKE_PCT = 10.0
PRICE_FLOOR = 5.0
LIQ_FLOOR = 5_000_000.0
MAX_SHORT_FLOAT = 0.20
MAX_DAYS_TO_COVER = 7.0
EARNINGS_BLOCK_DAYS = 10
LIMIT_BUFFER_PCT = float(os.environ.get("CB_LIMIT_BUFFER_PCT", "0.005"))
SIZE_UNIT_USD = 250.0
TODAY = date.fromisoformat(os.environ.get("CB_TODAY")) if os.environ.get("CB_TODAY") else date.today()


def jload(name, default=None):
    p = name if os.path.isabs(name) else os.path.join(WORKDIR, name)
    try:
        return json.load(open(p))
    except Exception:
        return default


def congress_ex_altsignal(s):
    bw, cb, bp = s.get("band_weight", 0), s.get("cluster_breadth", 0), s.get("bipartisan")
    if bp is None:
        return (bw * 0.35 + cb * 0.30) / 0.65
    return (bw * 0.35 + cb * 0.30 + bp * 0.15) / 0.80


# ------------------------------------------------------------ candidates ---
def cmd_candidates():
    ins = jload("insiders_short_today.json", {}) or {}
    cong = jload("stage1_scores.json", {}) or {}
    buys = (jload("insiders_today.json", {}) or {}).get("survivors", {}) or {}
    pos = jload("positions.json", {}) or {}
    held = {p.get("contract_description") for p in pos.get("positions", []) if (p.get("position") or 0) > 0}

    cands, dropped = {}, []
    for t, s in (ins.get("survivors") or {}).items():
        cands.setdefault(t, {"sources": [], "insider": None, "congress": None})
        cands[t]["sources"].append("insider_sell")
        cands[t]["insider"] = s
    for t, s in cong.items():
        if s.get("net_direction") != "sell":
            continue
        ex = congress_ex_altsignal(s)
        if ex < CONGRESS_FLOOR:
            continue
        cands.setdefault(t, {"sources": [], "insider": None, "congress": None})
        cands[t]["sources"].append("congress_sell")
        cands[t]["congress"] = {"conviction": s.get("conviction"), "conviction_ex_altsignal": round(ex, 1),
                                "n_buys": s.get("n_buys"), "n_sells": s.get("n_sells"),
                                "n_filers": s.get("n_filers"), "top_band": s.get("top_band"),
                                "max_disclosure_date": s.get("max_disclosure_date")}

    keep = {}
    for t, c in cands.items():
        reasons = []
        i = c["insider"]
        if t in held:
            reasons.append("S4 not_held_long: currently held long")
        if (i and i.get("conflict_insider_buying")) or t in buys:
            reasons.append("S3 no_buy_conflict: insider open-market BUYING in same window")
        if i and not c["congress"]:
            if i.get("max_stake_sold_pct") is None:
                reasons.append("S2 conviction: stake sold unknown (indirect holdings only) -- cannot separate a decision from routine liquidation")
            elif i["max_stake_sold_pct"] < MIN_STAKE_PCT:
                reasons.append(f"S2 conviction: max stake sold {i['max_stake_sold_pct']}% < {MIN_STAKE_PCT}% (routine trimming)")
            if (i.get("avg_sale_price") or 0) < PRICE_FLOOR * 0.8:
                reasons.append(f"S7 price_floor (pre-check on feed price ${i.get('avg_sale_price')})")
        if reasons:
            dropped.append({"ticker": t, "sources": c["sources"], "blocked_by": reasons})
        else:
            keep[t] = c
    out = {"script_version": SCRIPT_VERSION, "date": TODAY.isoformat(),
           "n_raw_candidates": len(cands), "n_need_technicals": len(keep),
           "need_technicals": sorted(keep), "candidates": keep, "pre_dropped": dropped}
    json.dump(out, open(os.path.join(WORKDIR, "short_candidates.json"), "w"), indent=1)
    print(" ".join(sorted(keep)))


# ------------------------------------------------------ outlook (shared) ---
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def outlook(t, s):
    """Same mechanical category-vote synthesis the long side uses (outlook_score.py
    from the 2026-09-18 run). A first pass -- the run may override with stated
    reasoning, exactly as on the long side."""
    tr, mo, vo, vl, st = t["trend"], t["momentum"], t["volatility"], t["volume"], t["stats"]
    ema = tr["ema"]
    stack_bull = ema["12"] > ema["26"] > ema["50"]
    stack_bear = ema["12"] < ema["26"] < ema["50"]
    cloud = tr["ichimoku"]["price_vs_cloud"]
    st_up, sar_up = tr["supertrend_dir"] == "up", tr["parabolic_sar_dir"] == "up"
    di_bull = tr["plus_di"] > tr["minus_di"] and tr["adx"] >= 20
    di_bear = tr["minus_di"] > tr["plus_di"] and tr["adx"] >= 20
    trend = (1 if stack_bull else -1 if stack_bear else 0) + (1 if cloud == "above" else -1 if cloud == "below" else 0)
    trend += (1 if st_up else -1) + (1 if sar_up else -1) + (1 if di_bull else -1 if di_bear else 0)
    if s and s.get("sma50_slope_20d") is not None:
        trend += 1 if s["sma50_slope_20d"] > 0 else -1
    if s and s.get("sma200_slope_20d") is not None:
        trend += 0.5 if s["sma200_slope_20d"] > 0 else -0.5
    rsi = mo["rsi14"]
    mom = 0.5 if rsi > 70 else 1 if rsi > 55 else -1 if rsi < 30 else -0.5 if rsi < 45 else 0
    mom += (1 if mo["stochastic_k"] > mo["stochastic_d"] else -1) + (1 if mo["roc_10"] > 0 else -1)
    mom += (1 if mo["roc_20"] > 0 else -1) + (1 if mo["cci"] > 0 else -1)
    if mo["mfi"] < rsi - 5:
        mom -= 0.5
    vol = (1 if vl["obv_slope_20"] == "rising" else -1) + (1 if vl["ad_line_slope_20"] == "rising" else -1)
    vol += (1 if vl["cmf"] > 0 else -1) + (0.5 if vl["klinger_above_signal"] else -0.5)
    rel = 0
    r1 = r3 = None
    if s and "relative" in s:
        r1, r3 = s["relative"]["1M"]["excess_vs_spy"], s["relative"]["3M"]["excess_vs_spy"]
        rel = (1 if r1 > 0 else -1) + (1 if r3 > 0 else -1)
    score = round(clamp(trend * 0.35 + mom * 0.35 + vol * 0.35 + rel * 0.55, -8, 8), 2)
    pct_b, z20 = vo["bollinger"]["pct_b"], st["zscore_vs_sma20"]
    if pct_b > 1.0 or z20 > 2.5:
        term = "Constructive" if score > 1.5 else "Neutral"
    elif score >= 2.5:
        term = "Constructive"
    elif score >= 0.5:
        term = "Neutral" if score < 1.2 else "Constructive"
    elif score > -1.5:
        term = "Neutral"
    elif score > -3.5:
        term = "Deteriorating"
    else:
        term = "Weak"
    return term, score, r1, r3


# ------------------------------------------------------ yfinance helpers ---
def yf_extra(tkr):
    """Short interest, days-to-cover, 90d avg dollar volume. Never raises."""
    out = {"short_pct_float": None, "days_to_cover": None, "avg_90d_usd_volume": None, "yf_error": None}
    try:
        import yfinance as yf
        t = yf.Ticker(tkr)
        h = t.history(period="90d")
        if len(h):
            out["avg_90d_usd_volume"] = float((h["Close"] * h["Volume"]).mean())
        try:
            info = t.info or {}
            out["short_pct_float"] = info.get("shortPercentOfFloat")
            out["days_to_cover"] = info.get("shortRatio")
        except Exception as e:  # info endpoint is flaky
            out["yf_error"] = f"info: {e}"[:120]
    except Exception as e:
        out["yf_error"] = str(e)[:120]
    return out


# ------------------------------------------------------------------ gate ---
def cmd_gate(tech_files, supp_files):
    cand = jload("short_candidates.json", {}) or {}
    tech = {}
    for f in tech_files:
        for r in jload(f, []) or []:
            tech[r.get("ticker")] = r
    supp = {}
    for f in supp_files:
        supp.update(jload(f, {}) or {})
    alt = jload("altsignal.json", {}) or {}

    passed, blocked = [], []
    for tkr, c in sorted((cand.get("candidates") or {}).items()):
        t, s = tech.get(tkr), supp.get(tkr)
        flags, reasons = [], []
        row = {"ticker": tkr, "sources": c["sources"], "insider": c["insider"], "congress": c["congress"]}
        if not t or "trend" not in t:
            blocked.append({**row, "blocked_by": ["S1 data_quality: no full technicals panel"]})
            continue
        close = t["close"]
        term, score, r1, r3 = outlook(t, s)
        rsi, pct_b, z20 = t["momentum"]["rsi14"], t["volatility"]["bollinger"]["pct_b"], t["stats"]["zscore_vs_sma20"]
        atr = t["volatility"]["atr_pct"]
        row.update({"close": close, "outlook": term, "outlook_score": score, "rsi14": rsi,
                    "pct_b": pct_b, "z20": z20, "atr_pct": atr, "rel_1m_vs_spy": r1, "rel_3m_vs_spy": r3,
                    "name": (s or {}).get("name")})
        if term not in ("Weak", "Deteriorating"):
            reasons.append(f"S5 outlook: {term} (needs Weak/Deteriorating)")
        if rsi < 30 or pct_b < 0.0 or z20 < -2.5:
            reasons.append(f"S6 not_oversold: RSI {rsi}, pct_b {pct_b}, z20 {z20} -- already washed out, bounce risk")
        if close < PRICE_FLOOR:
            reasons.append(f"S7 price_floor: ${close} < ${PRICE_FLOOR}")
        i = c["insider"]
        if i and i.get("avg_sale_price"):
            ratio = close / i["avg_sale_price"]
            row["price_sanity_ratio"] = round(ratio, 2)
            if ratio > 10 or ratio < 0.1:
                reasons.append(f"S11 price_sanity: feed ${i['avg_sale_price']} vs close ${close} ({ratio:.2f}x) -- data defect")
            elif ratio > 2 or ratio < 0.5:
                flags.append(f"S11 price_sanity FLAG: {ratio:.2f}x between feed and close")
        # yfinance calls are rate-limited (~170/session); only spend them on names
        # that have already cleared every other gate.
        if reasons:
            blocked.append({**row, "blocked_by": reasons})
            continue
        yx = yf_extra(tkr)
        row.update(yx)
        if yx["avg_90d_usd_volume"] is None:
            reasons.append("S8 liquidity: 90d dollar volume unavailable")
        elif yx["avg_90d_usd_volume"] < LIQ_FLOOR:
            reasons.append(f"S8 liquidity: ${yx['avg_90d_usd_volume']:,.0f} < ${LIQ_FLOOR:,.0f}")
        spf, dtc = yx["short_pct_float"], yx["days_to_cover"]
        if spf is None or dtc is None:
            flags.append("S9 squeeze_risk FLAG: short interest / days-to-cover unavailable -- verify before acting")
        if spf is not None and spf > MAX_SHORT_FLOAT:
            reasons.append(f"S9 squeeze_risk: short {spf*100:.1f}% of float > {MAX_SHORT_FLOAT*100:.0f}%")
        if dtc is not None and dtc > MAX_DAYS_TO_COVER:
            reasons.append(f"S9 squeeze_risk: {dtc} days to cover > {MAX_DAYS_TO_COVER}")
        ne = (s or {}).get("next_earnings")
        row["next_earnings"] = ne
        if ne:
            try:
                d = date.fromisoformat(str(ne)[:10])
                if 0 <= (d - TODAY).days <= EARNINGS_BLOCK_DAYS:
                    reasons.append(f"S10 earnings: {ne} within {EARNINGS_BLOCK_DAYS} days -- gap risk")
            except ValueError:
                flags.append(f"S10 earnings FLAG: unparseable date {ne}")
        else:
            flags.append("S10 earnings FLAG: next earnings date unknown")
        a = alt.get(tkr) or {}
        row["altsignal_info_only"] = {k: a.get(k, 0) for k in ("short_squeeze", "gov_contract_momentum", "news_momentum")}
        row["flags"] = flags + ["BORROW: locate/borrow fee not queryable via the IBKR connector -- verify in IBKR before acting"]
        if reasons:
            blocked.append({**row, "blocked_by": reasons})
            continue
        entry = round(close * (1 - LIMIT_BUFFER_PCT), 2)
        stop_pct = clamp(2 * atr / 100, 0.05, 0.12)
        raw_stop = entry * (1 + stop_pct)
        res = sorted(l for l in ((s or {}).get("swing_resistance") or []) if l > close)
        nearest_res = res[0] if res else None
        stop, ceiling_fired = raw_stop, False
        if nearest_res is not None and raw_stop < nearest_res:
            stop, ceiling_fired = nearest_res * 1.005, True
        qty = math.floor(SIZE_UNIT_USD / entry) if entry > 0 else 0
        row["reference_plan"] = {
            "NOT_AN_ORDER": "research only -- nothing drafted or submitted",
            "entry_ref_sell_short_limit": entry, "limit_buffer_pct": LIMIT_BUFFER_PCT,
            "buy_stop": round(stop, 2), "stop_pct_from_entry": round((stop / entry - 1) * 100, 2),
            "stop_basis": f"2xATR14 ({2*atr:.2f}%) clamped to {stop_pct*100:.2f}%"
                          + (f"; raised above nearest swing resistance ${nearest_res} (structural ceiling fired)"
                             if ceiling_fired else f"; raw stop already above nearest swing resistance ${nearest_res}"),
            "qty_whole_shares": qty, "notional_usd": round(qty * entry, 2),
            "max_loss_at_stop_usd": round(qty * (stop - entry), 2),
            "size_note": "whole shares only (IBKR does not short fractional shares)"
                         + ("; price above the $250 unit -- one share exceeds it" if qty == 0 else ""),
            "margin_note": "Reg T initial margin on a short is 150% of the position value; the account is already using margin.",
        }
        passed.append(row)

    out = {"script_version": SCRIPT_VERSION, "date": TODAY.isoformat(), "mode": "RESEARCH_ONLY",
           "n_candidates_gated": len(cand.get("candidates") or {}), "n_passed": len(passed),
           "passed": sorted(passed, key=lambda r: -((r.get("insider") or {}).get("short_conviction") or 0)),
           "blocked": blocked, "pre_dropped": cand.get("pre_dropped", []),
           "ordering_note": "Passing names are listed by short conviction for readability only. Per evidence_ledger.gate_ranking, gates define a universe but have not been shown to rank it.",
           "standing_caveats": [
               "No 10b5-1 flag: insider-sell scores are inflated by pre-scheduled plan sales by an unknown amount.",
               "Short-side signals have n=0 realised outcomes in this pipeline. Every gate threshold here is a judgement call.",
               "Losses on a short are unbounded; a gap through the stop is possible, especially around news.",
               "Borrow availability and fee are not checked by the pipeline."]}
    json.dump(out, open(os.path.join(WORKDIR, "short_research_today.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("passed", "blocked", "pre_dropped")}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("candidates", "gate"):
        print(__doc__)
        sys.exit(1)
    if a[0] == "candidates":
        cmd_candidates()
    else:
        tf = [a[i + 1] for i, x in enumerate(a) if x == "--tech"]
        sf = [a[i + 1] for i, x in enumerate(a) if x == "--supp"]
        cmd_gate(tf, sf)
