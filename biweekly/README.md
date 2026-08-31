# biweekly/

Reusable scripts for the biweekly portfolio review, hosted here for the same reason
`scripts/` exists for the daily briefing: so a review run can `curl` a byte-exact copy
instead of an LLM re-deriving (and re-paying the token cost for) the same logic every
two weeks.

| File | Purpose |
|---|---|
| `capture_ratio.py` | Insider-signal-to-fill capture ratio (review §6) — achievable vs captured return, pp lost to lag |
| `altsignal_refresh.py` | Rebuilds `altsignal.json`'s `short_squeeze` / `gov_contract_momentum` (bulk) and `news_momentum` (per-ticker) from live QuiverQuant REST endpoints |

## Why these two, first

Both were built from scratch, inline, during the 2026-08-31 biweekly review — the same
review that first discovered `beta/live/insiders` and `beta/live/wallstreetbets` /
`beta/live/govcontractsall` / `beta/live/quivernews?ticker=X` all work over plain REST
with the stored `quiver_api_key`, contradicting an earlier assumption that this data
needed the QuiverQuant MCP tool (which is not always connected in this session). Formalizing
the two calculations here means the next review doesn't have to rediscover or re-derive
either one.

## Adding to this directory

Same rule as `scripts/`: keep scoring/methodology logic in version-controlled, testable
code, not in a prompt. If a future review invents a new calculation worth repeating,
land it here rather than leaving it as one-off inline Python in that session's transcript.

## Fetching

```
curl -o capture_ratio.py https://raw.githubusercontent.com/maa2lparte/congressional-briefing-scripts/main/biweekly/capture_ratio.py
curl -o altsignal_refresh.py https://raw.githubusercontent.com/maa2lparte/congressional-briefing-scripts/main/biweekly/altsignal_refresh.py
```

No base64/tar step needed — these are small enough, and few enough, that a public raw
fetch straight to disk is simplest. See the parent `README.md` for the daily pipeline's
larger bundle, which followed the same reasoning.
