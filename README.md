# congressional-briefing-scripts

Pipeline scripts for the daily disclosure briefing scheduled task, hosted here so the
scheduled task can `curl` them directly (byte-exact, no base64 round-trip through an
LLM's context — see the scheduled task's own "OPERATIONAL EFFICIENCY FIXES" section
for why that matters).

| File | Purpose |
|---|---|
| `scripts/stage1a_v2.py` | Congressional disclosure scoring (filer-weighted, AltSignal-aware) |
| `scripts/insiders.py` | SEC Form 4 insider trade scoring (cluster-independence test) |
| `scripts/supplement.py` | SMA trend base, relative performance, swing levels, earnings/analyst data |
| `scripts/technicals_v2.py` | Extended technical indicator panel (trend/momentum/volatility/volume/stats) |
| `scripts/train_ml.py` | Trains the informational-only ML classifier |
| `scripts/score_ml.py` | Scores tickers against the trained model |

All six were fetched from Google Drive and verified byte-exact (`wc -c` + `ast.parse`)
on 2026-08-27/28 before being committed here.
# congressional-briefing-scripts
