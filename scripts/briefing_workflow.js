export const meta = {
  name: 'congressional-briefing-parallel',
  description: 'Daily briefing: parallel data gather, chunked Stage 2 write-ups, one facilitator, independent verifiers, drafts only',
  whenToUse: 'DRAFT for policy v17 RUN_MECHANICS. Not yet adopted -- run manually as a pilot alongside the normal scheduled run before replacing it.',
  phases: [
    { title: 'Gather', detail: 'one agent runs parallel_runner.py + existing scoring scripts (no LLM fan-out for data)' },
    { title: 'Synthesize', detail: 'Stage 2 prose blocks, tickers split into <=4 chunks' },
    { title: 'Facilitate', detail: 'single owner of gates, ranking, rebalance, proposed drafts (no IBKR writes)' },
    { title: 'Verify', detail: 'independent long-side and short-side verifiers try to refute each proposed draft' },
    { title: 'Commit', detail: 'create verified drafts, write journal + trades + report + email' },
  ],
}

// ---------------------------------------------------------------------------
// Design notes (Mollick "Twilight Factory" mapping) -- see policy v17 RUN_MECHANICS
//  * Deterministic fan-out (HTTP, yfinance, ARIMA/GARCH) happens INSIDE one agent
//    via parallel_runner.py -- processes and threads, not subagents. Subagents
//    are only used where judgment or independent checking adds something.
//  * Exactly ONE facilitator owns "does this still add up". Chunk writers never
//    decide gates or trades; they describe.
//  * Verifiers are independent: they re-derive from raw files + policy and are
//    told to REFUTE. A draft survives only if its verifier confirms it.
//  * Approval stays human: the pipeline creates SAVED DRAFTS ONLY. Nothing here
//    submits, modifies or cancels a live order, and nothing edits policy files.
//  * Agent count: 1 + <=4 + 1 + 2 + 1 = <=9.
// args: { today: 'YYYY-MM-DD', workdir: '/tmp/cb', synth_chunks: 4 }
// ---------------------------------------------------------------------------

const TODAY = (args && args.today) || 'UNKNOWN_DATE'
const WD = (args && args.workdir) || '/tmp/cb'
const CHUNKS = Math.min(4, Math.max(1, (args && args.synth_chunks) || 4))

const COMMON = `Date: ${TODAY}. Working directory: ${WD} (set CB_WORKDIR=${WD}, YF_DISABLE_CURL_CFFI=1).
Authoritative rules: the LATEST policy-*.json in Google Drive folder 1ZsIsLCwLSQAiFXakw4L5LqZSTL3TPb2b
(precedence policy > trades > SKILL.md). Never submit, modify or cancel a LIVE IBKR order. Never write or
edit a policy file. If a required input is missing, say so in your output rather than guessing.`

const GATHER_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    policy_version: { type: 'string' },
    tickers_long: { type: 'array', items: { type: 'string' } },
    tickers_short: { type: 'array', items: { type: 'string' } },
    files: { type: 'object' },
    manifest_summary: { type: 'string' },
    validation_problems: { type: 'array', items: { type: 'string' } },
    validation_warnings: { type: 'array', items: { type: 'string' } },
    degraded_notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['ok', 'policy_version', 'tickers_long', 'tickers_short', 'files', 'validation_problems'],
}

const PROPOSAL_SCHEMA = {
  type: 'object',
  properties: {
    account_state_summary: { type: 'string' },
    stale_draft_actions: { type: 'array', items: { type: 'object' } },
    proposed_drafts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          side: { type: 'string', enum: ['BUY', 'SELL'] },
          book: { type: 'string', enum: ['long', 'short'] },
          ticker: { type: 'string' },
          trigger: { type: 'string' },
          quantity: { type: 'number' },
          limit_price: { type: 'number' },
          notional_usd: { type: 'number' },
          gate_evidence: { type: 'object' },
        },
        required: ['id', 'side', 'book', 'ticker', 'trigger', 'quantity', 'limit_price', 'notional_usd', 'gate_evidence'],
      },
    },
    blocked: { type: 'array', items: { type: 'object' } },
    human_attention: { type: 'array', items: { type: 'string' } },
  },
  required: ['proposed_drafts', 'blocked', 'human_attention'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          confirmed: { type: 'boolean' },
          reasons: { type: 'array', items: { type: 'string' } },
        },
        required: ['id', 'confirmed', 'reasons'],
      },
    },
    goal_drift_notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdicts'],
}

// ---------------------------------------------------------------- Gather
phase('Gather')
const g = await agent(`${COMMON}

You are the data runner. Do NOT make trading judgments.
1. Read the latest policy-*.json from Drive and note its _policy_version.
2. Fetch the pipeline scripts from windows.script_fetch_source_v10 (GitHub raw, curl to disk) and check
   byte sizes against known_good_byte_sizes. parallel_runner.py and supplement_batch.py must be present;
   if either is missing, stop and return ok=false with the reason (the normal scheduled task is the fallback).
3. Read quiver_api_key from Drive config.json into ${WD}/config.json.
4. python3 scripts/parallel_runner.py fetch   (bulk congress, insiders, wsb, govcontracts concurrently).
   A 403 on raw_insiders.json is expected: fetch insiders via the QuiverQuant MCP get_insider_trading tool
   only if policy says so; otherwise follow policy windows.insider_source.
5. Run the existing scoring scripts in the order policy/SKILL.md defines (stage1a_v2.py, insiders.py,
   insiders_short.py, short_screen.py candidates), and build the day's combined long+short ticker list.
6. python3 scripts/parallel_runner.py enrich --tickers-file <that list>   (technicals chunked across CPUs,
   supplement_batch.py once, dark pool per policy -- add --no-darkpool if policy keeps dark pool on MCP,
   then collect it with get_dark_pool as journal-spec directs).
7. Run score_ml.py and short_screen.py gate --tech v2_today.json --supp supplement_today.json.
8. Return file paths and the validation block from run_manifest.json. Copy warnings verbatim.`,
  { label: 'gather', phase: 'Gather', schema: GATHER_SCHEMA })

if (!g || !g.ok || (g.validation_problems || []).length) {
  log(`Gather failed or validation problems: ${JSON.stringify(g && g.validation_problems)}`)
  return { status: 'ABORTED_AT_GATHER', gather: g }
}
log(`policy ${g.policy_version}; ${g.tickers_long.length} long / ${g.tickers_short.length} short tickers`)

// ---------------------------------------------------------------- Synthesize
// Chunked, not per-ticker: per-ticker agents would each re-read the policy and
// multiply token cost ~Nx for prose a single agent writes consistently.
const all = [...new Set([...g.tickers_long, ...g.tickers_short])].sort()
const chunks = Array.from({ length: CHUNKS }, (_, i) => all.filter((_, j) => j % CHUNKS === i)).filter(c => c.length)

const blocks = await pipeline(
  chunks,
  (chunk, _orig, i) => agent(`${COMMON}

Write the Stage 2 per-ticker blocks for ONLY these tickers: ${chunk.join(', ')}.
Inputs: ${JSON.stringify(g.files)}. Use the block format the skill and policy require (including the AltSignal
line with '0/100 (no data)' where absent, and informational-only items labelled as such).
Describe; do NOT decide gates, rankings or trades -- the facilitator does that.
Write your blocks to ${WD}/stage2_blocks_part${i}.md and return that path plus any data gaps you saw.`,
    { label: `synth:${i}`, phase: 'Synthesize', effort: 'low' }),
)

// ---------------------------------------------------------------- Facilitate
phase('Facilitate')
const proposal = await agent(`${COMMON}

You are the FACILITATOR -- the single owner of "does this all still add up". Inputs: ${JSON.stringify(g.files)};
chunk writers' notes: ${JSON.stringify(blocks)}; runner warnings: ${JSON.stringify(g.validation_warnings || [])}.
Read-only IBKR calls are allowed (positions, orders, order instructions, trades DAYS_7, price snapshots).
Do NOT call create_order_instruction or delete_order_instruction.
Apply the policy end to end: draft_staleness check, the four mechanical sell triggers, DAILY_REBALANCE_TO_CAP,
gates 1-8 for buys/top-ups, STAGE 3S gates S1-S11 and SHORT_EXITS. For every proposed draft, include the exact
numbers that justify each gate in gate_evidence. Put anything a human should look at in human_attention
(data staleness, FLAG gates, conflicts between chunk notes, anything surprising). Do not invent missing data.`,
  { label: 'facilitator', phase: 'Facilitate', schema: PROPOSAL_SCHEMA })

if (!proposal) return { status: 'ABORTED_AT_FACILITATE', gather: g }

// ---------------------------------------------------------------- Verify
// Barrier justified: Commit needs every verdict together. Two verifiers, split by
// book, so the long and short rulebooks are each checked by a fresh context.
const books = ['long', 'short']
const verdicts = await parallel(books.map(book => () => {
  const mine = proposal.proposed_drafts.filter(d => d.book === book)
  if (!mine.length) return Promise.resolve({ verdicts: [], goal_drift_notes: [] })
  return agent(`${COMMON}

You are an independent VERIFIER for the ${book} book. Try to REFUTE each proposed draft below.
Re-derive every gate from the raw files (${JSON.stringify(g.files)}) and the policy yourself -- do not trust the
facilitator's gate_evidence. Check sizing arithmetic (4dp floor / whole shares for shorts, buffered limits,
caps and minimums), stale-data warnings, and that no draft conflicts with a live order or open position.
Default to confirmed=false if any gate cannot be re-derived. Also note any sign the proposal has drifted from
the policy's intent (goal_drift_notes). Read-only; no IBKR writes.
Proposed drafts: ${JSON.stringify(mine)}`,
    { label: `verify:${book}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' })
}))

const vmap = {}
verdicts.filter(Boolean).forEach(v => v.verdicts.forEach(x => { vmap[x.id] = x }))
const confirmed = proposal.proposed_drafts.filter(d => vmap[d.id] && vmap[d.id].confirmed)
const rejected = proposal.proposed_drafts.filter(d => !(vmap[d.id] && vmap[d.id].confirmed))
  .map(d => ({ ...d, verifier: vmap[d.id] || { confirmed: false, reasons: ['no verdict returned'] } }))
log(`${confirmed.length} drafts confirmed, ${rejected.length} held back for human review`)

// ---------------------------------------------------------------- Commit
phase('Commit')
const report = await agent(`${COMMON}

Create SAVED DRAFTS (create_order_instruction) for exactly these verifier-confirmed drafts and no others:
${JSON.stringify(confirmed)}
Perform the stale-draft deletions/refreshes the facilitator listed ONLY where the ticker also appears above
(REFRESH) or the facilitator marked it EXPIRED: ${JSON.stringify(proposal.stale_draft_actions || [])}.
Then write, per policy and journal-spec: the decision journal, trades-*.json (with the sizing block mirrored
verbatim), the Drive report, and the summary email. Include a "Held back by verifier" section listing:
${JSON.stringify(rejected)}
and a "Needs your attention" section with: ${JSON.stringify([...(proposal.human_attention || []), ...verdicts.filter(Boolean).flatMap(v => v.goal_drift_notes || [])])}.
Include the Stage 2 blocks from ${WD}/stage2_blocks_part*.md and run_manifest.json timings. Return the report path.`,
  { label: 'commit', phase: 'Commit' })

return { status: 'OK', policy: g.policy_version, confirmed: confirmed.length, held_back: rejected.length, report }
