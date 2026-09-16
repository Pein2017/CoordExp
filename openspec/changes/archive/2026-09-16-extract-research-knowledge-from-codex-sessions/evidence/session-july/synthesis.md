# July 2026 research-influential session synthesis

This is a source-bounded synthesis of the six covered sessions plus the
manually promoted sensitivity candidates. Raw JSONL remains authoritative;
memory and Notion owner IDs are routing aids only.

### Frozen painted-GT baselines expose two different model symptoms
- Sources: `/data/CoordExp/.codex/sessions/2026/07/05/rollout-2026-07-05T04-14-00-019f307b-b066-7e11-a52e-4456fbcfaa93.jsonl` (session `019f307b-b066-7e11-a52e-4456fbcfaa93`)
- Current owner: painted-GT research unit / tiny-training gate; exact current owner is not established in this package.
- Question and contrast: whether frozen paint-all and stepwise teacher-prefix baselines are interpretable enough to authorize tiny training.
- Observed evidence surface: named frozen artifacts, materialized manifests, scored JSONL, debug F1, parser/drop counters, and stop reasons.
- Scientific disposition: valid as a paired frozen baseline; paint-all has dense-slice under-recall, while stepwise has overgeneration, length truncation, and late-prefix empty stops.
- Technical/infrastructure disposition: artifact/evaluation checks were internally coherent; no inference or training rerun was performed.
- Decision or continuation impact: tiny tasks 7.3 and 7.4 may proceed with the same slice, decode, parser, matching, and denominator contract; larger training remains gated.
- Not claimed: not a larger-run result, not proof of an evaluation bug, and not evidence that stepwise F1 is comparable to paint-all F1.
- Notion coverage: needs-summary — training and research infrastructure owner `3c79d9ce-3f59-8186-838b-c775b98c787c`; exact page evidence was not checked.

### Qwen dense-enumeration behavior is an objective/decode mismatch, not one scalar metric failure
- Sources: `/data/CoordExp/.codex/sessions/2026/07/13/rollout-2026-07-13T09-18-59-019f5ac5-c940-7463-af2c-3b3ed40f0555.jsonl` (session `019f5ac5-c940-7463-af2c-3b3ed40f0555`); related baseline evidence in `/data/CoordExp/.codex/sessions/2026/07/05/rollout-2026-07-05T04-14-00-019f307b-b066-7e11-a52e-4456fbcfaa93.jsonl`.
- Current owner: `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/` unit/index/compass route.
- Question and contrast: whether dense-slice stepwise teacher prefixes make the frozen model emit the current target row and stop, versus following its full-image detection prior.
- Observed evidence surface: forward/attestation audit, schedule and prompt-continuation checks, raw/scored row behavior, truncation and empty-stop counters.
- Scientific disposition: the observed surface supports a narrow behavior diagnosis: free generation continues with multiple rows and can truncate; long teacher prefixes can produce immediate empty stops.
- Technical/infrastructure disposition: sampled metric-bearing execution was blocked by three P1 provenance/attestation gaps, including incomplete effective generation configuration and non-canonical score hashing.
- Decision or continuation impact: fix the attestation and runtime gates before interpreting sampled exposure as scientific evidence; preserve target-first, target-any, stop, and malformed counters.
- Not claimed: no natural-model causal mechanism, no trainability conclusion, no CUDA/B=3 versus B=4 parity claim, and no promotion from CPU contract to runtime validity.
- Notion coverage: needs-summary — causal evaluation/evidence owner `3c79d9ce-3f59-8173-ba20-c5cade809d97`.

### Exact-history HF backend is a narrow infrastructure seam, not a research framework
- Sources: `/data/CoordExp/.codex/sessions/2026/07/25/rollout-2026-07-25T16-54-55-019f9a33-85e1-7223-870b-051e78fa523c.jsonl` (session `019f9a33-85e1-7223-870b-051e78fa523c`).
- Current owner: `/data/CoordExp/.worktrees/research-probe-infras`, specifically `src/inference/hf_backend.py` and its two research consumers.
- Question and contrast: whether exact-history materialization, Qwen M-RoPE, teacher-forced rank evidence, and lifecycle observations need a reusable seam.
- Observed evidence surface: source audit, 19/19 seam/research tests, 12/12 existing HF-session tests, explicit smoke receipts, and OpenSpec validation.
- Scientific disposition: no new scientific result; the seam preserves caller-owned cohort, prefix, candidate, intervention, matching, estimand, claim, and stop semantics.
- Technical/infrastructure disposition: accepted with follow-ups; four parity receipts exist but are not durably indexed by the change authority, and stale legacy tests remain a separate hygiene issue.
- Decision or continuation impact: keep the seam narrow, add a verification note before archive, and do not build a generic runner/DAG/receipt framework.
- Not claimed: no proof of current GPU trainability, no claim that all historical scripts should migrate, and no archive authorization.
- Notion coverage: needs-summary — training and research infrastructure owner `3c79d9ce-3f59-8186-838b-c775b98c787c`.

### K=8 presentation-cache preparation completed but makes full materialization the launch bottleneck
- Sources: `/data/CoordExp/.codex/sessions/2026/07/28/rollout-2026-07-28T07-45-55-019fa7af-f7fd-7b71-974e-5ec73834bc69.jsonl` (session `019fa7af-f7fd-7b71-974e-5ec73834bc69`).
- Current owner: `/data/CoordExp/.worktrees/permutation-bundle-coordinate-noise-pilot/openspec/changes/permutation-bundle-coordinate-noise-pilot/` and its cache/admission owner.
- Question and contrast: whether the current K=8 same-image presentation cache is semantically reusable and whether preparation is compatible with first-GPU evidence.
- Observed evidence surface: `COORDEXP_CACHE_EXIT=0`, strict semantic receipt, cache fingerprint/manifest digest, phase timing, RSS, SQLite scratch, and admission requirements.
- Scientific disposition: mechanics-only; cache completion and strict admission candidacy do not establish a model-quality contrast.
- Technical/infrastructure disposition: preparation took about 4h53m and reached about 338 GiB RSS; workers parallelize encoding, not the single-process planning/identity/validation critical path.
- Decision or continuation impact: retain strict typed admission, run a two-step/eight-rank current-code smoke, and prefer bounded online generation or bounded streaming when first-GPU latency matters.
- Not claimed: no cache freshness under changed code, no power-loss durability proof, no production-scale throughput claim, and no scientific result from cache existence.
- Notion coverage: needs-summary — training and research infrastructure owner `3c79d9ce-3f59-8186-838b-c775b98c787c`.

### Online permutation result separates a real diagnostic contrast from a benchmark claim
- Sources: `/data/CoordExp/.codex/sessions/2026/07/29/rollout-2026-07-29T11-24-43-019fad9e-a47e-71b0-a555-7750fd99e7c4.jsonl` (session `019fad9e-a47e-71b0-a555-7750fd99e7c4`).
- Current owner: permutation-bundle-coordinate-noise-pilot research unit/OpenSpec; exact result/index owner requires current-worktree verification.
- Question and contrast: random source versus same-image permutation versus sorted/global-shuffle presentation, under greedy and K16 sampled-union diagnostics.
- Observed evidence surface: completed step-4887 run, val200 metrics, 12-image K16 union, replay/coverage checks, and zero parser/scoring/drop failures.
- Scientific disposition: bounded diagnostic evidence: global-shuffle beats random on val200 but remains below same-image permutation; sorted has the strongest sampled recoverability in the 12-image diagnostic.
- Technical/infrastructure disposition: run completed and artifacts are readable; the sampled panel is not a benchmark and has limited coverage.
- Decision or continuation impact: preserve the contrast and provenance, improve bounded worker layout separately, and do not promote the 12-image panel into a deployment or generalization claim.
- Not claimed: no causal explanation of why ordering helps, no complete-population claim, no production efficiency conclusion, and no architecture promotion.
- Notion coverage: needs-summary — serialization/geometry/ordering owner `3c79d9ce-3f59-81c0-8a6c-dbac1a9f2254`.

### Codex and Claude memory stores should remain separate provenance caches
- Sources: `/data/CoordExp/.codex/sessions/2026/07/30/rollout-2026-07-30T09-51-56-019fb270-0e93-7223-a1f5-c94aaa2d8385.jsonl` (session `019fb270-0e93-7223-a1f5-c94aaa2d8385`).
- Current owner: formal project rules and research documents under `/data/CoordExp`; memories are non-authoritative navigation caches.
- Question and contrast: whether Codex and Claude auto-memory can safely share one writable store.
- Observed evidence surface: current memory paths, loading behavior, project `CLAUDE.md`/`AGENTS.md` linkage, and the proposed event-triggered pointer-first sync boundary.
- Scientific disposition: no scientific result; this is provenance and authority infrastructure.
- Technical/infrastructure disposition: separate writers and formats should remain separate; formal knowledge stays in AGENTS, research units, specs, source, and receipts.
- Decision or continuation impact: use registry-first navigation and explicit incremental sync with conflict reporting; do not treat memory text as launch or claim authority.
- Not claimed: no guarantee of current memory freshness, no claim that an unverified Notion page is covered, and no authorization to mutate either memory store in this package.
- Notion coverage: needs-summary — global router `3c79d9ce-3f59-8148-96e7-c63c2a8e68ad`.

## Gaps and stop boundary

The July corpus contains many root/child and workflow sessions whose assistant
tails repeat the same engineering review. They remain source rows, but no
additional unique research finding was promoted without a distinct contrast or
artifact identity. Six L2 routes were not available in this session, and no
claim depends on an L2 report. Current owners, Notion page contents, and any
post-July supersession require a fresh owner-side check before promotion.
