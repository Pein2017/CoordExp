# Agent routing-cost audit v1

Status: **lead-inspected, scope-limited accounting snapshot**; not an SFT-only
cost, acceptance-cost estimate, or model ranking. The lead checked source paths,
scan/pricing coverage and all four root records. This snapshot excludes later
closeout activity; it is not a final bill.

## Scope

- Root thread: `01a06f6d-2336-7670-a929-8c4d56ed54ba`; `--include-root` enabled.
- Wrapper invocation used the complete root-thread subtree; it has no hour/minute timestamp filter, so this includes the whole September 5 task, including earlier N2/Human13 work. Existing persisted records span `2026-09-05T02:36:58.558Z` through `2026-09-05T21:53:41.907Z` (all 42 records are dated September 5). SFT-only attribution is therefore unavailable.
- Generated: `2026-09-05T21:56:42.480749+00:00`; policy: `followup_aware` (`completed rollout with no observed followup_task proxy`).

## Coverage

- Files seen: 7228; scoped records: 42; emitted: 42; parse errors: 0.
- Measured tokens: 143,093,448; fully priced records: 36/42; unpriced segments: 6; unpriced tokens: 74,891,277.

## Route × effort (all scoped records)

| Model | Effort | n | input | cached | output | billable uncached | billable cached | billable output | est. USD | priced n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-luna | max | 1 | 21,614,652 | 21,290,496 | 36,561 | 324,156 | 21,290,496 | 36,561 | 0.534514 | 1 |
| gpt-5.6-luna | medium | 29 | 30,325,035 | 28,379,136 | 143,352 | 1,945,899 | 28,379,136 | 143,352 | 1.128785 | 29 |
| gpt-5.6-sol | high | 1 | 2,456,432 | 2,244,480 | 17,148 | 211,952 | 2,244,480 | 17,148 | 2.696440 | 1 |
| gpt-5.6-sol | medium | 5 | 13,532,039 | 13,020,032 | 76,952 | 512,007 | 13,020,032 | 76,952 | 11.378611 | 5 |
| gpt-6-astra | medium | 2 | 2,324,539 | 2,192,256 | 20,755 | — | — | — | — | 0 |
| gpt-6-astra | ultra | 2 | 33,521,434 | 32,705,536 | 162,986 | — | — | — | — | 0 |
| gpt-6-astra | ultra,xhigh | 1 | 38,676,623 | 36,089,728 | 184,940 | — | — | — | — | 0 |
| unknown-model | unknown-effort | 1 | 0 | 0 | 0 | — | — | — | — | 0 |

Notes: `input/cached/output` are measured persisted deltas; cached input is a
subset of input, not an extra token quantity. Repeated contexts across calls
are counted repeatedly, so these are not unique-text counts. Billable columns
are present only for fully priced segments. The local snapshot is an estimate,
effective 2026-08-06, SHA-256
`cb3c1e5da544b43d76f72af812c009ce1742598267191118a7646e185de4f74b`;
it has rates for Luna/Sol/Terra, not Astra.

## Lead accounting

- The included root thread has **4 persisted records/segments**, totaling 72,198,057 measured input, 68,795,264 cached input, 347,926 output, 72,545,983 total tokens (one zero-usage settings record included); Astra has no local rate, so lead estimated cost and lead cost share are **unavailable**, not zero.
- Child priced cost shown by the ledger: $15.73835024 across 36 priced records; this is not an acceptance cost because dispositions are `followup_aware` proxies, not explicit lead acceptance. `cost_per_accepted_task` is therefore a proxy only.
- Qualitative routing evidence is confounded by task class and surface (metadata correction, manifest recovery, eval import repair, and pre-existing source gates); no global or route-default ranking is supported.

## Reproduction and artifacts

```bash
CODEX_HOME=/data/CoordExp/.codex conda run -n ms python /data/CoordExp/.codex/skills/codex-usage-ledger/scripts/run_ledger.py \
  --root-thread-id 01a06f6d-2336-7670-a929-8c4d56ed54ba --include-root \
  --prices /data/CoordExp/codex-usage-ledger/prices-gpt56-standard.toml \
  --disposition-policy followup_aware --format json --pretty \
  --output research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/agent-routing-audit-v1-attempts.json \
  --summary-out research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/agent-routing-audit-v1-summary.json
```

- Detail was moved unchanged out of the Git record surface after the scan:
  [attempts](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/agent-routing-audit-v1/attempts.json).
- Summary: [summary](agent-routing-audit-v1-summary.json).

## Practical pilot lessons, not a model league table

- Luna/max delivered the bounded full-DoRA freeze change; a lead correction was
  still needed to distinguish optimizer exclusion from frozen-tensor provenance.
- Luna/medium data preparation needed an artifact-integrity recovery by
  Sol/medium. The corrected brief/verifier and different task difficulty prevent
  attributing the difference to model family alone.
- Sol/medium's evaluation package passed function-level checks but missed the
  real direct-script import environment. A narrow Luna/medium repair plus a
  subprocess entry check closed it without repeating any model execution.
- Two qualification attempts exposed missing retained source-gate assets.
  This was a launch-preflight decomposition gap, not model-quality evidence;
  the executors correctly stopped instead of bypassing the gates.
- Use these observations to improve bounded briefs and real-entry checks, not
  to add a persistent multi-layer scheduler. Overall lead-cost reduction versus
  an alternative orchestration strategy has not been measured here.

Stop rule met: one wrapper scan, compact inspection, no custom parser or additional scan.
