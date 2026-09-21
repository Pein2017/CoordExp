# Same-input PK accounting: bounded final readout

Status: complete for the authorized as-of snapshot. This is an estimate, not
an invoice, and it is not a global routing recommendation.

## Scope and integrity

- Root thread: `01a06f6d-2336-7670-a929-8c4d56ed54ba`.
- Receipt scope: `2026-09-09T02:37:13Z` inclusive through the fixed cutoff
  `2026-09-09T03:55:44Z` inclusive. The root `03:29:00Z` cap question is in
  scope.
- Exact staged logical records: 8 (root, 5 PK attempts, shared data/reducer,
  shared accounting helper).
- Scan: 8 files, 355 modern receipts, 40,866,386 measured tokens,
  `parse_errors=0`, invalid receipts `0`, missing identity `0`, foreign `0`,
  conflicting duplicates `0`, duplicate receipts `0`.
- Strict outcomes: 4 accepted, 1 failed, 3 intentionally unlabeled shared/root
  records. The failed contaminated Sol science attempt is retained and fully
  charged.

The root record is active rather than terminal because root remains active
through final response. Root cost is therefore only the measured as-of-cutoff
portion. Fourteen root receipts and nine accounting-helper receipts after the
cutoff were excluded and are explicitly unmeasured; no all-in claim is made.

## Per-record accounting

Costs use uncached input = input minus cached input, plus cached input and
output. Reasoning is included in output. Tokens are persisted receipt totals.

| logical record | model / effort | lifecycle | disposition | cached input | uncached input | output | estimated USD |
|---|---|---|---|---:|---:|---:|---:|
| `eng_alpha` | gpt-5.6-sol / xhigh | 02:45:56.346–03:28:05.969Z | accepted | 8,803,840 | 269,404 | 61,883 | 5.836812 |
| `eng_beta` | gpt-6-astra / low | 02:46:16.300–03:06:31.622Z | accepted | 2,547,584 | 97,055 | 22,975 | 4.666884 |
| `science_alpha_failed` | gpt-5.6-sol / max | 03:19:36.757–03:31:22.307Z | failed | 1,060,352 | 98,644 | 17,330 | 1.1653168 |
| `science_alpha_retry` | gpt-5.6-sol / max | 03:30:56.599–03:55:22.126Z; two task turns | accepted | 1,942,016 | 102,623 | 36,905 | 1.9253984 |
| `science_beta` | gpt-6-astra / medium | 03:19:59.697–03:25:03.299Z | accepted | 625,408 | 55,623 | 7,529 | 1.558088 |
| `row_cross_data` (shared) | gpt-6-astra / medium | two accepted tasks | unlabeled | 2,535,680 | 90,615 | 28,351 | 4.85938 |
| `row_cross_accounting` (shared) | gpt-5.6-luna / high | two persisted turns; post-cutoff work excluded | unlabeled | 5,920,000 | 242,244 | 39,538 | unknown |
| root current-study segment | gpt-6-astra / ultra | active as of 03:55:44Z | unlabeled | 15,807,744 | 369,242 | 83,801 | 23.690214 |

The retry's initial turn ended at `03:50:11.914Z`; its correction turn began at
`03:52:07.071Z` and ended at `03:55:22.126Z`. The failed original plus retry
therefore form one Sol scientific accepted chain costing `1.1653168 +
1.9253984 = 3.0907152 USD`, divided by one accepted replacement task:
`3.0907152 USD/accepted task`. Engineering has two accepted candidate rows:
`10.503696 USD` total, `5.251848 USD` per accepted row. The five PK attempts
sum to `15.1524992 USD`; the four accepted PK attempts sum to `13.9871824 USD`.

## Cost totals and caveats

- Known priced total, including root and shared data: **$43.7020932 USD**.
- Strict accepted-attempt cost: **$13.9871824 / 4 = $3.4967956** per accepted
  attempt. This excludes the failed attempt from the accepted denominator but
  retains its cost in the known total and Sol scientific chain.
- `gpt-5.6-luna/high` is unpriced: **6,201,782 tokens unknown cost**, not zero.
- Shared data/reducer and root costs are reported separately; neither is
  attributed to the Sol/Astra pair.
- The rate snapshot is the user-supplied estimate basis (Astra `$10/$1/$50`,
  Sol `$4/$0.40/$20` per million), SHA-256
  `efc2b936f8e94953c9d6e213aeaf6efc0a7a6b971cb45cd7a8db16e02348a324`.
  Cache-write usage was zero in all staged records; a cache-write rate was not
  silently imported.
- Lifecycle wall time includes orchestration and waiting; it is not model
  compute time. No root acceptance-time precision beyond the supplied cutoff
  is inferred. GPU bound (`2,309.2564 GPU-s`) is separate from API token cost.

## Exact artifacts

- Staging manifest and sanitized receipt segments:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/benchmark/accounting/staging-final-20260909T035544Z/`
- Final detailed ledger:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/benchmark/accounting/final-ledger.json`
- Final compact summary:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/benchmark/accounting/final-summary.json`
- Strict outcomes:
  `/data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-source-rweak-row-cross/benchmark/outcomes-final.jsonl`
- Price snapshot:
  `/data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-source-rweak-row-cross/benchmark/accounting/pricing-user-20260909.toml`
