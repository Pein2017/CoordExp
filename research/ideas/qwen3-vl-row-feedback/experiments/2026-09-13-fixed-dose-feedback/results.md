# Fixed-dose row-feedback pilot results

Status: annotation endpoint and bounded physical review lead-accepted; exposed content diagnostic pending.

## Question and scope

At equal admitted exposure, does a row-end continuous feedback slot (F) improve natural enumeration relative to an ordinary contextual slot (S)? Both arms independently start from the original N16 adapter, train for 64 updates on the same 16 admitted successor packages across 11 images, and use the same 54-record native-teacher protection. Each package receives eight exposures to both entry-c and post-c successor-w CE. The selection, dose, architecture, losses and 3,084-visible-token allowance were fixed before trained outputs.

S inserts the existing box-end embedding. F additionally inserts the just-completed row final hidden state, RMS-matched at fixed scale 1. Both retain native causal history and all earlier slots. This experiment concerns an additional continuous route; it cannot show that native KV lacks memory, isolate storage from readout or additional computation, or establish a sample-efficiency curve. The training histories originate from an older acquisition adapter, not the N16 anchor on policy.

## Frozen natural endpoint

The independent 32-image panel contains 280 annotations. Root inspected all seven successful outer/inner shard receipts, exact coverage, saved-adapter identities and paired input identities, then replayed the frozen scoring consumer. The replay was exactly equal to the original result (SHA256 `ba9b23fac2018c5811370befc25aad95ef6db53e3e1ccee7d0bbe7d91957f255`).

| Annotation matching threshold | S owners | F owners | F minus S | Gained / lost / retained |
|---|---:|---:|---:|---|
| IoU 0.50 | 193 | 197 | +4 | 9 / 5 / 188 |
| IoU 0.60 | 177 | 180 | +3 | 8 / 5 / 172 |
| IoU 0.80 | 129 | 125 | -4 | 4 / 8 / 121 |

The primary annotation-relative result is +4/280 owners (+1.43 percentage points), with 9 gains and 5 losses. The sign reverses at IoU 0.80, so the result is sensitive to localization precision. This single-seed, fixed-dose observation is a small coverage difference; it does not establish a robust physical-owner or memory advantage.

| Diagnostic | S | F |
|---|---:|---:|
| Parsed predictions | 295 | 294 |
| Invalid predictions | 0 | 0 |
| Parser drops | 0 | 0 |
| Strict repeats (any-class IoU > .95) | 1 | 1 |
| Visible generated tokens | 2832 | 2821 |
| EOS / cap | 32 / 0 | 32 / 0 |

All outputs terminate naturally; neither arm uses the length cap. Annotation-unmatched detections remain unresolved rather than automatic hallucinations. Strict-repeat counts do not resolve lower-IoU same-owner aliases.

## Preselected physical review

One source-blind pass covered all eight images selected before outputs: 306 proposals in 230 exact-class/geometry display groups. Two Sol reviewers owned disjoint four-image sets and saw only original images, proposal crops and the source-blind queue. Root verified exact coverage, inspected decision-bearing crops, and made six documented invalid-to-uncertain corrections before unblinding. These corrections changed no admitted owner assignment. The original reviewer files are preserved. The accepted aggregate uses the original sealed source-map manifest.

| Dense8 reviewed proposal measure | S | F |
|---|---:|---:|
| Clearly attributable physical owners | 110 | 113 |
| Additional proposals for the same owner | 3 | 3 |
| Plausible proposals | 113 | 116 |
| Uncertain proposals | 36 | 32 |
| Invalid proposals | 6 | 3 |

The admitted owner sets retain 107, gain 6 and lose 3. This is a +3 difference among clearly attributable proposals, with no reduction in same-owner recurrence (3 versus 3). Uncertain proposals remain neutral. A gained/lost admission does not prove that the opposite arm never mentioned the physical object: changed box extent or class ambiguity can alter admission. This is a proposed-owner review, not an exhaustive scene census or recall estimate.

The annotation and physical readouts can disagree per image. On images 125936 and 370711, annotation owners increase by two each while the reviewed physical-owner count is unchanged. Image 579655 loses two annotation matches but gains two clearly attributable reviewed owners. These examples reinforce that localization, class and proposal ambiguity can drive the aggregate difference; they do not identify a memory mechanism.

Root acceptance: `evaluation/natural/physical-review-root-acceptance-v1.json` (SHA256 `aa5c2ba97d9988403aa0ec3e723c16e9facd5a73392d19e1c4469991fce35e73`).

## Exposed content diagnostic

The three registered cup/spoon/donut cases use the trained F adapter, correct feedback, exact-self replay, and one-site replacement by a norm-matched physically different same-image owner source. That donor is captured after h+c+w and transplanted to the earlier C boundary: it is a future-completion diagnostic, not a deployable causal state-writing rule. Exact-self replay must match exactly before wrong-owner divergence is interpreted.

The first attempt failed before any replay/generation because the content consumer used the source256 materializer rather than the bank-bound train/dev union. Root accepted a consumer-only repair after reproducing the source miss and passing seven targeted tests. Shared runtime, packet, cases, histories and intervention remain unchanged. One explicitly authorized retry is running inside the original 3.75-GPU-hour content cap. The failed attempt is technical-invalid and carries no scientific outcome.

## Cost and acceptance boundary

Formal fit conservative paired allocation: 11.731481 GPU-hours. Frozen natural endpoint: 0.227980 GPU-hours. The through-endpoint receipt records at least 12.629817 GPU-hours across all disjoint stages, including the failed content attempt. Six early outer timing intervals are absent, so this is a known lower bound, not an exact total or verified remaining balance. The declared stage envelope is 40 GPU-hours inside the 48-GPU-hour round ceiling; planning ceilings are separate from measured spend. Retry cost and final scientific disposition remain pending.

## Evidence

- Protocol and scope: [unit.md](unit.md).
- Full artifact root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-row-feedback-pilot`.
- Fit acceptance: `training/fit-acceptance-v1.json`.
- Endpoint root acceptance: `evaluation/natural/endpoint-root-acceptance-v1.json`.
- Full paired result and literal outputs: `evaluation/natural/paired-result-v1.json` and its hash-bound inputs.
- Original blind queue and sealed manifest: `evaluation/natural/blind-review-v1/`.
- Accepted physical review and aggregate: `evaluation/natural/blind-review-lead-accepted-v1.json`, `evaluation/natural/physical-review-aggregate-v1.json`.
- Content repair and retry authorization: `content/run-v1-consumer-materialization-repair-v1.json`, `content/retry-authorization-v1.json`.
- Corrected cost accounting: `accounting/cost-through-endpoint-v2.json`.
