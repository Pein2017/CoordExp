# Fixed-dose row-feedback pilot results

Status: complete bounded pilot; paired fit, annotation endpoint, physical review and content diagnostic lead-accepted. No checkpoint promotion.

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

The three registered cup/spoon/donut cases use the trained F adapter, correct feedback, exact-self replay, and one-site replacement by a norm-matched physically different same-image owner source. W is captured after h+c+w and transplanted to the earlier C boundary: this is a future-completion diagnostic, not a deployable causal state-writing rule.

The first attempt failed before any replay/generation because the content consumer searched source256 rather than the bank-bound train/dev union. Root accepted the consumer-only repair after reproducing the source miss and passing seven targeted tests. The sole authorized retry completed all three cases and nine generations. Root verified case/source bindings, exact prepared-input equality with the F fit, finite distinct source vectors, the registered replacement site, and token-exact self replay in all three cases. Shared runtime, packet, histories and intervention are unchanged. Both attempts and their costs remain preserved.

| Exposed case | Correct F | Wrong-owner source | Bounded observation |
|---|---|---|---|
| Cup 210457 | 3,084 visible tokens; cap | 3,084; cap | First difference at zero-based token 790; one nonrecurring later box changes its lower coordinate. Both runs still degenerate. |
| Spoon 219546 | 11 tokens; one row; EOS | Exactly identical | No visible effect at this intervention; inconclusive. |
| Donut 417044 | 61 tokens; six rows; EOS | 41 tokens; four rows; EOS | W remains row 1. Three later geometries shift and two correct-only regions disappear; these include donut-hole groups. |

**W remains the first generated row in both correct and wrong-source continuations for all three cases.** Substituting W's vector for C's vector therefore does not produce selective suppression of W in these tests. The two visible changes demonstrate sensitivity to added vector content, without establishing an owner-specific visited-object ledger, storage-versus-readout explanation, native-memory incapacity or sample efficiency.

The cup's strict parser accepts 165 rows and drops 178 (177 geometry-invalid and one malformed) in each of the three generations. There are 65 distinct class/coordinate-bin rows and 100 additional exact serialized duplicates. Pixel rounding gives 64 distinct class/native-pixel boxes; neither duplicate count is the natural endpoint's any-class IoU > .95 metric. The sole changed accepted geometry lies over the right person's torso rather than either reviewed blue cup. This supplies no selective owner effect or escape from repetition.

Root inspected the donut image and the bound decision crops. The correct-only third row spans the lower-right donut-hole group; the sixth covers a small right-edge group region. A reduction from six to four emitted rows is therefore **not** evidence that two independently identified physical donuts were erased or omitted. The common W donut remains present, and the other shared changes are geometric. [Decision crops](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-row-feedback-pilot/content/comparison-v1/donut-417044-1079494-decision-crops.png) make this distinction reviewable.

Root content acceptance: `content/run-v2-root-acceptance-v1.json` (SHA256 `004e7bc386e4e781c5e311c8f2b436872578b842224984314b49db614844079f`). Complete literal continuations, source receipts and comparisons are retained under `content/run-v2/` and `content/comparison-v1/`.

## Cost and acceptance boundary

Formal fit conservative paired allocation: 11.731481 GPU-hours. Frozen natural endpoint: 0.227980 GPU-hours. Successful content retry: 0.228698 GPU-hours; both content attempts together: 0.232801 GPU-hours. Final accounting records a known allocation lower bound of 12.858515 GPU-hours across disjoint stages. Six early outer timing intervals are absent, so this is not an exact total or verified remaining balance. The declared stage envelope is 40 GPU-hours inside the 48-GPU-hour round ceiling; planning ceilings are separate from measured spend. The latest producer and wrapper have exited, and all model work in this round is settled.

Content resource counters need one explicit scope distinction: 10,760 model forwards, nine image forwards, 1,174 internal slots and 9,448 visible generated tokens are sums of the nine **generation** receipts. Three donor replays are separately counted, but their detailed forward/slot totals were not persisted. Whole-run outer time includes donor work. This counter limitation does not change the self-parity, intervention or visible-output evidence.

## Decision and stop rule

The finite round is closed. The primary endpoint has a small positive difference at IoU .50, an opposite sign at IoU .80, and no reduction in reviewed same-owner recurrence. The added continuous route is mechanically usable and can affect continuations, but the exposed substitutions do not establish selective physical-owner memory. These observations support retaining the implementation as a mechanism probe; they do not justify promoting it as a memory/recurrence solution or expanding training on this evidence alone.

One paired 64-update fit, the frozen 32-image endpoint, the preselected dense8 review and the fixed three-case diagnostic are complete. No dose, architecture, cohort, additional retry or checkpoint promotion is included. A future protocol would need to distinguish owner-specific content effects from general continuation/geometry changes under matched visible histories and computation; that is a new research question, not an authorized follow-on run.

Round closeout: `round-closeout-v1.json` (SHA256 `e65906704109e3f5387af4853c3fd63a8a292199ffa408a19c145ba865f85fc8`).

## Evidence

- Protocol and scope: [unit.md](unit.md).
- Full artifact root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-row-feedback-pilot`.
- Fit acceptance: `training/fit-acceptance-v1.json`.
- Endpoint root acceptance: `evaluation/natural/endpoint-root-acceptance-v1.json`.
- Full paired result and literal outputs: `evaluation/natural/paired-result-v1.json` and its hash-bound inputs.
- Original blind queue and sealed manifest: `evaluation/natural/blind-review-v1/`.
- Accepted physical review and aggregate: `evaluation/natural/blind-review-lead-accepted-v1.json`, `evaluation/natural/physical-review-aggregate-v1.json`.
- Content repair and retry authorization: `content/run-v1-consumer-materialization-repair-v1.json`, `content/retry-authorization-v1.json`.
- Final cost accounting: `accounting/final-cost-v1.json`, building on the preserved `accounting/cost-through-endpoint-v2.json`.
- Root content checks: `content/run-v2-root-mechanical-checks-v1.json`, `content/run-v2-root-parser-checks-v1.json`.
- Complete round acceptance: `round-closeout-v1.json`.
