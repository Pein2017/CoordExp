---
title: Static-Dynamic Owner Interface Crossover Results
description: Completed no-training static, dynamic, crossover, and gradient-path study on the admitted 13-image panel.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
unit_id: 2026-08-05-static-dynamic-owner-interface-crossover
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_with_bounded_hold
updated: 2026-08-06
---

# Static-Dynamic Owner Interface Crossover Results

## Verdict

At A3 step-2445 event `A/gt:2299:2`, query-to-image static routing is
behaviorally causal: the native and self/no-op arms release covered owner
`gt:2299:0`, while target-B-exclusive key spotlight `K10` releases the strict
target row for `gt:2299:2`. Muting the latest dynamic terminal/history state
does not change either the one-row or three-row outcome, and the fixed
static-by-dynamic interaction is zero. This supports H2 at one A checkpoint and
event, contradicts H1 and H5 there, and does not support H3 or H4 there.

S step-2444 has no causal P1--P4 verdict. Its only eligible event,
`S/gt:5001:15`, exhausted the one allowed repair before any actuator call. Its
sealed leaf is a technical HOLD with null measured counts, not a zero effect.
All H1--H5 explanations therefore remain unresolved for S.

The A observation cannot be attributed to `<|commit|>`, InfoNCE, A3 owner
binding, or any other single training difference. The checkpoints and native
wrappers are not a matched attribution pair. No training, architecture,
decoding policy, or production behavior is promoted.

## Execution accounting

- The derived stable x-then-y panel preserves all 392 manually calibrated
  owners; SHA-256 is
  `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`.
- H0 attempted and completed 26 native baselines: 13 for S and 13 for A.
- Each checkpoint materialized the frozen 32-owner cohort. Each produced one
  verified source-specific A/B pair and 31 `no_verified_B` entries.
- The hybrid denominator is 64 checkpoint-specific event entries: 62
  ineligible entries materialized CPU-only as `not_measured`, one A live event
  plus its P4 repair overlay, and one sealed S pre-actuator HOLD.
- The observational census contains eight images, four owners per image, 31
  layers, and 992 owner-layer rows per checkpoint.
- No efficacy threshold gated execution. Unmatched, invalid,
  `not_applicable`, and HOLD outcomes were preserved rather than converted to
  zeros or silently dropped.

## H0 substrate

| Checkpoint/slice | mAP | AP50 | AP75 | mRecall100 |
| --- | ---: | ---: | ---: | ---: |
| S all 13 | 0.414959 | 0.536872 | 0.447846 | 0.427613 |
| S legacy 12 | 0.416220 | 0.533509 | 0.452175 | 0.428088 |
| S image 2299 | 0.270291 | 0.529703 | 0.257891 | 0.299342 |
| A all 13 | 0.379397 | 0.514524 | 0.390484 | 0.402968 |
| A legacy 12 | 0.383172 | 0.518325 | 0.395144 | 0.406338 |
| A image 2299 | 0.139372 | 0.218544 | 0.169378 | 0.153618 |

S records 172 strict owner TPs and 220 FNs; A records 149 TPs and 243 FNs.
The split is S `143/203` on legacy-12 and `29/17` on image 2299, versus A
`133/213` and `16/30`. The subset metrics are fresh reductions through the
same evaluator, not independent sealed benchmark runs. The small selected
panel and substrate differences make raw S-versus-A gaps descriptive only.

## Observational carrier census

Both checkpoints expose coarse owner-related information across the captured
image field, but specificity is weak. Final-layer retrieval-at-one is `7/30`
for S and `10/29` for A; the mean owner-versus-hardest-negative margin is
negative at every captured layer. These 992-row-per-checkpoint readouts support
coarse accessibility and motivate the causal static test, but they do not by
themselves establish an owner-selective carrier.

## A eligible event

### P1: static image field

- `K00` and byte-identical `K01` both release `gt:2299:0`.
- Target-B-exclusive spotlight `K10` switches the strict complete row to
  `gt:2299:2`.
- Covered-A removal `K11` and equal-area background spotlight `K12` are
  unmatched; `K13` is explicitly `not_applicable` because no declared
  same-class competitor region exists.
- Residual self/no-op `R00` releases `gt:2299:0` at blocks 13, 23, and 27.
  `R10` and `R11` are unmatched at block 13 but return `gt:2299:0` at block
  23. Empty shared-core `R12` cells are `not_applicable`. Block-27 sentinels do
  not change behavior.

The qualified `K10` switch establishes a bounded static query-to-image routing
handle. The later residual field does not show the same clean owner-selective
transfer and should not be treated as an equivalent carrier.

### P2: dynamic history

`D00`, `D01`, `D10`, `D11`, and `D20` are behaviorally identical. At one row
they release `gt:2299:0` with net utility zero; at three rows they release
owners `0,3,4` with net utility `+2`. `D12` is invalid because no earlier
equal-length completed row exists, and `D21` is `not_applicable` because no
independently proven same-parent donor row exists. The tested dynamic carrier
is idle at this event.

### P3: fixed crossover

`Y00` and `Y01` are identical: one-row owner `0`, then owners `0,3,4` at the
three-row horizon with net `+2`. `Y10` and `Y11` are also identical: the first
row is unmatched and the three-row sequence is unmatched, owner `2`, owner `6`,
with net `+1`. Therefore `Delta_static=-1`, `Delta_dynamic=0`, and `tau=0` for
the frozen utility. The static factor is causal but harmful under this selected
covered-A-removal operator; the dynamic factor neither rescues nor compounds
it.

### P4: frozen gradient paths

The first P4 capture is retained as technical-invalid lineage. Its one allowed
fragmented-capture repair is valid, uses no optimizer, mutates no parameter,
and records finite gradients to image residual, matched background, latest row
span, and latest terminal carrier.

| Objective | Value | Target/control ratio |
| --- | ---: | ---: |
| target-B complete-row NLL | 19.010838 | 0.232533 |
| uncovered-B versus covered-A margin loss | 0.288910 | 0.039276 |
| fixed sum | 19.299747 | not defined |

The reported higher-is-better target-B margin is `-0.288910`. Background
gradient norms exceed target-region norms, and target-B and uncovered-B are the
same teacher-forced row; no independent uncovered-owner bank was available.
P4 therefore establishes graph reachability only, with weak owner specificity.
It is not evidence that an SGD update would improve rollout behavior.

## S eligible HOLD

Two attempts failed before model/actuator execution and the repair budget is
exhausted. The final administrative leaf seals 16 P1 cells, 14 P2 horizon
cells, eight P3 horizon cells, and three P4 objectives as technical invalid.
Actual model, actuator, and scientific counts remain null. The failed leaves
and repair lineage remain part of the evidence; no further rerun is admitted by
this unit.

## H1--H5 disposition

| Hypothesis | A event | S | Scope-aware decision |
| --- | --- | --- | --- |
| H1: no usable static carrier | contradicted | unresolved | `K10` yields the strict target-B row in A. |
| H2: static carrier with salience/routing failure | supported | unresolved | Supported for A query-to-image routing; later residual evidence is weaker. |
| H3: dynamic coverage failure | not supported | unresolved | Dynamic terminal/row muting is idle in A; no causal reinforcing, suppressive, or reallocating effect is observed. |
| H4: coupled bottleneck | not supported | unresolved | The fixed A crossover has `tau=0`. |
| H5: neither tested handle is causal | contradicted | unresolved | A's static key intervention is causal. |

The aggregate evidence bundle deliberately leaves machine-generated H1--H5
and recommendation fields null because its overall qualification is HOLD. This
table is the lead-owned scientific interpretation of qualified A contrasts and
the explicit absence of S actuator evidence.

## Recommendation

Reject dynamic commit/history carrier training as the next objective family on
this evidence. A later, separately authorized pilot should first test a bounded
static owner-localized/query-key routing or region-contrastive objective with
explicit matched background, covered-A, same-class, and non-target controls.
It must add an independent uncovered-owner specificity bank and a matched
attribution control before scale-up. Scalar density remains an inventory or
remaining-mass diagnostic, not an owner carrier; a combined static/dynamic
pilot is not justified by the zero interaction.

This is a recommendation for the next discriminator only. It is not
authorization to train, add tokens, change wrappers, alter decoding, promote an
architecture, or launch production work.

## Evidence

The accepted hybrid bundle and its receipts are materialized under:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/p1-p4/final-hybrid-v2/`

| Artifact | File SHA-256 | Embedded self SHA-256 |
| --- | --- | --- |
| `aggregate_summary.json` | `c33ccce86dbb9f09569fb05eae4fc6ba9f1e02f51d827b25699bee5f608faa88` | `7793b54b4615647c02ef9c95e1505005daaeae3acc9edfa0c2cb53439c1f7699` |
| `aggregate_receipt.json` | `a08f8a970a3030b3d135cd506f0fb55ae786eb2ca3f089f681cf523dbd4db798` | `d66b20b0f1a3fcd41d2443a79afe973bc04b778f51c82c7e8035b745d4980d1b` |
| `evidence_bundle.json` | `2126d35cf517e0bc0256dae32a534e7ce67816fb4efafb2edfaccb4576697ca9` | `85b3262d598eb64f8a3be411803cea2da9ea0fb5685d8c76497789457cf4c9e7` |
| `evidence_bundle_receipt.json` | `b674107babd9848a71b6b9c1236a2bb8040a14d28af2a8df423064b1731fd609` | `9dc7134e27e227d15a01017171df0a632b8192a9c3425ae04a875710fcfd26dd` |
| `raw_event_evidence.jsonl` | `326a50f5ed5989be2a961adee96bcab2e03fa83ffdf79900241586cc05cbd68a` | 64 records |
| `h0_baseline_evidence.jsonl` | `398099362e32008847e49d6c6908869b2fa9bf5015575dff0c583ca1bd31ac33` | 26 records |

The bundle is `completed` with overall qualification `hold`; its H1--H5,
interpretation, and recommendation fields remain null. A's raw event contains
standalone hashes for both the all-stage base and all six P4 repair-overlay
files. The earlier immutable `final-hybrid/` bundle is superseded only because
its raw-event row required a join through the aggregate summary to discover
the overlay paths; it remains intact as provenance and is not the accepted
result.

The final S HOLD leaf file SHA-256 is
`65a325ae4c400251be4d4d23d4053b7c5d4eea98aa397c91535767c94e5cb168`;
its receipt SHA-256 is
`4d07fa1139b820c939029f5b13c2122aba889528d9d6d8f41eb5fba427364931`.
The execution-contract `unit.md` and `tasks.md` remain byte-frozen at their
launch hashes
(`da7cd13cfe83990647a68fca45f7f00ecf65fbd87e61856dcf53e1b7fe701513`
and `ecb7eb1a8d20d6dbfd09f42638b2bbcabdd5ab262707c52a2ba8ae0f07d4128c`
respectively); this
result and the experiment index own lifecycle completion rather than rewriting
the contract bytes after execution.
