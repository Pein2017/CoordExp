# Continuation: shared geometry loss and paired parameter starts

## Current authority and preparation status

The 2026-09-15 takeover request and subsequent COCO-80 scope ruling supersede
the predecessor's user pause. The user subsequently directed the lead to continue
implementation and coordination, preserving unrelated dirty changes. The versioned teacher, shared-loss integration and full paired batch are
lead-accepted. Both final endpoints achieve scoped218 FN0/F1=1; see
[the paired result](dual-start-results.md) for the preserved historical deficits. The prior accepted result remains historical
evidence. The [preserved protocol](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-training-set-completion-curriculum/unit.md)
retains its original population, conditions and receipts.

The [preparation receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-coco80-preparation-v1/preparation.json)
owns exact checkpoint, embedding and source hashes, the complete owner partition,
common recipe and work still required before training. It is an input inventory,
not a launch manifest or technical admission. Lifecycle stays with [state.json](state.json).

## Authorized comparison

With one compliant teacher, the same training recipe and low-weight shared
geometry loss, does initialization from the full-teacher step256 adapter or the
original geo_sorted_xy step2444 adapter yield better final natural-greedy FN/F1,
incumbent-owner retention and complete-output quality at matched new exposure?

- A starts from fourth-fit step256, the mainline for retaining existing gains.
- B starts from original geo_sorted_xy step2444, testing earlier-start learnability.
- Both use the bound step2444 special-token embedding delta and **fresh AdamW**.
  Do not use the old continuation entry to restore optimizer/RNG state.
- Keep the retained teacher rows' relative order identical across arms; no first-
  comparison rollout refresh or ordering ablation. Checkpoint age is not a winner rule.
- Preserve the existing language DoRA surface and lr1e-5 recipe. Geometry starts
  at weight0.01 and margin1/999 without changing its expected-coordinate semantics.

## Versioned scope ruling

User ruling: “采用有版本的 COCO-80 子集，保留完整历史账本”.

The receipt partitions the original232 owners into included, confirmed outside
COCO-80, and unresolved category/scope. Unresolved descriptions are not guessed
or promoted from their old masked teacher text. This first comparison does not
add later-known owners. Preserve both the frozen232 and current-known248 ledgers;
neither is replaced by the new training subset, and unresolved owners are not
declared completed. Build new token sequences and recompute all conditional
forwards after row removal; old suffix KV/logits/probabilities cannot be reused.

## Shared implementation and acceptance boundary

Use one shared src/losses implementation through configuration, LossRunner,
ordinary and streaming loss computation, normalization, metrics and the real
training entry. The experimental probe reuses that implementation. Bind boxes
through LossContext and CoordinateLossTarget by example/segment/object/slot;
do not assume consecutive coordinate positions or group by matching geometry.

Verify packed/compact and masked-position behavior, finite directional gradients,
complete/incomplete/no-box behavior, fail-closed mapping errors, default total-loss
and logging participation, and relevant distributed normalization. Run the
smallest real training entry check after CPU verification. Hinge zero does not
guarantee argmax-valid geometry or prevent malformed output.

## Next execution boundary

Before training, finalize and validate the compliant teacher and close the shared
loss real-entry check. The first batch's budget and metrics are fixed below.
Each training job is at most five hours; long GPU work uses named tmux and durable
receipts. Read back both starting adapters and matched new-update endpoints from
original images, empty assistant prefixes, greedy RP1 and cap3084. Report errors
and gained/lost alongside the primary metric, including separate historical ledgers.

No claim yet selects A or B. Unequal exposure, loading failures or broken target
mapping cannot be interpreted as evidence against an initialization. No training-
image expansion, refresh, ordering contrast or held-out veto belongs to this first
comparison. This bounded batch is now closed; see the paired result for the next decision boundary.

## First paired batch: frozen before model execution

- Both arms: seed42, 256 **new** optimizer updates, checkpoint updates64/128/256,
  11 image forwards per update, 2816 training image forwards per arm, 7200 seconds
  maximum training wall time per arm. No restore of historical optimizer/RNG.
- User subsequently requested maximal use of all eight GPUs. Execution changes
  to four ranks per arm (A GPUs0-3, B GPUs4-7), preserving the global11-image update
  and equal-image objective. Uneven local work (3/3/3/2 images) must reduce to the
  serial gradient before clipping/AdamW. This is an execution-topology amendment,
  not an additional arm or dose. Require real distributed qualification before
  launch; preserve the earlier single-GPU runtime draft as superseded preparation.
- Cold natural readback of each immutable source adapter (new update0) and each
  saved update64/128/256: 88 requests total, each capped at3084 generated tokens.
  Saved source adapters remain immutable even if their cold readback is scheduled
  after training. Update256 is the decision endpoint; earlier doses are diagnostics.
- Primary scoped218 metrics: FN count and FN/218, plus annotation-relative micro
  precision and F1. Eligible predictions are **all valid parsed predicted rows**;
  TP comes from the existing class-agnostic, cardinality-first one-to-one IoU>=0.5
  matcher. Micro F1 is 2TP/(218 + eligible predictions), aggregated over11 images.
  Its unmatched term is annotation-unmatched, never automatically physical FP.
  Per-image ratios and IoU0.8 remain separately labeled diagnostics.
- Preserve raw row totals, malformed, geometry-invalid, EOS/cap and category
  errors, including an explicit literal non-COCO-80 description count even when
  a box receives class-agnostic owner credit. Invalid rows do not enter this detection precision denominator but
  cannot disappear from stage acceptance. Physical unknown is separate from
  confirmed FP; unresolved repeat candidates remain distinct from confirmed
  physical repeats. Reuse only matching/evidence identities that support the claim.
- Independently evaluate scoped218, historical232 and current-known248 owner
  sets. Retained/gained/lost uses each arm's own source output; also compare the
  final A and B owner sets. A smaller training scope cannot erase historical loss.
- Choose the next start using final scoped FN/F1, incumbent retention and output
  errors together, without an undeclared composite score or age preference. If
  these disagree, report the tradeoff explicitly. No new arm, dose extension,
  refresh, ordering change or image expansion follows automatically from this batch.
- Technical failure or unequal realized training dose leaves the planned paired
  comparison unanswered. A failed reader/reducer can be repaired against retained
  checkpoints/raw evidence without retraining valid model execution.

### Interpretation limit

This contrast estimates the value of the two parameter starts under the same
new teacher and new-update budget. Their historical exposure differs by design.
A lower result at update256 does not prove an initialization cannot eventually
learn. Shared geometry is an engineering integration with preserved semantics;
both arms receive it, and the predecessor already used the hinge. This batch
cannot attribute gains to introducing geometry loss, removing uncertain rows,
resetting the optimizer, or an ordering change relative to the predecessor.
The strongest alternative to a persistent start-quality difference is different
learning speed under the fixed dose; saved intermediate curves can describe it,
but do not authorize an additional dose or a convergence claim.

## Active package ownership

Root owns this contract, the launch release, cross-package decisions and final
acceptance. Native L1 packages have disjoint writes; no L2 delegation is granted.

- `shared_geometry` (Sol xhigh): shared loss/config/runner integration, probe formula
  reuse and CPU plus minimum actual shared-entry checks. Owns GPU6 only for a
  bounded two-update/ten-minute mechanics smoke before the research launch.
- `teacher_evaluation` (Terra xhigh): compliant teacher, target partitions and
  saved-readback evaluation. CPU preparation first; physical review remains bound
  to exact unmatched evidence, not guesses about annotation-unmatched predictions.
- `dual_start_runtime` (Luna high): fresh-optimizer manifests, bounded tmux runtime,
  durable cold readback and recovery. Prepares commands now; research GPU launch
  follows root acceptance of the shared entry and teacher admission.
- `distributed_training` (Sol xhigh): the user-requested four-rank-per-arm backend,
  unequal-work gradient parity and bounded real multi-GPU qualification. Owns a
  separate probe module, coordinates GPU handoff after the shared-entry smoke,
  and supplies its command to the runtime owner. Qualification is at most two
  updates per arm (44 global image forwards combined), ten minutes per attempt.

Package evidence lives in dedicated children of the existing output root. Owners
share interface changes directly and route semantic changes to root. Existing
dirty documentation migration is preserved and is not a reason to delay this work.

## Lead acceptance checkpoints

- Teacher: 11 images, 218 included owners and 2088 active tokens; exact canonical
  COCO-80 membership, retained order, prompts, geometry and token identities checked.
  [Receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-lead-acceptance-v1/teacher.json).
- Shared geometry: default weight0.01, equal eligible-segment normalization, ordinary
  and streaming integration accepted. The real training entry completed one finite
  applied update and saved its checkpoint. Lead review reproduced an empty-only
  context crash; the owner corrected both empty paths, and the fresh focused suite
  passed 11 tests. [Receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-lead-acceptance-v1/shared-geometry.json).
- Provenance correction: the legacy readback initially labeled B step2444 actually
  loaded the full-fixedP-N16-v2 adapter. It cannot establish original-B performance.
  Preserve and supersede those derived results; the scheduled fresh source0
  readback is mandatory for both arms.

These are engineering/data admissions, not a result of the paired experiment.

## Released execution and continuation

The lead accepted real distributed qualification: both arms completed2 updates
and22 image forwards in about24.5 seconds, all four ranks agreed on parameter and
optimizer state, and saved adapters passed fresh inspection. The earlier failed
attempt consumed zero model forwards. Full execution is bound by
[release.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/release.json)
and runs in tmux `coordexp-dual-start-v3`, A GPUs0-3 and B GPUs4-7.

After controller settlement, verify its terminal and both training dose receipts.
Use `probes.training_set_completion.paired_evaluation score --preparation` with
`dual-start-evaluation-preparation-v1/preparation-v4.json`, `--runtime-result`
`dual-start-v3/readback-result.json`, each `--arm A|B` and `--step 0|64|128|256`.
Use the existing evaluation owner for eight endpoint scores, own-source comparisons
and final cross-arm comparison. Preserve raw readbacks and use fresh output paths.
Unmatched is not physical FP; review unresolved exact-image evidence before final
physical claims. A reader/reducer failure does not authorize retraining.

All accepted engineering receipts are under `dual-start-lead-acceptance-v1`.
Current quality and choice of a subsequent start remain pending full natural
greedy readback, retention and output-error assessment.

## Candidate for a subsequent round: CE normalization

User supplied the side-chat handoff after the paired v3 launch. Register this as
a candidate for the next decision, not an authorized new arm or a change to the
active trial. Current CE is mean_i(S_i/T_i); the candidate is sum_i(S_i)/sum_i(T_i).
Binary masks apply to both NLL numerator and token denominator; fully masked
images are excluded from the eligible-image denominator. Normally supervised EOS
remains supervised, and masking does not remove input context.

The frozen teacher has2088 active tokens across11 images, mean189.818. Lengths
range48 to345, so current per-token coefficients relative to token mean range
3.955 to0.550. Image528944 has78 tokens: its image weight would change from1/11
to78/2088. These are coefficient facts, not evidence of a causal FN/EOS effect.
Token mean gives more total weight to long teacher sequences and may better align
with micro owner recall when sequence length tracks object count; this remains a
hypothesis, since descriptions and EOS are also tokens and gradient difficulty
differs by image. No matched normalization experiment has established a winner.

If the subsequent decision is normalization at the selected initialization, the
smallest comparison holds that start, teacher, dose, seed, optimizer, geometry
weight/reduction, ordering and readback fixed, with a token-equal arm against the
matching sample-equal arm. Reuse an existing control only if all bindings match.
If the question is whether initialization ranking changes under normalization,
that requires both starts under both normalizations; one selected-start contrast
cannot establish the interaction. Neither extension is automatically authorized.

Use a global active-token denominator across ranks. Sample sum followed by image
mean would add a189.818-fold CE scale change for this teacher and change relative
geometry strength. Proper token mean removes that deterministic scaling artifact,
but need not preserve actual loss/gradient norms or clipping frequency. Keep
geometry reduction fixed and report those diagnostics alongside final natural
FN/F1, per-image and per-ledger retention, EOS, invalid output and physical errors.

## Final acceptance

The full paired batch is lead-accepted and closed. Both final256 endpoints cover
218/218 with zero output debt; A first shows that result at saved64, B at saved256.
Historical232/current-known248 still have14/30 missing. See
[the paired result](dual-start-results.md) and current state for the complete
retention, cost, remaining-scope and interpretation record. No next run is launched.
