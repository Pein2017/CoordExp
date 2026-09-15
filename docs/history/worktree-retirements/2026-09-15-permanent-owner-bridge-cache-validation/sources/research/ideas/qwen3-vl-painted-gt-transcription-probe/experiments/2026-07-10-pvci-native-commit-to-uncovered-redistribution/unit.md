---
title: PVCI Native Commit-to-Annotated-Uncovered Redistribution
description: Tests whether a native completed object row acts as an instance-specific commit that suppresses revisit and redistributes continuation toward still-unmatched annotated objects.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-10-pvci-native-commit-to-uncovered-redistribution
topic: qwen3-vl-painted-gt-transcription-probe
status: active
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - autoregressive-commit
  - coverage
updated: 2026-07-10
---

# PVCI Native Commit-to-Annotated-Uncovered Redistribution

## Question

On a clean image and the ordinary one-shot detection prompt, does a freely
generated, syntactically closed, uniquely matched object row cause frozen
Qwen3-VL to perform an instance-specific set transition?

The required transition is stronger than repetition suppression: after
committing annotated instance `A`, the model must become less likely to revisit
`A`, more likely to continue toward a distinct annotated-uncovered instance
`B`, and must not merely move the removed probability into EOS, malformed
continuation, or an unmatched output.

This unit first tests the functional behavior of the native decoder. It does
not assume a slot, selector, anti-key, writable ledger, or final architecture.

## Decision Relevance

- Functional capability or uncertainty: whether Qwen's ordinary generated
  prefix already provides a usable instance-specific commit and coverage
  state, including how long that state persists across later row boundaries.
- Costly or hard-to-reverse choice this result could change: whether later
  training should strengthen a native prefix/KV transition, introduce an
  explicit commit/coverage state, or first solve a more basic instance-binding
  failure.
- Outside this unit's scope: learned slots or proposals, a new detector or
  backbone, trained coverage memory, final STOP calibration, architecture
  promotion, and claims of complete visible-object coverage.

## Definitions

- **Native:** clean-image generation from the ordinary compact detection prompt
  with no GT prefix, painted mark, forced object token, residual intervention,
  repaired row, or external proposal.
- **Uninterrupted native continuation:** the next row already present in the
  completed source rollout's single generation trajectory.
- **Full-prefix re-prefill continuation:** a matched diagnostic that supplies
  the exact native token prefix through `A` as model input and branches at
  repetition penalty `1.0` or `1.1`. It controls the decode processor but is
  not assumed numerically identical to the uninterrupted cached trajectory.
- **Commit:** a native row that reaches `<|box_end|>`, parses without repair,
  and matches exactly one still-available eligible annotation under the frozen
  description and IoU policy.
- **Annotated-uncovered:** an eligible annotated COCO-80 instance not yet
  claimed by an earlier one-to-one native match. Unmatched predictions remain
  unknown or ambiguous; they are not automatically false positives.
- **Functional redistribution:** a commit-conditioned decrease for the
  committed instance together with an increase in aggregate score or free
  selection of valid annotated-uncovered instances, without absorption by
  STOP or malformed output.
- **Persistence:** the committed instance remains selectively suppressed at
  later natural row boundaries. A later recovery of its score or native
  duplicate is evidence of forgetting or failed tracking, not a new object.

## Competing Hypotheses

- H1: native instance-specific commit and redistribution.
  - Expected signature: committing `A` lowers `A` relative to the same-image
    annotated-uncovered set, raises aggregate annotated-uncovered mass, and
    native continuation selects a distinct valid remaining instance. In
    symmetric diagnostic prefixes, committing `A` suppresses `A`, while
    committing `B` suppresses `B`, including same-class pairs.
  - Meaningful falsifier: only generic row position or class changes, the
    suppressed identity does not follow the committed instance, or removed
    mass moves primarily to STOP/malformed output.
- H2: fixed `geo_sorted` or salience traversal without coverage semantics.
  - Expected signature: the scheduled spatial successor wins regardless of
    which instance is represented as committed; candidate changes are
    explained by row index, prefix length, or the previous box location.
  - Meaningful falsifier: on the same image, matched counterfactual commits
    swap the specifically suppressed instance despite conflicting spatial
    schedule, and non-successor annotated objects gain probability.
- H3: repetition-penalty artifact rather than a native model transition.
  - Expected signature: committed token strings are suppressed only after the
    generation-time repetition processor; direct forward logits do not show
    instance-specific redistribution, and free decoding at penalty `1.0`
    loses the effect.
  - Meaningful falsifier: pre-processor direct-forward scores and penalty-1.0
    free continuation retain the instance-specific transition.
- H4: generic textual continuation with no instance ledger.
  - Expected signature: a valid row merely raises row-open versus EOS; phrase-
    only, geometry-only, mismatched phrase/box, and syntax-matched controls act
    similarly, or next-row identity remains unresolved.
  - Meaningful falsifier: only the correct phrase-geometry conjunction causes
    the matched instance-specific suppression and annotated-uncovered gain.
- H5: short-lived or capacity-limited native commit.
  - Expected signature: the effect is present immediately after a correct row
    but decays with row distance or object count and precedes native duplicate
    bursts in dense scenes.
  - Meaningful falsifier: suppression remains stable across later native row
    boundaries after controlling for candidate-set changes.

## Completion Promise

This unit is complete when:

- Evidence gate: the long-trained pure-CE/type-gated step-4887 checkpoint is
  identity-attested; its clean val rollout is exact and complete; a deterministic
  eligibility ledger is frozen before candidate scores are inspected; a
  two-event real-model prompt/logit/generation attestation passes through `A`;
  source-uninterrupted and full-prefix-reprefill continuations are kept
  separate; and every frozen eligible event is retained with explicit outcome
  or failure reason.
- Primary cohort: first-to-second-row native transitions with at least one
  annotated-uncovered object. Later transitions form a separately reported
  persistence/forgetting cohort and are clustered by source image.
- Acceptable evidence: source-uninterrupted native next-row outcomes; direct
  pre-processor
  conditional scores for committed, annotated-uncovered, and STOP candidates;
  offline repetition-penalty scores; matched full-prefix-reprefill penalty-1.0
  versus penalty-1.1 continuation; restart-drift receipts; fixed-order controls;
  symmetric instance-commit controls; same-class strata; all denominators; and
  exact prompt/token/runtime receipts through the committed row.
- Insufficient evidence: raw attention alone; a probe/readout without native
  free continuation; lower repeat probability without an annotated-uncovered
  gain; EOS when annotations are exhausted; a result from the specialized E1
  checkpoint only; or a positive restricted to the `geo_sorted` successor.

## Outcome Interpretation

- If H1 passes in raw scores and free continuation: native prefix/KV commit is
  a viable substrate to strengthen before adding explicit memory.
- If suppression is raw but redistribution appears only after the repetition
  processor: the decode processor, not native coverage, explains the effect.
- If the scheduled successor wins every control: ordering is acting as an
  implicit cursor; no instance-level coverage claim is supported.
- If `A` decreases but STOP/malformed rises and remaining objects do not gain:
  suppression exists without conservation and likely worsens low recall.
- If symmetric commits work in candidate likelihood but not free decoding:
  native information exists but ordinary argmax/STOP dynamics fail to read it
  robustly; decoding and training credit assignment remain separate issues.
- If same-class commits cannot be separated: class/geometry memory is
  insufficient for instance commitment, even if aggregate scenes improve.
- If immediate commit works but decays before dense-scene completion: later
  research should target persistence or explicit write/read training rather
  than a new visual recognizer.
- If the result is negative at this handle: do not infer a universal model
  limit; the compact row language, fixed ordering, checkpoint objective, or
  chosen conditional readout may still be mismatched.
- Possibilities this probe cannot distinguish: the exact internal storage
  site, which attention/KV route implements the effect, or the best trainable
  mechanism for strengthening it.

## Evidence Scope

- Checkout or branch:
  `/data/CoordExp/.codex/worktrees/69ed/CoordExp`,
  `codex/continue-handoff-session`.
- Baseline commit: `e07c6b73` plus the current uncommitted research tooling.
- Primary checkpoint manifest:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json`.
- Primary checkpoint identity: `step-4887`, resolved training-config
  fingerprint
  `e86e2d98447c8235f2c390b2ed674fc9a312f64952aff35b917f74fd64ae7e06`.
- Base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
- Adapter scope: language-tower DoRA, rank 16, alpha 32; vision tower and LM
  head were not adapter targets. Special coord/syntax embedding deltas are
  loaded from the same checkpoint payload.
- Primary native-rollout artifact:
  `/data/CoordExp/outputs/infer/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_step4887_val200_bsz4_temp0_rp1p10_max3084_8gpu/`.
- Primary resolved config:
  `/data/CoordExp/outputs/infer/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_step4887_val200_bsz4_temp0_rp1p10_max3084_8gpu/resolved_config.json`.
- Current-runtime scorer config:
  `configs/coordexp_swift/infer/research/pvci_native_commit_pure_ce_step4887_probe.yaml`.
  This smoke-only config loads the same immutable checkpoint/prompt surface; it
  does not replace or regenerate the completed val200 rollout.
- Planned artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_native_commit_uncovered_redistribution/`.
- Completed scorer artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_native_commit_uncovered_redistribution/native_candidate_row_commit_full32/`.
- Completed analysis artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_native_commit_uncovered_redistribution/native_candidate_row_commit_full32/analysis/`.
- Executed-code provenance limitation: the completed scorer artifact receipts
  config, checkpoint, ledger, and event hashes, but it does not embed the
  scorer-script SHA-256 or scoped dirty Git state. The current script bytes are
  not retrospective proof of the executed bytes. Treat this bundle as a
  session-scoped research result; future probes must embed scorer/analyzer code
  identity before durable cross-session publication.
- Frozen CPU plan:
  `/data/CoordExp/outputs/painted_gt/pvci_native_commit_uncovered_redistribution/plan_pure_ce_step4887/`.
  `hashes.json` is the byte-level authority. `plan.json` additionally receipts
  the planner and research-unit SHA-256, exact normalized parameters, Git
  HEAD/branch/dirty state, source rollout artifacts, checkpoint manifest, and
  processed annotation manifests.
- Completed source rollout: clean val200, deterministic HF generation,
  `temperature=0`, `top_p=0.9`, `max_new_tokens=3084`, repetition penalty
  `1.1`, batch size 4, eight data-parallel ranks, strict compact-row parsing,
  and zero recorded inference errors. With greedy decoding, `top_p` is
  inactive but remains part of the provenance. The probe additionally replays
  selected next-row transitions at penalty `1.0`.
- Matching policy: generated-order one-to-one matching, normalized exact COCO
  description, and pixel-space IoU at least `0.50`. Immediate natural
  continuations are independently attributed against covered and
  annotated-uncovered partitions so repeats of earlier covered instances are
  not hidden in the unknown bucket.
- Annotation universe: exactly the materialized COCO-80 objects in
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`.
  Its SHA-256 matches the provenance manifest, and all first 200 rows match the
  scored rollout in image ID, object count, ordered descriptions, and all 5,776
  GT pixel coordinates under the frozen source formula
  `round(coord_bin * (extent - 1) / 999)`. The materialized schema carries no
  crowd/ignore fields. Historical raw crowd and ignore counts are unavailable,
  so this unit makes no claim about globally visible or unannotated objects.
- Planned sample window: 24-32 eligible transitions from at least 12 images,
  deliberately retaining first-to-second, later-row, dense, and same-class
  strata. The frozen plan selects 24 first-to-second transitions plus eight
  later-row persistence transitions. Expand only when a predeclared stratum is
  unavailable, never because an observed effect is weak.
- Known limitations: COCO annotations are incomplete; `geo_sorted` is a strong
  learned prior; teacher-forced candidate scores are diagnostic; and the
  primary checkpoint still reflects one compact serialization/objective.

## Procedure

1. Reuse the completed clean val200 rollout and receipt checkpoint, tokenizer,
   template, special-embedding, config, decode, image-plan, and output
   identities. Do not rerun inference merely to rebuild the ledger.
2. Build a CPU-only generated-order transition ledger. Freeze native commits,
   annotated-uncovered sets, exact token prefixes through each commit, natural
   next rows, fixed `geo_sorted` successors, strata, and exclusion reasons.
3. Run a two-event runtime attestation. Require exact prompt and native-prefix
   parity through `A`; direct raw logits before processors; and offline
   penalty-1.1 equivalence to the source generation score at the first token of
   `A`. Compare, but do not equate, the source-uninterrupted next row with the
   full-prefix-reprefill RP1.1 continuation.
4. At the boundary immediately before native row `A` and immediately after its
   closing `<|box_end|>`, score compact rows for `A`, every earlier covered
   annotation, every annotated-uncovered candidate, and STOP. For row zero the
   pre-boundary is the prompt; for later rows it is the exact native prefix
   through the preceding row. Report paired raw and offline-RP scores,
   aggregate remaining mass, prior-covered conservation mass, and the relative
   remaining-versus-`A` redistribution index.
5. Treat the frozen rollout's natural next row as the uninterrupted RP1.1
   outcome. Separately run matched full-prefix-reprefill next-row continuations
   at penalties `1.0` and `1.1`. Classify identity, geometry,
   duplicate-covered, annotated-uncovered, unknown/ambiguous, malformed, EOS,
   and truncation separately, and report exact/identity/coordinate restart
   drift instead of silently merging the two execution modes.
6. On predeclared same-image panels, render symmetric diagnostic commit
   prefixes for `A` and `B`, plus phrase-only, geometry-only, mismatched
   phrase/box, and token-length-matched syntax controls. These are causal
   diagnostics, never native headline events.
7. For later native boundaries, trace the committed candidate's score and
   native revisit outcome by row distance. Report persistence curves clustered
   by image; do not reinterpret annotation exhaustion as correct global STOP.
8. Interpret against H1-H5 and stop. Do not automatically implement a ledger,
   selector, slot system, or training objective.

## Pre-Launch Contract And Risk Gate

- Hold if the pure-CE step-4887 checkpoint, adapter, or special-token embedding
  payload fails identity validation.
- Hold if prompt or native prefix through committed `A` does not align exactly,
  or if offline penalty processing cannot reproduce the installed runtime
  semantics at the attested token. A next-row mismatch after full-prefix
  re-prefill is recorded as restart drift and cannot replace the source
  uninterrupted outcome.
- Hold if the ledger is selected after score inspection, if generated-order
  matching is replaced by outcome-favorable global matching, or if exclusions
  are silently dropped.
- Hold if continuation and next-instance identity are collapsed into one
  metric, if `geo_sorted` successor agreement is omitted, or if removed repeat
  mass is not accounted across annotated-uncovered, STOP, malformed, and
  outside-annotation outcomes.
- Hold if unmatched objects are labeled background/false without adjudication,
  or if EOS after annotation exhaustion is called a correct STOP.

## Observations

- Direct observation: the existing pure-CE step-4887 val200 artifact contains
  1,294 emitted rows. Generated-order exact-description and IoU-at-least-0.50
  matching yields 855 native commits and 783 transitions with at least one
  annotated-uncovered object. There are 155 eligible first-row commits, 628
  eligible later commits, and 522 eligible transitions with a same-class
  annotated-uncovered instance. The candidate-score-blind selection policy
  froze 24 primary and eight survival-conditioned persistence events before
  candidate scores were inspected.
- Planner exclusions: generated-order commit eligibility excludes 426 emitted
  rows with no still-available exact-description IoU match and 13 with multiple
  eligible same-description matches. Immediate natural-next attribution is
  separately partitioned so known repeats of already-covered annotations do
  not remain in the unknown bucket. No unmatched row is relabeled background
  or false positive.
- Descriptive native behavior before causal interpretation: among all 783
  eligible commits, the immediate next valid row uniquely matches an
  annotated-uncovered object in 498 cases, EOS occurs while at least one
  annotated-uncovered object remains in 81 cases, two immediate rows revisit
  current `A`, four revisit a different earlier-covered annotation, five are
  ambiguous among annotated-uncovered objects, and 193 remain unmatched under
  the strict annotation rule. Across the full later trajectory, eight eligible
  commits eventually revisit current `A`. These RP-1.1 rollout counts are not
  evidence of an endogenous ledger because repetition processing,
  `geo_sorted` traversal, and incomplete annotations remain confounders.
- Precision and runtime attestation: native model forward and generation remain
  `torch.bfloat16` so the probe does not replace the native trajectory with a
  new numerical regime. Candidate logits, raw log-softmax, and offline
  repetition-penalty logits/log-softmax are explicitly `torch.float32`;
  candidate-bucket logsumexp is `torch.float64`; sequence sums use
  `math.fsum`/Python double. The analyzer fails closed when these receipts are
  missing or inconsistent and performs no bf16/fp16 reduction.
- Full execution retained all 32 frozen attempts. Twelve of 24 primary
  first-to-second events scored, while 12 primary events and all eight
  persistence events failed the exact current-runtime prefix-replay contract.
  The failures are mostly coordinate-bin trajectory drift, not float32
  measurement failures. Therefore the scored primary cohort is a selected
  exact-replay subset, and persistence/forgetting remains entirely unmeasured.
- Conditional score shift on the 12 scored primary events: all 12 show the raw
  predicate `A decrease AND annotated-uncovered LSE increase AND relative
  index increase`; all 12 retain the same signs under offline RP1.1. Median raw
  sequence-sum changes are `A=-14.7232`, annotated-uncovered
  `LSE=+7.6179`, and relative index `+23.5180`. This weakens a pure decode-time
  repetition-penalty explanation but is not by itself functional coverage.
- Span decomposition localizes that shift. The committed row's raw phrase and
  coordinate spans fall by medians `-4.1544` and `-10.3412`; under offline
  RP1.1 they fall by `-7.7661` and `-16.1314`. Among 21 same-class
  annotated-uncovered candidates from five events, the candidate-weighted raw
  phrase-span median is approximately zero (`-0.00086`) while the coordinate
  span rises by `+3.0542` (`+3.0482` under RP1.1). This is evidence that the
  post-row state can distinguish candidate geometries even when the phrase is
  held constant. It remains compatible with coordinate-token memory and does
  not yet prove visually grounded instance commitment.
- The annotated-uncovered shift is broader than only the scheduled successor
  in the scored subset. After removing the next geo-sorted candidate, a
  remaining-set LSE is defined for nine events and rises by median `+3.3347`
  raw and `+5.5190` under offline RP1.1. The other three events have no
  remaining candidate after that exclusion. These are descriptive event-level
  results, not independent-candidate inference.
- Free behavior is weaker than the conditional readout. Only six of 12 source
  uninterrupted RP1.1 continuations, and likewise six of 12 RP1.0/RP1.1
  re-prefill continuations, uniquely match an annotated-uncovered object.
  Unknown continuations are not treated as negatives because annotations are
  incomplete and several retain the expected class or approximate geometry.
  STOP likelihood rises concurrently in 11 of 12 scored events. STOP is a
  one-token diagnostic while candidate scores are exact full-row sequences;
  they do not form an exhaustive same-granularity partition, so no probability
  conservation or mass-transfer claim is supported.
- Fixed-order conflict is still absent from the successful free outcomes. All
  six strict annotated-uncovered continuations are simultaneously the
  identity-adjacent successor, next geo-sorted uncovered object, and row-slot
  expected object; the other six strict outcomes are unknown. Thus the
  free-rollout evidence does not falsify an implicit raster/list continuation
  policy even though non-successor candidate mass rises in conditional scores.
- Runtime counterexample: current `flash_attention_2` did not reproduce the
  frozen prefix coordinates even though the historical summary names FA2;
  SDPA did reproduce the prefix through `A` exactly. After exact-prefix
  re-prefill, RP1.1 retained the next description `truck` but changed coordinate
  bins from `[45,519,386,627]` to `[48,519,386,624]`. Thus execution metadata
  and visible-prefix equality are insufficient to assume exact continuation
  equality; uninterrupted and re-prefill outcomes are now distinct evidence
  channels. Across the 12 scored events, only three RP1.1 re-prefill rows are
  token-exact restarts; nine preserve boundary and description but drift in
  coordinates.
- Counterexample or negative result: exact current-runtime replay is not a
  scalable persistence gate in the present environment. All eight predeclared
  later-row events drifted before their target boundary, so this run cannot
  answer whether a native commit persists, decays, or is forgotten.
- Artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_native_commit_uncovered_redistribution/native_candidate_row_commit_full32/analysis/report.md`.

## Interpretation

- Supported reading: after a valid first row, the native decoder state strongly
  suppresses the exact committed row and makes multiple remaining annotated
  rows more likely in direct raw logits. The same-class phrase-held-constant
  split shows that much of the discriminating signal can live at the
  coordinate span rather than only at the object word.
- Alternative reading: the decoder may be using ordinary text/coordinate-token
  history plus its learned geo-sorted list policy, not a visually grounded
  object ledger. A previously emitted coordinate sequence can suppress itself
  and favor later coordinates without representing object identity or
  coverage as such.
- Remaining uncertainty: whether suppression follows the committed visual
  instance, its spatial neighborhood, exact coordinate token IDs, its phrase,
  or merely row position; whether a counterfactual non-successor commit swaps
  the suppressed identity; why STOP rises alongside remaining-object scores;
  and whether any effect persists into dense late rows.
- More likely after this result: the ordinary prefix/KV state is a usable
  substrate for immediate row-to-row control, and the principal bottleneck may
  be reading that broad latent shift into one correct next instance rather
  than absence of all commit information.
- Less likely after this result: a pure repetition-penalty-only mechanism, or
  a score shift confined exclusively to the next geo-sorted candidate.
- Not established: a native instance ledger, visually grounded coverage,
  correct conservation away from STOP, or persistence/forgetting.
- Candidate implementations remain unresolved: strengthen native prefix/KV
  state, add an explicit commit/coverage carrier, or change selection/STOP.
  No choice is promoted until the committed identity is causally swapped.

## Research Unit Closeout

Observed:

An immediate post-row conditional score shift exists in the exact-replay
primary subset. It includes strong committed-coordinate suppression,
same-class-uncovered coordinate gain, and non-geo-successor remaining-set gain.
Free realization is only partial and remains aligned with the learned order.

Evidence gate:

Partially passed. Checkpoint, prompt/image, precision, processor, candidate, and
artifact contracts passed for 12 primary events. The exact replay gate rejected
20 events and all persistence events; those failures are retained in every
denominator and are not converted into negative model outcomes. Executed-code
identity is not embedded in the completed scorer artifact, so this closeout is
session-scoped rather than a durable cross-checkout reproduction receipt.

Supported:

Immediate prefix-conditioned candidate redistribution in raw logits; a
geometry-specific same-class contrast; and robustness of score signs to
offline repetition-penalty accounting.

Not supported yet:

H1 as a visually grounded instance-specific ledger, H2 falsification, H4
falsification, functional mass conservation away from STOP, or H5
persistence/forgetting.

Architecture update:

No architecture is promoted. The result justifies one smaller causal
disambiguation before any selector, slot, ledger, or training objective.

Next decider:

Run a paired same-image commit swap that puts the committed identity in direct
conflict with the geo-sorted successor. Prefer same-class `A/B` pairs and a
non-successor `B`. Compare no-commit, commit-`A`, commit-`B`, phrase/geometry
cross-splices, and a length-matched syntax control. Score `A`, `B`, other
remaining candidates, and STOP with the same float32 readout. Add spatially
jittered candidates with no shared coordinate-token IDs to distinguish a
region-level commit field from exact coordinate-token repetition. Only if the
suppressed spatial identity swaps with the committed row should later work
search for a trainable native commit/read bridge.

Promotion decision:

Not promoted. The completed bundle remains session-scoped until a future probe
records executed scorer/analyzer code identity.
