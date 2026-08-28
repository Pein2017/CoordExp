## Context

See [proposal.md](proposal.md) for motivation and the delta
[specification](specs/coordexp-swift-human13-k-union-greedy-probe/spec.md)
for required behavior. The current Swift path already owns multimodal encoding,
no-padding packing, Qwen forward execution, planned-step accumulation, DoRA
optimization, checkpoint publication, HF/vLLM inference, and detection
evaluation. The missing piece is a deliberately narrow research adapter that
turns one hash-bound Human-13 panel and its frozen sampled trajectories into
well-defined arm events without weakening the generic blind-image policy.

This is an overfit-only laboratory. The research unit owns `G/H/M`, prefix and
suffix meanings, the arm matrix, and interpretation. The OpenSpec change owns
only the mechanical path needed to execute that unit faithfully. The original
bounded table is complete; the user has now authorized one successor that
repairs the unavailable A4/A6/A8-prime mechanics and executes only those arms
from fresh Source and optimizer states.

## Goals / Non-Goals

**Goals:**

- Materialize one immutable experiment ledger from exact panel, model,
  prompt, decode, parser, matcher, token, owner, prefix, duplicate, and target
  identities.
- Reuse the accepted Swift execution spine while adding only the four pure
  loss geometries and event projections required by the approved arms.
- Make K=16 discovery efficient and unambiguous through four physical batches
  of four explicit `n=1` requests per image.
- Keep every panel exposure at one parameter state, even when its isolated
  segments require several physical no-padding packs.
- Produce the shortest evidence path from CPU preflight to one production-
  shaped image and, only after separate authorization, to independent arm
  runs on at most eight GPUs.

**Non-Goals:**

- Generalizing StateBank admission, candidate-tree search, online target
  refresh, GT-derived coordinate search, K-miss supervision, or an owner
  bridge.
- Prefix-KV or image-encoder reuse, a new distributed trainer, chunk-local A4
  union objectives, exact optimizer resume, or a new artifact/evidence
  framework. Exact fixed-parameter A4 score/gradient replay is in scope because
  it preserves the original logical objective.
- Validation, checkpoint promotion, production readiness, or any claim beyond
  fitting the exact thirteen images.
- Launching a model, allocating a GPU, or changing parameters during this
  planning change.

## Decisions

### 1. An experiment-local manifest owns research semantics

**Concept and owner.** A `Human13KUnionManifest` is produced by an
experiment-local builder under `scripts/research/` and consumed by a matching
experiment-local runner. It contains content-addressed input identities plus
typed image, trajectory, owner, selected-row, prefix, duplicate-event, arm,
and denominator records. The Human-13 research unit owns their meaning; the
manifest code owns validation and serialization only.

**Caller knowledge.** The builder knows the exact panel hash, Source payload,
prompt/wrapper/tokenizer, K recipe, parser/matcher versions, and the explicit
`overfit_only` unit identity. The generic StateBank loader sees none of these
exceptions and remains unchanged.

**Hidden implementation.** The builder first classifies chronological
duplicates on raw trajectories, then excludes those rows before owner matching
and `G/H/M` projection. Prefix token surgery, target-row selection, duplicate
provenance, and zero-denominator checks remain inside the builder. Consumers
receive frozen token spans and declared loss roles; they do not reconstruct
scientific labels from generated text. Manifest finalization proves the
duplicate-row set is disjoint from every matched, replay, target, and candidate
positive set.

**Failure and receipts.** Any identity mismatch, missing/duplicate seed,
ambiguous owner assignment, non-complete target row, altered token span,
unlicensed terminal target, or denominator mismatch fails before model
execution. One canonical JSON manifest plus its SHA-256 is the receipt.

**Alternative rejected:** adding a general StateBank opt-out or new ordinary
config flag. That would broaden blind-data behavior to solve a one-panel
research need and make accidental reuse difficult to detect.

### 2. K discovery uses explicit request fan-out, not `n=4`

The collector reuses the backend-neutral vLLM session but submits, for each
image, four successive physical batches. Each batch contains four independent
requests with `n=1` and an explicit seed; results are restored to canonical
seed order before sealing. Sampling uses temperature `0.4`, top-p `0.95`,
repetition penalty `1.10`, and 512 new-token maximum. Clean-greedy evaluation
remains an HF batch-size-one request with repetition penalty `1.0`.

This preserves request-to-seed identity and exploits batch scheduling without
depending on engine-specific child seed expansion. Runtime cache counters are
recorded when exposed, but the design never infers actual prefix or visual
reuse from shared request content.

**Alternative rejected:** one request with `n=4`. It is shorter to write but
couples scientific identity to vLLM child indexing and obscures per-request
seed provenance.

### 3. Raw and clean prefixes are different immutable views

The manifest retains exact generated token identifiers. For Source greedy and
every K trajectory it derives `P_clean` by left-to-right deletion of each later
complete row whose class-agnostic predicted-box IoU with an earlier retained
complete row exceeds `0.95`; retained spans are concatenated without re-
tokenization. The duplicate classification wins over a potential distinct-owner
match, as required by the user's chronological rule. `P_raw` continues to own
every duplicate-event decision state. Unmatched non-duplicate and invalid
spans remain context-only and are masked.

Every duplicate event selects only its final box-closing coordinate token for
fp32 unlikelihood. The pure helper evaluates the mathematically equivalent
`softplus(z_target - logsumexp(z_non_target))`, never `1-softmax`, so saturated
events stay finite. All events are consumed, then normalized first within image
and then across eligible images. This avoids whole-row penalties on shared
wrappers/descriptions while satisfying the uncapped event contract.

**Alternative rejected:** deleting duplicate rows without preserving their raw
states, or applying whole-row/tokenwise unlikelihood. The former destroys the
causal negative context; the latter suppresses many non-identifying shared
tokens.

### 4. Pure loss helpers consume selected compact logits

Add a focused module `src/losses/human13_k_union.py` with side-effect-free fp32
helpers for:

- owner-mean masked row CE;
- once-per-image prefix-free union mass over exact native candidate rows;
- coherent full-residual token bottleneck hinge; and
- duplicate-event token unlikelihood.

The experiment runner maps manifest token sites to existing compact causal
logits and passes tensors plus explicit denominators into these helpers. The
helpers do not parse rows, match owners, select targets, modify prefixes, or
decide arm membership. They return a scalar numerator/denominator and bounded
diagnostics suitable for the existing planned-step finite gate.

A8-prime and A1 share exactly one coherent full-H chain per image. A8-prime's
competitor index is detached and its required margin is sealed by the
no-update cross-surface census. A4 keeps each image's entire candidate set in
one logical normalization. If that set exceeds one physical pack, the runner
first scores every candidate at unchanged parameters, computes the global
fp32 softmax weights, then replays the same candidates with detached weights
and accumulates all gradients before one AdamW step.

**Alternatives rejected:** extending the generic rollout-calibration objective
registry with research-specific owner policy, or implementing independent H1
top-1 margins. The former makes a narrow experiment a production abstraction;
the latter asks mutually exclusive sibling tokens to be strict argmax at the
same state.

### 5. Existing isolated packing is adapted at the event boundary

The experiment runner emits logical segments into the accepted no-padding
varlen path. Every segment repeats its own image/prompt/prefix and receives an
independent causal-attention boundary and MRoPE reset. Independent segments are
stably sorted by descending encoded length and first-fit under 12,000 tokens.
A1, A8-prime, and full-GT keep one coherent segment per image. A4 keeps one
logical atomic candidate group per image, while its candidate segments MAY be
distributed across physical packs.

For A4, let `s_j(theta)` be the summed token log-probability of candidate row
`j`. The reference loss is `-logsumexp_j s_j`. Its exact gradient is
`-sum_j softmax(s)_j * grad(s_j)`. The implementation therefore performs:

1. a no-grad score pass over all candidate packs at one unchanged `theta`;
2. one fp32 global per-image `logsumexp` and normalized weight vector;
3. a differentiable replay of the same candidate packs using detached global
   weights; and
4. one optimizer step only after every replay/background pack contributes.

This is physical streaming of one logical objective, not multiple union
losses. Chunk-local `-logsumexp` terms, parameter updates between passes,
incomplete candidate coverage, or stale/mismatched scores fail closed.

Before backward, the runner knows complete panel denominators for owner,
image, and duplicate-event terms. Each physical pack contributes a globally
scaled numerator; all packs are accumulated at unchanged parameters, followed
by exactly one AdamW step for the panel exposure. The implementation records
pack count, logical and packed tokens, padding, utilization, GPU seconds, wall
time, and peak memory.

The resolved arm plan carries the exact trainable surface, AdamW values,
sixteen-step cosine schedule, clipping, and family coefficients frozen by the
spec. The materializer rejects absent or changed values. A7 and A0 are explicit
coefficient ablations; the runner never automatically normalizes the sum of
active family weights.

**Alternatives rejected:** padding the panel, stepping after each event or
pack, adding prefix-cache reuse, or computing one A4 union loss per chunk.
Padding wastes the dominant variable-length space; intermediate steps change
the estimand; cache reuse needs a different causal/gradient design; chunk-local
normalization is not the declared union objective.

### 5a. A6 and A8 repairs preserve sealed provenance

A6's donor treatment prefix is the exact donor trajectory prefix before the
selected target row after deletion of all earlier frozen duplicate-row spans.
The manifest binding, materializer, runner validation, and receipt SHALL derive
the same bytes; no raw-prefix fallback is permitted.

The A8 census may clone processor skeletons for packed and HF surfaces, but the
clone SHALL preserve experiment-local prompt-boundary, owner-row-token, and
image-identity metadata. Census publication is all-or-nothing under a new
immutable root. Only a complete output plus execution receipt can freeze the
packed-versus-HF drift and required margin; an abandoned plan or partial run is
not evidence.

### 6. Eight GPUs parallelize arms, not one arm

When separately authorized, the matrix launcher runs each arm as an isolated
world-size-one Accelerate process on one assigned GPU, with a byte-identical
Source payload, fresh AdamW state, and unique output root. At most eight arms
run concurrently. No DDP collective is introduced for this thirteen-image
panel.

The launcher treats applicable arms as independent jobs. Conditional A6 is
materialized only if the frozen ledger contains an eligible `H_mid` donor; it
does not occupy a GPU otherwise.

**Alternative rejected:** one eight-rank DDP arm. Uneven, arm-specific atomic
groups would add cross-rank scheduling and denominator choreography while
serializing the scientific matrix.

### 7. One analyzer owns the decision projection

`scripts/research/analyze_human13_k_union.py` consumes the frozen manifest and
Source/arm clean-greedy raw outputs. It repeats the frozen chronological
duplicate exclusion, gives later duplicates burden but no owner credit, and
then applies the declared cardinality-first, maximum-total-IoU one-to-one
matcher to retained rows. It emits one owner-outcome table with
per-image, legacy-twelve, image-2299, and pooled slices. Gains, retention,
losses, K-miss incidental gains, burden, and runtime are never collapsed into
one net score.

Teacher-forced/trie/margin metrics remain secondary columns. Only original-
prompt clean-greedy owner outcomes can answer the research question.

**Alternative rejected:** matching before duplicate exclusion, reusing the
visualization greedy matcher, or selecting an arm by training loss. Their
assignment/estimand differs from the owning unit's decision metric.

### 8. Verification stops at conclusion-changing seams

Implementation, once authorized, follows four gates:

1. CPU-only manifest, loss, packing-plan, and analyzer tests;
2. separately authorized full-panel Source/K discovery, canonical manifest
   freeze, and no-update census;
3. one eligible image selected from that manifest through materialization,
   pack, forward/backward, one
   update, checkpoint write/read, HF clean greedy, and outcome projection; and
4. the approved matrix only after the user accepts the vertical-slice cost and
   mechanics.

Each gate reuses the same compact manifest and ordinary Swift run artifacts.
There is no second evidence journal, general seal service, cache subsystem, or
exhaustive permutation framework. Missing optional telemetry is recorded as
unavailable and does not create another audit cycle.

## Risks / Trade-offs

- **[Fixed K support is incomplete]** -> Treat `M` as unknown and gradient-
  neutral; make only K-hit consolidation claims.
- **[A near-identical later row could match a distinct dense owner]** -> Apply
  the user-frozen duplicate rule before owner assignment and prove duplicate
  rows occur in no positive set.
- **[Clean-prefix surgery is counterfactual]** -> Preserve `P_raw`, exact
  removed spans, and separate raw-state duplicate loss; label treatment
  prefixes synthetic.
- **[Packed and HF logits may differ near ties]** -> Seal aligned strict-margin
  drift before training and mechanically block A8-prime if finite alignment or
  the bounded drift criterion fails.
- **[A4 logical groups exceed 12,000 tokens]** -> Require every physical
  segment to fit, then use exact fixed-theta two-pass streaming with global
  weights; fail if coverage, score/replay identity, or parameter-state
  invariance cannot be proven.
- **[Repeated complete segments do not save prefix FLOPs]** -> Claim only
  measured padding/launch efficiency and report actual runtime counters.
- **[World-size-one arms leave some GPUs idle when fewer than eight apply]** ->
  Accept idle capacity rather than fabricate extra arms or multi-rank work.
- **[Same-panel selection overfits adaptively]** -> State this as the intended
  laboratory; never infer validation or transfer.
- **[DoRA/optimizer may not fit even full GT]** -> Keep the full-GT capacity
  control separate so treatment failure is not confused with a capacity-floor
  failure.

## Migration Plan

This is additive and experiment-local; no production migration is required.
For the authorized missing-arm successor:

1. land pure records, manifest validation, and loss tests with all model paths
   disabled by default;
2. land the runner/config/analyzer path and verify CPU dry-run behavior;
3. retain the already sealed full Source/K ledger and run a fresh immutable
   no-update census after the A8 metadata repair;
4. verify the A6 clean-prefix repair and the A4 two-pass gradient equivalence
   through CPU tests and a production-shaped vertical slice;
5. execute only A4/A6/A8-prime under the user's bounded successor authority;
   and
6. retain or delete the experiment-local surfaces based on the unit's final
   disposition without changing stable production defaults.

Rollback is removal of the new experiment-local scripts, loss helper, tests,
and configs. Existing checkpoints, StateBank admission, Swift entrypoints, and
stable specs remain untouched.
