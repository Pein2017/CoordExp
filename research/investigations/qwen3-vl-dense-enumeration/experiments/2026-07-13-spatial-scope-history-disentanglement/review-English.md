---
title: Independent Audit of Spatial Scope and History Disentanglement
description: Independent scientific, research-contract, and implementation-readiness audit.
type: investigation
role: independent-review
authority: non_normative_research
reviewed_unit: 2026-07-13-spatial-scope-history-disentanglement
status: complete
updated: 2026-07-13
---

# Independent Audit of Spatial Scope and History Disentanglement

## Terminology and Name Registry

This registry covers every abbreviation, identifier, metric symbol, configuration
token, and coined mechanism used by this review. A name's presence here does not
mean the review accepts the causal interpretation implied by that name.

### Review severity and decision terms

- **Critical (`P0`)**: execution or interpretation would be fundamentally
  invalid. A Critical finding is reserved for a defect that is not already
  prevented by the planned unit's fail-fast or lifecycle gates.
- **High (`P1`)**: the issue must be resolved before the unit becomes ready or
  before the affected result becomes interpretable.
- **Medium (`P2`)**: an important ambiguity or reproducibility weakness that
  does not necessarily block the primary launch.
- **Low (`P3`)**: a clarity, maintainability, or presentation weakness.
- **Superiority**: a predeclared rule under which one arm is judged better than
  another by both a paired effect criterion and required safety guardrails.
- **Equivalence**: a predeclared rule under which the complete uncertainty
  interval for a paired effect lies inside a practically negligible margin.
  A nonsignificant difference is not, by itself, equivalence.
- **Noninferiority guardrail**: a predeclared lower bound showing that an arm
  does not lose an unacceptable amount of retention, precision, validity, or
  natural closure while improving rescue.

### Model, dataset, object, and protocol terms

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the approximately
  two-billion-parameter multimodal model family and frozen adapted checkpoint
  under study.
- **Common Objects in Context 80-category ontology (`COCO-80`)**: the closed set
  of 80 reportable object categories used by this unit. A visible object outside
  this ontology is out of evaluation scope, not necessarily background or a
  hallucination.
- **Validation-200 (`val200`)**: the existing fixed 200-image validation cohort
  used for planning and baseline sanity. It is not a population-representative
  benchmark by declaration.
- **Weight-Decomposed Low-Rank Adaptation (`DoRA`)**: the low-rank adapter method
  used by the frozen checkpoint.
- **Cross-entropy (`CE`)**: ordinary next-token negative-log-likelihood training.
- **Bounding box (`bbox`)**: four coordinates defining one rectangular object
  localization.
- **Bounding-box Intersection over Union (`IoU`)**: intersection area divided by
  union area for a predicted and reference box.
- **Terminal no-more-objects decision (`STOP`)**: the generated action that ends
  enumeration; it is not assumed to be a calibrated coverage certificate.
- **Core region**: the disjoint spatial cell whose half-open boundaries own a
  prediction when the prediction's global bounding-box center falls inside.
- **Halo region**: clipped contextual pixels surrounding a core; halo pixels
  provide context but do not own predictions.
- **Per-region reset**: every spatial call starts from the same fresh base
  prompt, with no prior accepted object rows.
- **Cumulative history**: earlier accepted rows are serialized into the prompt
  for later spatial calls. The present unit does not yet define “accepted” or
  the exact serialization completely; that is a finding below.
- **Independent bagging**: repeated sampled full-image calls from the same base
  prompt, with predeclared different pseudo-random seeds, followed by one
  deterministic object-level merge.
- **K spatial calls (`K`)**: the number of cells in the chosen grid and therefore
  the number of calls in each `K`-call arm.
- **Experimental arm (`A`)**: one fully resolved combination of image input,
  prompt history, decoding, call budget, and aggregation.
- **Reference object (`o`)**: one reportable object in either the official or
  audit-augmented frozen reference ledger.
- **Empirical probability (`P`)**: the observed conditional fraction over paired
  ledger objects; no parametric probability model is implied.
- **Seed vector**: the immutable ordered list of pseudo-random seeds. A spatial
  call binds a seed to canonical cell identity rather than traversal position.
- **Matched-call comparison**: arms have the same number of model invocations,
  per-call decode cap, and declared seed multiset. It does not mean equal
  computation or equal opportunities per object.
- **Owning-seed per-object contrast**: for object `o` owned by cell `j`, the raw
  detection outcome in spatial call `j` is paired with the raw full-image call
  using seed `j`. This is the smallest comparison that equalizes one sampled
  opportunity per object.
- **Same-feature post-vision restriction**: encode the unmodified full image
  once, hold those visual features fixed, and change only which already-computed
  visual features the multimodal decoder may consume. This is the later
  discriminator needed for a stronger post-vision localization claim.
- **Official-annotation ledger**: the unchanged source COCO-80 instance
  annotations, with original crowd and provenance fields retained.
- **Audit-augmented ledger**: the official ledger plus arm-independent,
  pre-execution human additions produced under a frozen audit and adjudication
  protocol.
- **JavaScript Object Notation (`JSON`)**: the structured serialization format
  used by manifests and receipts.
- **JavaScript Object Notation Lines (`JSONL`)**: one independent JSON record per
  text line, used for cohort and per-attempt artifacts.
- **OpenSpec**: the repository workflow for a stable reusable implementation or
  compatibility contract. It must not own scientific hypotheses, cohorts, or
  verdict thresholds.

### Arm identifiers

- **Full-Image Single Rollout (`FULL_SINGLE`)**: one complete-image rollout under
  the unit's resolved decode policy.
- **Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)**: `K` sampled
  complete-image rollouts, one per seed, followed by the frozen object merge.
- **Native-Scale Tile with Per-Tile Reset (`TILE_RESET`)**: unresized
  core-plus-halo tiles, each decoded from a fresh prompt.
- **Native-Scale Tile with Cumulative History (`TILE_CUMULATIVE`)**: the same
  tiles decoded while accepted earlier rows are present in the prompt.
- **Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)**: full-size
  canvases exposing one core-plus-halo region, each decoded from a fresh prompt.
- **Full-Canvas Masked Region with Cumulative History
  (`MASK_CUMULATIVE`)**: the same masked canvases decoded with accepted earlier
  rows in the prompt.
- **Full-Canvas Masked Core-Only Reset (`MASK_RESET_CORE_ONLY`)**: the masked
  reset arm exposing only the core and no halo.
- **Full-Canvas Masked Cumulative Permuted Order
  (`MASK_CUMULATIVE_PERMUTED`)**: the cumulative masked arm executed in one
  predeclared non-raster cell order.

### Hypothesis and invariant identifiers

- **Spatial-Scope Competition (`HYPOTHESIS_SPATIAL_SCOPE`)**: the proposed
  mechanism in which simultaneous object candidates suppress a locally valid
  next-object continuation.
- **History-Horizon Instability (`HYPOTHESIS_HISTORY_HORIZON`)**: the proposed
  mechanism in which a longer autoregressive prefix makes later object
  retrieval or continuation less stable.
- **Tile-Local Computation Explanation
  (`HYPOTHESIS_TILE_LOCAL_COMPUTATION`)**: the residual family in which native
  tiles help through fewer visual tokens, a reset coordinate/positional frame,
  padding differences, or local visual re-encoding.
- **Partition and Boundary Context (`HYPOTHESIS_CONTEXT_BOUNDARY`)**: the
  proposed effect of cut objects or missing surrounding context near spatial
  boundaries.
- **Geometry-Sorted Traversal Prior (`HYPOTHESIS_GEOMETRY_ORDER`)**: the proposed
  sensitivity to raster-like traversal because the adapter was trained with
  geometry-sorted rows.
- **Dense-Scene Annotation Gaps (`HYPOTHESIS_ANNOTATION_GAPS`)**: the proposed
  explanation that some unmatched predictions are real but absent COCO-80
  annotations.
- **Residual Recognition or Control-State Limit
  (`HYPOTHESIS_RESIDUAL_LIMIT`)**: the deliberately bundled residual family in
  which persistent misses reflect either inadequate recognition or inability to
  convert object-specific visual evidence into decoder control.
- **No-Resize Scale Invariant (`INVARIANT_NO_RESIZE_SCALE`)**: the protocol rule
  that input construction must preserve each object's original pixel scale.
  It is an invariant, not a causal hypothesis and not a claim of equal visual
  representations.

### Metric and diagnostic identifiers

- **Local Rescue Rate (`LRR`)**: for an arm, the fraction of reference objects
  missed by paired `FULL_SINGLE` that the arm detects.
- **Official-Annotation Local Rescue Rate**: `LRR` against the official ledger.
- **Audit-Augmented Local Rescue Rate**: `LRR` against the audit-augmented
  ledger.
- **Retention**: the fraction of reference objects detected by `FULL_SINGLE`
  that an arm also detects.
- **Average Precision (`AP`)**: the area summarized from the detection
  precision-recall curve under a fully named evaluator and threshold policy.
- **Manual unique recall**: the fraction of distinct audit-ledger objects
  matched at least once by final valid predictions.
- **Manual precision**: the fraction of final valid predictions matched
  one-to-one to the audit-augmented ledger.
- **Image-Clustered Bootstrap Confidence Interval**: an uncertainty interval
  built by resampling whole images and retaining all objects and paired arm
  outcomes for each sampled image.
- **`Local_Rescue_Rate_A`**: serialized metric key for `LRR` of arm `A`.
- **`Retention_A`**: serialized metric key for retention of arm `A`.
- **`Spatial_Scope_Effect`**: the unit's current name for
  `LRR(MASK_RESET) - LRR(FULL_BAG_K)`. This review recommends the
  behavior-level name **Masked-Canvas versus Full-Image Bagging Rescue
  Difference** because the contrast does not isolate an internal mechanism.
- **`Masked_History_Effect`**: the unit's current name for
  `LRR(MASK_RESET) - LRR(MASK_CUMULATIVE)`. This review recommends
  **Masked Reset versus Cumulative Accepted-Row Prefix Rescue Difference**.
- **`Tile_History_Effect`**: the unit's current name for
  `LRR(TILE_RESET) - LRR(TILE_CUMULATIVE)`. This review recommends
  **Tile Reset versus Cumulative Accepted-Row Prefix Rescue Difference**.
- **`Local_Frame_Effect`**: the unit's current name for
  `LRR(TILE_RESET) - LRR(MASK_RESET)`. This review recommends **Native-Tile
  versus Masked-Canvas Rescue Difference** because token count, coordinate
  frame, padding, context, and encoding all change.
- **Duplicate component**: an intended group of mutually duplicate-like
  predictions. The edge predicate, class rule, threshold, and component
  algorithm are not yet defined; this is a readiness gap.
- **Invalid or malformed rate**: attempted generated rows that cannot enter
  matching, divided by a predeclared attempted-row or attempted-call
  denominator. The exact denominator must be frozen.
- **Natural closure**: generation ends with `STOP` under the model's normal
  semantics, rather than by token cap, controller cap, error, or forced
  truncation.
- **Incremental unique recall**: the additional distinct ledger objects first
  recovered by each successive call under a frozen call order.

### Decode, processor, annotation, and provenance tokens

- **`do_sample`**: the generation switch that activates stochastic token
  sampling. `false` means deterministic greedy selection in the current
  backend.
- **Sampling temperature (`temperature`)**: positive scaling of token logits
  before sampling; a value of zero denotes greedy decoding in the existing
  artifacts.
- **Nucleus-sampling cumulative-probability threshold (`top_p`)**: the
  probability-mass cutoff defining the candidate token set during sampling.
- **Maximum newly generated tokens (`max_new_tokens`)**: the hard per-call cap
  on generated suffix length.
- **Repetition penalty (`repetition_penalty` or historical shorthand `rp`)**:
  token-logit adjustment for tokens already present in the considered sequence;
  `1.0` is neutral.
- **`prompt_ignore_length`**: the installed Transformers processor option that
  excludes a leading prompt span from repetition-penalty accounting.
- **`do_resize`**: the image-processor switch controlling geometric resizing.
  `false` must be attested for this unit.
- **`iscrowd`**: the COCO annotation field marking a crowd region, which has
  different official evaluation semantics from an ordinary individual
  instance.
- **`gaussian_rps`**: a historical artifact-path and configuration token only.
  Its canonical plain-language research name is **Gaussian Soft-Target
  Coordinate Cross-Entropy with Ordered Cumulative-Distribution Penalty**.
  This review does not promote `gaussian_rps` into a canonical term.
- **`coordexp_swift`**: provenance token naming the native CoordExp-Swift
  training and inference stack.
- **`prod`** and **`infer`**: provenance tokens for production-training and
  inference artifact families.
- **`qwen3_vl_2b`**: provenance token for the approximately
  two-billion-parameter Qwen3 Vision-Language model.
- **`desc_first`** and **`geo_sorted`**: provenance tokens for
  description-first row serialization and geometry-sorted object order.
- **`dora_r16a32`**: provenance token for Weight-Decomposed Low-Rank Adaptation
  with rank 16 and low-rank scaling parameter 32.
- **`llm_12000`**, **`accelerate8`**, **`ebs24`**, **`8epoch`**, and
  **`warmup0p1`**: provenance tokens for a 12,000-token packed-sequence limit,
  eight distributed Accelerate processes, effective batch size 24, eight
  training epochs, and scheduler warmup ratio 0.1.
- **`step-4887`** and **`checkpoint.json`**: provenance tokens for optimization
  step 4,887 and its JSON checkpoint manifest.

## Executive Verdict

**Verdict: narrow and re-review.**

- **Scientifically valuable:** yes. The unit has a strong intervention matrix,
  explicit non-claims, useful matched-call bagging, separate official and
  audit-augmented ledgers, and an appropriate no-architecture-promotion posture.
- **Ready for implementation:** no. The cumulative-prefix, object-merge,
  manual-ledger, decision-rule, and sampled-generation contracts are not yet
  executable.
- **Ready for execution:** no. The current inference backend is deterministic,
  while the defining `FULL_BAG_K` control requires seeded sampling.
- **Can the primary comparisons distinguish the intended mechanisms:** only at
  the policy level after corrections. They can compare a structured
  masked-input policy, a cumulative accepted-row prompt policy, and repeated
  full-image sampling. They cannot by themselves identify post-vision candidate
  competition, pure history length, or a vision-versus-language owner.
- **Single largest scientific uncertainty:** `MASK_RESET` changes the input to
  the vision tower. A gain can arise from mask-induced designation, background
  suppression, or changed visual encoding rather than competition among
  already-computed decoder candidates.

The audit fixed point is worktree
`/data/CoordExp/.worktrees/research-probes` at commit
`cea57a0ba070369312397daf14020222bcc6e204`. The worktree was already dirty and
the investigation tree was already untracked; those user-owned changes were
not altered. The named checkpoint manifest and historical `val200` artifact
directory exist, but no run directory for this unit exists and no artifact path
is treated as execution proof.

Historical painted-object evidence is relevant but bounded. Correct and
wrong-object visual marks strongly redirected outputs, showing that privileged
image-side designation can control the decoder
(`/data/CoordExp/.worktrees/research-probes/research/ideas/qwen3-vl-painted-gt-transcription-probe/overview.md:93-123`).
The same historical record explicitly forbids concluding that unmodified
full-image visual features were already sufficient
(`/data/CoordExp/.worktrees/research-probes/research/ideas/qwen3-vl-painted-gt-transcription-probe/overview.md:280-284`).
The autoregressive-binding synthesis reports conditionally recoverable misses
but labels most evidence as selected, tiny, or case-study scope
(`/data/CoordExp/.worktrees/research-probes/research/investigations/autoregressive-binding-template-study/findings.md:13-27`).
Those records motivate the current unit; they do not establish its mechanism.

## Severity-Ranked Findings

### Critical (`P0`)

No Critical finding is assigned. This is not an approval: the unit is still
`planned`, has `evidence_status: none`, requires a readiness amendment, and
contains fail-fast language that should prevent the current deterministic
backend from masquerading as sampled bagging. If that fail-fast boundary were
bypassed, finding P1-5 would become a fundamentally invalid primary comparison.
The residual Critical risk is therefore an execution that ignores the unit's
own readiness gates.

### High (`P1`)

#### P1-1 — The masked-canvas contrast is an intervention effect, not a candidate-competition localization

- **Category:** scientific-design and terminology finding.
- **Claim reviewed:** `MASK_RESET` rescue beyond `FULL_BAG_K` is presented as
  evidence for Spatial-Scope Competition and serialized as
  `Spatial_Scope_Effect`.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:120-128`,
  `:172-176`, `:183-187`, and `:391-407`.
- **Observed issue:** masking keeps canvas dimensions and nominal visual-token
  count but changes pixels before the vision tower, changes all downstream
  visual activations, and creates a conspicuous region boundary. The unit
  acknowledges the first part, yet its hypothesis and metric names still sound
  mechanistic.
- **Scientific or contract impact:** a positive contrast supports the utility
  of a masked-input spatial-restriction policy. It does not show that the
  full-image model had already recognized the rescued object, that competition
  occurs after the vision tower, or that simultaneous decoder candidates are
  the causal owner.
- **Strongest competing explanation:** mask-induced visual designation,
  background/contrast suppression, or different global visual encoding. The
  historical painted-object intervention shows that image-side cues can
  dominate selection without establishing autonomous enumeration.
- **Smallest correction or discriminator:** rename the estimand to
  **Masked-Canvas versus Full-Image Bagging Rescue Difference**, narrow all
  positive claims to the intervention policy, and reserve causal localization
  for a same-feature post-vision restriction unit.
- **Blocking scope:** interpretation and architecture promotion; it does not
  block mechanics execution or a policy-utility claim.

#### P1-2 — Matched call count is not matched opportunity per object

- **Category:** scientific-design finding.
- **Claim reviewed:** `FULL_BAG_K` matching `MASK_RESET` or `TILE_RESET` is
  described as a falsifier of spatial-scope benefit.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:183-187`,
  `:293-300`, `:306-325`, `:352-355`, and `:451-454`.
- **Observed issue:** every object is visible in all `K` full-image bagging
  calls, but unique core ownership gives a spatial object principally one
  score-bearing owning-cell call. Equal total calls therefore do not equalize
  stochastic opportunities for each object.
- **Scientific or contract impact:** `MASK_RESET` beating bagging is a strong,
  conservative policy result, but equality does not falsify a local-scope
  benefit. Post-merge equality can also hide different raw per-call mechanisms.
- **Strongest competing explanation:** full-image bagging catches up because it
  gives every object `K` sampled opportunities, while the spatial policy
  allocates calls across regions.
- **Smallest correction or discriminator:** retain post-merge bagging as the
  utility comparator, and add the owning-seed per-object raw contrast plus the
  pre-to-post-merge change for every object.
- **Blocking scope:** the claimed spatial-scope falsifier and mechanism
  interpretation; not the final-policy comparison.

#### P1-3 — Reset versus cumulative history does not isolate history length

- **Category:** scientific-design finding.
- **Claim reviewed:** reset exceeding cumulative history is treated as evidence
  for History-Horizon Instability and as a route to compact state or
  short-horizon training.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:60-63`,
  `:163-164`, `:183-186`, and `:199-204`.
- **Observed issue:** the contrast changes prefix length, semantic content,
  prior-object information, prior errors, repeated structural tokens, and the
  opportunity to suppress duplicates. Reset removes helpful history at the
  same time that it removes harmful history.
- **Scientific or contract impact:** the contrast identifies the total effect
  of a particular endogenous accepted-row prefix policy, not pure memory
  horizon or length instability.
- **Strongest competing explanation:** malformed or wrong prior rows, useful
  coverage information, class/coordinate token overlap, or learned traversal
  semantics rather than length itself.
- **Smallest correction or discriminator:** rename the effects to behavior-level
  reset-versus-cumulative-prefix differences. Before a pure horizon claim, add
  a small fixed-prefix panel separating clean canonical history, model history,
  wrong or shuffled history, and length-matched neutral history.
- **Blocking scope:** History-Horizon Instability interpretation and any
  state/training route justified specifically by prefix length; not the policy
  comparison.

#### P1-4 — Prompt-inclusive repetition penalty differentially acts on cumulative arms

- **Category:** scientific-design and implementation-readiness finding.
- **Claim reviewed:** the same frozen repetition penalty is treated as a
  controlled decode factor, with its causal study deferred.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:209-211`;
  `/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml:20-25`;
  `/data/CoordExp/.worktrees/research-probes/src/inference/backend.py:147-155`;
  and
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/generation/logits_process.py:297-313`
  and `:365-368`.
- **Observed issue:** installed Transformers 4.57.1 includes decoder-only prompt
  tokens in repetition-penalty accounting unless `prompt_ignore_length` is
  supplied. The current backend passes only the numeric penalty. Cumulative
  prompts contain prior class, wrapper, coordinate, and structural tokens that
  reset prompts do not.
- **Scientific or contract impact:** `RESET > CUMULATIVE` can be a direct
  logits-processor effect. Freezing the number `1.10` does not freeze its
  treatment because the token sets differ.
- **Strongest competing explanation:** prompt-token suppression mechanically
  reduces repeated object-wrapper, coordinate, class, or closure tokens in
  cumulative calls.
- **Smallest correction or discriminator:** use neutral
  `repetition_penalty=1.0` for the primary history contrast. Retain `1.10` only
  as a predeclared factorial sensitivity, or implement and attest a prompt-
  excluding processor if that is the intended semantics.
- **Blocking scope:** all history-horizon interpretation.
- **Lightweight executed probe:** with identical positive logit `2.0` and
  penalty `1.1`, the installed processor returned `1.818181753` when the token
  was in the prompt and `2.0` when absent. No model inference was run.

#### P1-5 — The live inference backend cannot execute sampled bagging

- **Category:** implementation-readiness finding.
- **Claim reviewed:** `FULL_BAG_K` is meaningful only with nonzero-temperature,
  independently seeded sampling.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:64-71`
  and `:329-347`;
  `/data/CoordExp/.worktrees/research-probes/src/config/inference.py:95-100`;
  `/data/CoordExp/.worktrees/research-probes/src/inference/backend.py:19-25`
  and `:147-155`; and
  `/data/CoordExp/.worktrees/research-probes/src/inference/pipeline.py:759-769`.
- **Observed issue:** the schema has no sampling flag or per-request seed,
  `DecodeRequest` carries neither temperature nor nucleus cutoff nor seed, the
  backend hard-codes `do_sample=false`, and the artifact policy records that
  deterministic choice.
- **Scientific or contract impact:** repeated live-backend calls are a
  determinism control, not independent bagging. Without `FULL_BAG_K`, the
  primary structured-scope comparison has no generic-sampling control.
- **Strongest competing explanation:** any apparent masked advantage is merely
  a comparison against repeated identical greedy decoding.
- **Smallest correction or discriminator:** after design acceptance, implement
  a research-only sampled decode path with explicit per-request seed ownership,
  temperature, nucleus cutoff, and reproducibility across batch sizes and
  traversal order. A tiny receipt must show stable replay and more than one
  unique raw rollout where the policy is expected to vary.
- **Blocking scope:** implementation readiness, execution, `FULL_BAG_K`, and
  all spatial-versus-bagging interpretation.

#### P1-6 — “Accepted rows” and cumulative prompt construction are not executable

- **Category:** research-contract and implementation-readiness finding.
- **Claim reviewed:** cumulative arms append accepted earlier rows while
  otherwise matching their reset arms.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:60-63`,
  `:104-116`, `:286-300`, `:311-339`; and
  `/data/CoordExp/.worktrees/research-probes/src/inference/prompt.py:50-74`.
- **Observed issue:** the unit does not say whether acceptance happens after
  parsing, class normalization, core ownership, invalid-row dropping, duplicate
  handling, or object merge; whether appended text is raw or canonicalized; how
  rows are ordered; or what happens at context overflow. `TILE_CUMULATIVE` also
  lacks a frozen choice between tile-local and global coordinate-token frames.
  The current prompt builder removes all assistant messages, so cumulative
  prompting is not an existing supported path.
- **Scientific or contract impact:** independent implementations can feed
  different histories, and a result can become a parser, ownership, coordinate,
  filtering, or truncation effect.
- **Strongest competing explanation:** halo predictions, coordinate-frame
  mismatch, canonicalization, or silent context truncation poisons only the
  cumulative arm.
- **Smallest correction or discriminator:** freeze a call-by-call state
  transition: exact admission stage, raw/canonical text rule, coordinate frame,
  ordering, duplicate behavior, prompt bytes and token identifiers, cache/reset
  semantics, and fail-fast context-window policy. Verify a two-cell synthetic
  fixture with exact second-call prompt receipts.
- **Blocking scope:** implementation and execution of all cumulative arms, plus
  every history contrast.

#### P1-7 — Post-merge object metrics lack a frozen semantic owner

- **Category:** research-contract and implementation-readiness finding.
- **Claim reviewed:** post-merge predictions own primary rescue and retention,
  using a frozen class normalization, merge, duplicate, and matching policy.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:74-76`,
  `:298-300`, `:327-339`, and `:352-376`;
  `/data/CoordExp/.worktrees/research-probes/src/inference/merge.py:83-101`;
  and
  `/data/CoordExp/.worktrees/research-probes/openspec/specs/coordexp-swift-detection-evaluator/spec.md:125-132`
  and `:164-171`.
- **Observed issue:** no object-level merge is defined. The live
  `src/inference/merge.py` merges distributed artifact shards, not overlapping
  detections. The current official evaluator is aggregate-only and explicitly
  limits category normalization to lowercase and whitespace, while the unit
  requests an unspecified alias map.
- **Scientific or contract impact:** different suppression/fusion, score,
  ownership, clipping, alias, and duplicate rules can change `LRR`, retention,
  precision, persistent misses, and `AP`.
- **Strongest competing explanation:** apparent rescue or precision change is
  produced by object aggregation or category remapping rather than the model
  intervention.
- **Smallest correction or discriminator:** freeze one pure object-level
  merger and one research analyzer, including class behavior, score source,
  overlap predicate, ordering, tie-breaks, ownership timing, coordinate
  clipping, duplicate graph, and pre/post rows. Either use identity-only
  official category normalization or version a research alias table and keep
  its metrics separately named.
- **Blocking scope:** implementation, execution, and every post-merge
  object-level headline.

#### P1-8 — The outcome map has no executable effect or safety decision rule

- **Category:** scientific-design and research-contract finding.
- **Claim reviewed:** arms that “exceed,” “match,” or are “approximately equal”
  drive named next routes, while `LRR` differences are the mechanism headline.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:183-191`,
  `:377-426`, and `:465-475`.
- **Observed issue:** no minimum meaningful effect, superiority rule,
  equivalence margin, retention floor, precision floor, or duplicate/invalid/
  closure noninferiority rule is predeclared. Safety diagnostics are only
  “also report.”
- **Scientific or contract impact:** the same primitive results can be assigned
  different outcome cells. An arm can increase `LRR` by emitting many rows while
  losing prior true positives, precision, validity, or closure.
- **Strongest competing explanation:** generic continuation or prediction-count
  inflation, not improved enumeration. Historical proposal feedback produced a
  small recall rise together with large precision, duplicate, invalid, and
  closure failures
  (`/data/CoordExp/.worktrees/research-probes/research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-causal-proposal-bridge/own-prefix-causal-behavior-results-2026-07-12.md:135-186`).
- **Smallest correction or discriminator:** freeze paired superiority and
  equivalence rules plus joint noninferiority guardrails for retention, manual
  precision, duplicates, invalid rows, natural closure, and prediction count.
  Define every diagnostic formula and denominator.
- **Blocking scope:** interpretation, route decisions, and architecture
  promotion; mechanics-only execution could precede it but cannot be
  metric-bearing.

#### P1-9 — The audit-augmented ledger is not reproducible or unambiguously arm-independent

- **Category:** research-contract finding.
- **Claim reviewed:** an exhaustive COCO-80 manual ledger is created before
  execution, while manual audit of new unmatched predictions is blinded to arm.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:268-277`,
  `:375-379`, `:409-414`, and `:441-449`.
- **Observed issue:** pre-execution exhaustive review and post-output “new
  unmatched prediction” review are not reconciled. The unit does not freeze
  annotator count, image-only blinding, visibility/partial-instance rules,
  ambiguous and crowd states, box policy, disagreement adjudication, immutable
  object identifiers, ledger version, or content hash.
- **Scientific or contract impact:** a mutable or proposal-triggered ledger can
  favor the arm that emits more candidates and directly inflate audit-augmented
  rescue and manual precision.
- **Strongest competing explanation:** audit ascertainment bias or rater
  disagreement, not missing official labels.
- **Smallest correction or discriminator:** seal one image-only,
  arm-independent ledger before outputs exist, using at least two independent
  reviewers and adjudication with explicit `accepted`, `ambiguous`,
  `partial`, `crowd`, and `out-of-scope` states. If predictions are later
  adjudicated, pool and deduplicate them with hidden provenance in a separately
  named secondary ledger.
- **Blocking scope:** audit-augmented `LRR`, manual recall/precision,
  annotation-gap interpretation, and the scientific headline.

#### P1-10 — Current val200 provenance cannot support official crowd semantics

- **Category:** dataset-validity and implementation-readiness finding.
- **Claim reviewed:** official COCO precision, recall, and `AP` will be reported
  on the val200-derived cohort, with crowd summaries used in construction.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:263-274`
  and `:416-420`;
  `/data/CoordExp/.worktrees/research-probes/src/eval/detection_consumer.py:478-495`;
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl:1`;
  and
  `/data/CoordExp/public_data/coco/raw/annotations/instances_val2017.json:1`.
- **Observed issue:** a read-only join found 1,444 non-crowd instances and 16
  crowd annotations across 16 of the selected images, but the val200 JSONL
  carries no `iscrowd` field. The live evaluator reconstructs every supplied
  annotation with `iscrowd: 0`.
- **Scientific or contract impact:** predictions overlapping omitted crowd
  regions can become apparent false positives, and “official” metrics do not
  reproduce official crowd-ignore behavior.
- **Strongest competing explanation:** a precision loss reflects dropped crowd
  provenance rather than hallucination or annotation incompleteness.
- **Smallest correction or discriminator:** join the frozen cohort to raw COCO
  annotations by immutable image and annotation identifiers, preserve crowd
  regions for official evaluation, and exclude them from individual-object
  `LRR` or define a separate non-instance policy.
- **Blocking scope:** official `AP`/precision interpretation and any manual
  claim involving crowd regions; it does not block non-crowd mechanics.

### Medium (`P2`)

#### P2-1 — Several listed “falsifiers” are only negative results for one intervention

- **Category:** scientific-design finding.
- **Claim reviewed:** the hypothesis table and falsification signatures use
  equality or null effects to falsify broad, nonexclusive mechanisms.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:181-191`
  and `:451-463`.
- **Observed issue:** candidate competition, tile-local computation,
  useful-versus-harmful history, annotation gaps, and residual recognition/
  control can coexist. A null under one mask, halo, seed vector, or sample does
  not falsify the general mechanism.
- **Scientific or contract impact:** negative evidence can be promoted into a
  stronger exclusion than the intervention supports.
- **Strongest competing explanation:** an ineffective handle, insufficient
  precision, or simultaneous opposing effects.
- **Smallest correction or discriminator:** relabel these cells as
  **disconfirming observations under the chosen intervention**, state the
  measurement/null failure for each hypothesis, and reserve “ruled out” for an
  adequate, mechanism-targeting test.
- **Blocking scope:** broad exclusion claims; none for descriptive execution.

#### P2-2 — One permutation measures one order, not the geometry-sorted training prior

- **Category:** scientific-design finding.
- **Claim reviewed:** one frozen permuted cumulative order is the
  Geometry-Sorted Traversal Prior control.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:114-116`,
  `:183-189`, `:311-320`, and `:336`.
- **Observed issue:** the comparison changes which region occurs at each prompt
  length, which errors enter the prefix, spatial adjacency, and one particular
  stochastic realization. It has no reset-permuted orchestration control and no
  counterbalanced order set.
- **Scientific or contract impact:** a difference establishes order
  sensitivity at most; it cannot attribute that sensitivity to geometry-sorted
  adapter training.
- **Strongest competing explanation:** generic row-position, prompt-length,
  spatial-adjacency, or error-propagation effects.
- **Smallest correction or discriminator:** canonicalize merge inputs by cell
  identity, add reset-permuted control, and use multiple predeclared reverse or
  counterbalanced orders. A random-order-trained adapter is the later direct
  training-prior discriminator.
- **Blocking scope:** geometry-training causality only.

#### P2-3 — The boundary hypothesis lacks a native-tile core-only comparison

- **Category:** scientific-design finding.
- **Claim reviewed:** core-plus-halo versus core-only behavior tests whether
  hard tile cuts or missing context dominate tile behavior.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:129-131`,
  `:183-189`, and `:311-320`.
- **Observed issue:** `MASK_RESET_CORE_ONLY` is the only core-only arm.
  Masked-canvas halo utility does not directly test native tile cropping,
  padding, or local re-encoding at boundaries.
- **Scientific or contract impact:** the current arm can support a masked-input
  context result but not a hard-tile boundary attribution.
- **Strongest competing explanation:** halo changes visible candidate count or
  mask area rather than repairing a tile cut.
- **Smallest correction or discriminator:** either add a secondary native-tile
  core-only arm under the same tile processor contract or narrow the hypothesis
  and result language to masked-canvas halo utility.
- **Blocking scope:** hard-tile boundary interpretation only.

#### P2-4 — Post-merge rescue is valid for policy utility but insufficient for mechanism accounting

- **Category:** metric-validity finding.
- **Claim reviewed:** primary `LRR` and retention use post-merge predictions;
  pre-merge rows are retained only for diagnostics.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:352-355`
  and `:416-426`.
- **Observed issue:** a nonlinear merge can create or remove object matches, and
  unioning repeated opportunities inflates the chance of at least one match.
  `FULL_BAG_K` is the right utility control but does not explain raw opportunity
  differences.
- **Scientific or contract impact:** a post-merge gain can be aggregation gain
  rather than changed single-call object selection.
- **Strongest competing explanation:** duplicate suppression, score ordering,
  or repeated opportunity.
- **Smallest correction or discriminator:** keep post-merge `LRR` as primary
  policy utility, but report raw owning-seed detection, raw ever-detected union,
  and the per-object pre-to-post-merge delta.
- **Blocking scope:** mechanism interpretation; none for a fully specified
  final-policy metric.

#### P2-5 — The matching tie-break is not fully algorithmic

- **Category:** metric-validity and implementation-readiness finding.
- **Claim reviewed:** maximum-cardinality, then maximum-total-IoU matching uses
  “source order” as the deterministic final tie-break.
- **Exact handle:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:357-365`.
- **Observed issue:** “source order” does not say whether prediction order,
  reference order, edge order, or a lexicographic assignment vector wins when
  several assignments have equal cardinality and numerically equal total IoU.
  Solver tolerance is also unspecified.
- **Scientific or contract impact:** different valid implementations can assign
  duplicates to different objects and change persistent-miss identity.
- **Strongest competing explanation:** analyzer tie resolution, not arm
  behavior.
- **Smallest correction or discriminator:** define immutable prediction and
  reference indices, numeric tolerance, and one lexicographic assignment
  ordering after the two primary objectives; test an equal-IoU same-class
  fixture.
- **Blocking scope:** exact reproducibility of matching; not most ordinary
  cases.

#### P2-6 — Bootstrap scope is conditional on one seed vector and an underdefined audited sample

- **Category:** statistical-validity finding.
- **Claim reviewed:** 10,000 image-clustered replicates yield a 95-percent
  interval for each paired estimand.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:68-71`,
  `:268-277`, and `:375-379`.
- **Observed issue:** the interval construction does not define percentile
  method, paired-effect recomputation, empty-denominator replicates,
  stratified resampling/weights, or whether inference is conditional on the
  fixed seed vector. Image resampling does not estimate decoding-seed
  variability.
- **Scientific or contract impact:** uncertainty may be attributed to the wrong
  target population or interpreted as covering stochastic decode variability
  that was never resampled.
- **Strongest competing explanation:** a seed-specific or deliberately
  oversampled-stratum effect.
- **Smallest correction or discriminator:** freeze the interval algorithm,
  strata/weight target, empty-replicate rule, and state explicitly that the
  primary interval is over images conditional on one decode seed vector. Add
  seed-block replication only if the claim generalizes to the sampling policy.
- **Blocking scope:** broad population or decode-policy uncertainty claims; not
  point estimates on the frozen cohort.

#### P2-7 — Calling out-of-ontology visible objects “background” creates a claim trap

- **Category:** terminology finding.
- **Claim reviewed:** visible object types outside COCO-80 are called
  “background.”
- **Exact handle:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:193-196`.
- **Observed issue:** an out-of-ontology object can be real, salient, and
  correctly named while still being unreportable in this closed-set unit.
- **Scientific or contract impact:** closed-set false predictions can be
  mislabeled as hallucinations, and COCO-80 enumeration can be overgeneralized
  to open-world object inventory.
- **Strongest competing explanation:** correct open-world recognition outside
  the evaluation ontology.
- **Smallest correction or discriminator:** replace “background” with
  **out-of-scope visible object** and state that all conclusions concern
  COCO-80 enumeration only.
- **Blocking scope:** generalization language; none for closed-set execution.

### Low (`P3`)

#### P3-1 — Several literal artifact handles are split across lines

- **Category:** research-contract presentation finding.
- **Claim reviewed:** the checkpoint, historical inference, logical run, and
  resolved run locations are exact artifact handles.
- **Exact handles:**
  `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md:238-249`
  and `:490-506`.
- **Observed issue:** literal paths contain Markdown line breaks and therefore
  are not directly copyable as one path.
- **Scientific or contract impact:** a reader can resolve them by joining
  lines, but an automated consumer or operator may treat the newline as part of
  the handle.
- **Strongest competing explanation:** presentation wrapping rather than a
  missing artifact.
- **Smallest correction or discriminator:** render each logical or absolute
  path on one literal line, while keeping historical path tokens unchanged.
- **Blocking scope:** none.

## Hypothesis-by-Hypothesis Audit

The fields below distinguish what the current intervention can identify from
the broader mechanism named by the hypothesis. “Falsifier” means the strongest
result this unit could legitimately use to reject or sharply narrow its own
intervention-level claim; it does not mean a universal mechanism has been
eliminated.

### Spatial-Scope Competition (`HYPOTHESIS_SPATIAL_SCOPE`)

- **Motivation:** dense full-image decoding exposes many plausible objects at
  each row boundary, and a locally restricted view may make a missed valid row
  more likely.
- **Identifiable mechanism:** after P1-2 is corrected, the unit can identify
  whether the masked-input policy improves raw owning-seed detection and final
  post-merge rescue beyond matched sampled full-image policies.
- **Strongest competing explanation:** changed vision-tower activations,
  boundary designation, background suppression, or a mask-specific
  distribution shift.
- **Testable prediction:** `MASK_RESET` has a positive owning-seed raw
  detection difference and a positive post-merge rescue difference over
  `FULL_BAG_K`, while passing retention, manual-precision, duplicate, validity,
  and closure guardrails.
- **Falsifier:** with adequate precision, verified mechanics, and the
  owning-seed comparison, masking has no practically meaningful raw or final
  benefit. This falsifies the chosen masking handle as useful evidence, not all
  possible candidate competition.
- **Remaining confound:** the spatial restriction occurs before visual
  encoding.
- **Allowed conclusion:** structured full-canvas pixel masking helps or fails to
  help this checkpoint, cohort, grid, mask, and decode policy. A decoder-only
  location remains forbidden.

### History-Horizon Instability (`HYPOTHESIS_HISTORY_HORIZON`)

- **Motivation:** longer own-generated prefixes can accumulate errors, repeated
  structure, and list-length priors while later objects remain.
- **Identifiable mechanism:** the current two-arm contrast identifies the total
  effect of adding the specified accepted-row prompt policy. Pure horizon
  instability becomes identifiable only after neutral repetition penalty and
  content/length controls.
- **Strongest competing explanation:** bad prior rows, removal of useful
  coverage information, traversal order, prompt-token repetition penalty, or
  coordinate-frame mismatch.
- **Testable prediction:** under `repetition_penalty=1.0`, reset exceeds
  cumulative prompting in both masked and tile families, the gap grows with
  realized prompt length after controlling cell/object difficulty, and a
  length-matched neutral prefix reproduces a length component.
- **Falsifier:** after adequately precise clean, corrupted, and length-matched
  prefix controls, no practically meaningful degradation follows longer clean
  history. A null only in endogenous history does not falsify the general
  horizon mechanism.
- **Remaining confound:** prefix length, content, and position are difficult to
  vary independently without a dedicated follow-up unit.
- **Allowed conclusion:** the frozen cumulative accepted-row policy is helpful,
  harmful, or equivalent to reset. A pure memory-length claim is forbidden
  without the control panel.

### Tile-Local Computation Explanation (`HYPOTHESIS_TILE_LOCAL_COMPUTATION`)

- **Motivation:** native tiles can reduce visual-token count and re-encode a
  smaller coordinate/positional frame while preserving object pixel scale.
- **Identifiable mechanism:** `TILE_RESET - MASK_RESET` identifies the net
  native-tile protocol bundle.
- **Strongest competing explanation:** fewer visual tokens, local visual
  encoding, positional reset, normalized coordinate distribution, padding,
  removed context, and altered candidate count are all changed together.
- **Testable prediction:** `TILE_RESET` safely exceeds `MASK_RESET` while
  `MASK_RESET` is equivalent to `FULL_BAG_K`.
- **Falsifier:** no current single result falsifies the whole residual family.
  `MASK_RESET` beating bagging shows that tiling is not necessary for all
  spatial benefit, but tile-local effects may still coexist.
- **Remaining confound:** the hypothesis deliberately bundles several
  vision-side and representation-side changes.
- **Allowed conclusion:** native tiling adds or does not add benefit beyond the
  selected masked-canvas policy. Attribution to token count or local vision
  encoding is forbidden.

### Partition and Boundary Context (`HYPOTHESIS_CONTEXT_BOUNDARY`)

- **Motivation:** cores can cut objects or remove context needed for
  classification and localization.
- **Identifiable mechanism:** the current matrix identifies the net effect of
  adding a halo to a masked full canvas, localized by distance to the core
  boundary.
- **Strongest competing explanation:** a halo exposes more objects and changes
  the mask shape/candidate set rather than repairing missing context.
- **Testable prediction:** core-plus-halo masked input improves boundary-near
  owning-seed matches without broad duplicate, precision, or invalidity
  regression.
- **Falsifier:** no boundary-local benefit under adequate sample size weakens
  the chosen masked halo. It does not falsify hard-tile crop effects.
- **Remaining confound:** there is no native-tile core-only arm.
- **Allowed conclusion:** the selected masked-canvas halo is useful, neutral, or
  harmful. A hard-tile boundary conclusion requires the direct tile comparison.

### Geometry-Sorted Traversal Prior (`HYPOTHESIS_GEOMETRY_ORDER`)

- **Motivation:** raster-like region order may align with the checkpoint's
  geometry-sorted output training.
- **Identifiable mechanism:** after counterbalancing, the unit can identify
  order sensitivity of cumulative masked decoding.
- **Strongest competing explanation:** generic prompt position, error
  propagation, spatial adjacency, one unlucky permutation, or merge ordering.
- **Testable prediction:** multiple non-raster orders consistently degrade
  cumulative but not reset arms, with effects following deviation from
  geometry order rather than a particular cell position.
- **Falsifier:** equivalence across multiple counterbalanced orders weakens an
  operational order-sensitivity claim.
- **Remaining confound:** only a comparison to a random-order-trained checkpoint
  can attribute sensitivity specifically to geometry-sorted training.
- **Allowed conclusion:** cumulative behavior is order-sensitive or
  order-insensitive under this checkpoint. A learned-training-prior cause is
  forbidden.

### Dense-Scene Annotation Gaps (`HYPOTHESIS_ANNOTATION_GAPS`)

- **Motivation:** official dense-scene annotations can omit visible COCO-80
  instances, so official false predictions need not be hallucinations.
- **Identifiable mechanism:** a sealed, arm-independent manual ledger can
  estimate the fraction of official unmatched predictions that correspond to
  additional reportable objects on the audited subset.
- **Strongest competing explanation:** true hallucinations, category/box
  mistakes, crowd-ignore loss, proposal-triggered audit bias, or visible
  out-of-ontology objects.
- **Testable prediction:** official precision falls while audit-augmented manual
  precision and unique recall hold or improve, and adjudicated additions are
  distributed across arms rather than concentrated through arm visibility.
- **Falsifier:** blinded adjudication rejects the new predictions or labels them
  ambiguous/out-of-scope at a rate that explains the official precision loss.
- **Remaining confound:** the audited subset and COCO-80 ontology do not
  establish annotation completeness in other datasets or open-world scenes.
- **Allowed conclusion:** a measured proportion of unmatched predictions on the
  frozen audited subset are missing COCO-80 labels. “Most false positives are
  missing annotations” outside that scope is forbidden.

### Residual Recognition or Control-State Limit (`HYPOTHESIS_RESIDUAL_LIMIT`)

- **Motivation:** the same visible objects may remain missed despite sampling,
  spatial masking, tiling, and reset.
- **Identifiable mechanism:** the unit can define a persistent-miss set relative
  to its tested policies.
- **Strongest competing explanation:** parser/class normalization failure,
  insufficient sampling, an ineffective spatial handle, prompt mismatch, or
  object-level audit error rather than recognition/control failure.
- **Testable prediction:** the same adjudicated objects remain unmatched in raw
  and merged outputs across the primary policies.
- **Falsifier:** reliable rescue by one policy removes that object from the
  residual set; it does not falsify residual limits for other objects.
- **Remaining confound:** recognition and conversion of recognition into
  object-specific decoder control are not separated.
- **Allowed conclusion:** the tested interventions did not rescue this named
  object set. “The model cannot recognize these objects” and “control state is
  absent” remain forbidden.

### No-Resize Scale Invariant (`INVARIANT_NO_RESIZE_SCALE`)

- **Motivation:** resized crops could magnify small objects and create a trivial
  recognition advantage.
- **Identifiable mechanism:** none; this is a mechanics condition.
- **Strongest competing explanation if violated:** magnification or geometric
  resampling rather than scope/history.
- **Testable prediction:** processor receipts and coordinate fixtures show
  unchanged object pixel dimensions, with only declared padding and patch-grid
  snapping.
- **Falsifier:** not applicable as a hypothesis. Any silent resize invalidates
  the run.
- **Remaining confound:** unchanged pixel scale does not hold image-token count,
  positional encoding, coordinate frame, padding, context, or visual features
  invariant.
- **Allowed conclusion:** object magnification by resizing did not explain an
  arm difference.

### Generic repeated-sampling explanation

This is a competing explanation represented by `FULL_BAG_K`, not a separately
named hypothesis identifier. It is supported only when sampled bagging safely
improves over both deterministic and sampled single-rollout anchors and is
equivalent to the structured policies under predeclared margins. Equality among
all policies without improvement over `FULL_SINGLE` means all may simply be
ineffective.

## Experimental-Arm Audit

### Full-Image Single Rollout (`FULL_SINGLE`)

- **What changes:** nothing beyond selecting one seed and the unit's sampled
  decode policy.
- **What remains invariant:** full image, base prompt, checkpoint, processor,
  parser, and per-call cap.
- **What it actually estimates:** the one-draw reference outcome that defines
  rescue and retention denominators.
- **What it cannot estimate:** deterministic baseline behavior, sampling-policy
  variability, or full-image capability independent of the selected seed.
- **Necessary:** yes. Also retain the existing deterministic `val200` result as
  a separately named descriptive anchor rather than silently calling the
  sampled draw a reproduction of it.

### Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)

- **What changes:** repeats the full-image call over the complete seed vector
  and merges the resulting objects.
- **What remains invariant:** full image, prompt, checkpoint, image-token count,
  decode settings other than seed, per-call cap, and final merge policy.
- **What it actually estimates:** final detection utility from spending `K`
  sampled calls on the full image.
- **What it cannot estimate:** the causal reason sampling helps, equal compute
  versus cumulative arms, or equal opportunity per object versus spatial
  allocation.
- **Necessary:** essential. Without an executable sampled form, the main unit
  must not run as metric-bearing.

### Native-Scale Tile with Per-Tile Reset (`TILE_RESET`)

- **What changes:** image extent, visual-token count, local visual encoding,
  positional/coordinate frame, padding, visible context, and spatial candidate
  set.
- **What remains invariant:** original object pixel scale, checkpoint, fresh
  prompt semantics, seed binding, decode cap, ontology, and final scoring after
  coordinate mapping.
- **What it actually estimates:** utility of the complete native-tile reset
  protocol.
- **What it cannot estimate:** token-count, local-frame, visual-encoding,
  context, or coordinate effects separately.
- **Necessary:** yes as an important baseline, but it is not the primary
  language-side discriminator.

### Native-Scale Tile with Cumulative History (`TILE_CUMULATIVE`)

- **What changes:** everything changed by `TILE_RESET` plus an expanding
  accepted-row prefix and its realized computation.
- **What remains invariant:** tile plan, checkpoint, per-cell seed identity,
  decode policy, and intended object pixel scale.
- **What it actually estimates:** the cumulative-prefix policy on native tiles,
  once its coordinate and prompt semantics are frozen.
- **What it cannot estimate:** pure history length or a language-only effect;
  tile-local/global coordinate ambiguity is currently fatal to that reading.
- **Necessary:** useful but not necessary for the primary masked history
  contrast. Defer it if the coordinate contract cannot be made unambiguous
  before the primary launch.

### Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)

- **What changes:** pixel content outside one core-plus-halo and all visual
  activations caused by that change.
- **What remains invariant:** canvas dimensions, nominal full-grid visual-token
  count, checkpoint, base prompt, call count, seed multiset, decode cap, and
  final merge relative to `FULL_BAG_K`.
- **What it actually estimates:** utility of structured full-canvas pixel
  masking with fresh prompts.
- **What it cannot estimate:** decoder-only candidate competition, prior
  full-image recognition, or unchanged vision computation.
- **Necessary:** essential primary structured arm.

### Full-Canvas Masked Region with Cumulative History (`MASK_CUMULATIVE`)

- **What changes:** accepted-row prompt history, prompt length, prompt-token
  repetition-penalty exposure, and realized computation relative to
  `MASK_RESET`.
- **What remains invariant:** masked pixel input for a given cell, full-canvas
  dimensions/token count, traversal order, seed-to-cell mapping, checkpoint,
  and per-call cap.
- **What it actually estimates:** total effect of the frozen cumulative-prefix
  policy under masked inputs.
- **What it cannot estimate:** pure history length until P1-3 and P1-4 are
  corrected.
- **Necessary:** essential for the primary prefix-policy comparison.

### Full-Canvas Masked Core-Only Reset (`MASK_RESET_CORE_ONLY`)

- **What changes:** removes halo pixels from the exposed region.
- **What remains invariant:** canvas size, reset prompt, checkpoint, grid,
  seed-to-cell mapping, decode, ownership, and merge.
- **What it actually estimates:** net masked-canvas halo utility, especially by
  distance to core boundary.
- **What it cannot estimate:** native-tile crop/boundary effects.
- **Necessary:** secondary; run only after the primary five-arm mechanics gate.

### Full-Canvas Masked Cumulative Permuted Order (`MASK_CUMULATIVE_PERMUTED`)

- **What changes:** traversal order and therefore the content associated with
  each cumulative-prefix position.
- **What remains invariant:** canonical seed-to-cell mapping, masked input for
  each cell, checkpoint, decode, ownership, and—after correction—canonical
  merge ordering.
- **What it actually estimates:** sensitivity to the selected order.
- **What it cannot estimate:** geometry-sorted training causality from one
  permutation.
- **Necessary:** secondary and not ready in its single-permutation form. Replace
  it with a small counterbalanced order panel or narrow its claim.

## Metric and Matching Audit

### Matching semantics

The core predicate is close to executable: class equality after frozen
normalization, `IoU >= 0.50`, one-to-one maximum-cardinality matching followed
by maximum total `IoU`, invalid rows excluded, duplicates unable to raise
recall, empty conditional denominators reported as not applicable, and a
predeclared `0.75` sensitivity. This correctly avoids many common ambiguities.

Before readiness it still needs:

1. one category registry for official metrics and, if desired, a separately
   versioned research alias map;
2. immutable prediction/reference ordering, numeric tolerance, and exact final
   assignment tie-break;
3. coordinate clipping and zero-area behavior;
4. explicit crowd exclusion/ignore semantics for object-level matching;
5. a synthetic fixture covering a duplicate, equal-IoU assignment, wrong class,
   audit addition, invalid row, crowd region, and empty denominator.

### Local Rescue Rate

Post-merge `LRR` is the right primary metric for final-policy utility, and the
unit correctly uses the same `FULL_SINGLE` miss denominator across paired arms.
It must not be interpreted alone as a mechanism metric. Report, for each
reference object:

- `FULL_SINGLE` raw detection;
- each full-image seed's raw detection;
- the owning spatial call's raw detection;
- raw union detection before merge;
- final detection after merge;
- the exact merge action that changed its state.

This makes repeated opportunities and aggregation visible. Numerators,
denominators, not-applicable cases, and object identities must remain in metric
primitives.

### Retention

Retention is necessary because a policy can rescue baseline misses while losing
objects the baseline already found. It must be a joint gate, not only a
diagnostic. Report object identities lost and gained, not only means, and use a
predeclared noninferiority margin.

### Precision and unique recall

Manual precision and manual unique recall should use the sealed
audit-augmented ledger, one-to-one matching, and post-merge predictions. Unknown
COCO-80 class strings, invalid rows, duplicate extras, and unmatched valid rows
must remain in the precision accounting under explicitly named categories.
Prediction count, precision, and recall must be interpreted jointly.

### Duplicates

Pre-merge and post-merge duplicate diagnostics are necessary. Freeze whether a
duplicate edge requires same normalized class and `IoU` at a named threshold,
whether components are connected components or pair counts, how transitive
chains are treated, and whether ownership filtering happens before graph
construction. Extra duplicates remain false predictions even if the merge
drops them from final-policy output.

### Invalid rows and natural closure

Every image/arm/call attempt must be written before model execution and then
closed with success, parser failure, runtime failure, token-cap stop, natural
`STOP`, or another terminal reason. Invalid rows never match, and failed
attempts stay in image-level denominators. Natural closure must exclude token
cap, controller cap, forced stop, and error. A differential arm failure is an
outcome, not a row that may disappear.

### Official versus audit-augmented ledgers

The unit correctly requires separately named metrics and forbids pooling them.
Preserve three layers:

1. official instances and crowd regions exactly as sourced;
2. sealed audit additions and ambiguity decisions;
3. optional post-run pooled-proposal adjudication, separately named and never
   substituted into the primary ledger.

Official `AP` must use the exact current official evaluator semantics, including
crowd behavior and identity-only category normalization. Research-alias metrics
must not be labeled official.

### Uncertainty estimation

Images, not objects, are the independent sampling units, so image-clustered
resampling is the correct base. Every paired arm outcome from a sampled image
must travel together. The readiness amendment must define the interval
algorithm, stratified sampling target/weights, empty-denominator replicate
behavior, and paired effect recomputation. The primary interval is conditional
on the fixed decode seed vector; seed variability requires a separate
replication design.

## Outcome-to-Belief Matrix

Every row below assumes all mechanics pass and the named superiority,
equivalence, and safety rules were frozen before outputs were inspected.

| Important result pattern | Supported explanation | Weakened explanation | Unresolved alternative | Strongest allowed claim | Next smallest discriminator |
|---|---|---|---|---|---|
| `MASK_RESET` safely exceeds `FULL_BAG_K` and `MASK_RESET` safely exceeds `MASK_CUMULATIVE` | Full-canvas pixel restriction is useful; the specified cumulative-prefix policy is harmful | Generic full-image sampling alone; cumulative history as unconditionally helpful | Mask-induced vision change versus post-vision competition; length versus content versus repetition penalty | Masked-input restriction and reset outperform the two named policies on this scope | Same-feature post-vision restriction, plus clean/corrupt/length-matched prefixes |
| `MASK_RESET` safely exceeds `FULL_BAG_K` and is equivalent to `MASK_CUMULATIVE` | Full-canvas pixel restriction is useful | A practically important cumulative-prefix penalty under this protocol | Vision-side versus post-vision owner | The masked-input policy helps, with no detectable reset/cumulative difference inside the frozen margin | Same-feature post-vision restriction without changing history |
| `MASK_RESET` is equivalent to `FULL_BAG_K` while `TILE_RESET` safely exceeds both | Net native-tile protocol benefit | The selected full-canvas mask as a sufficient scope intervention | Fewer visual tokens, local re-encoding, coordinate/positional frame, padding, or context | Native tiling adds benefit beyond these full-canvas policies | Orthogonally vary token count and coordinate frame while holding encoded features or input content as fixed as possible |
| `FULL_BAG_K`, `MASK_RESET`, and `TILE_RESET` are equivalent and all safely exceed `FULL_SINGLE` | Generic repeated opportunity is sufficient for aggregate final utility | A necessary structured spatial policy | Different arms may rescue different identities or use different raw mechanisms | The three `K`-call policies have equivalent final utility and all improve on one draw | Compare rescued-object overlap and owning-seed raw outcomes |
| `FULL_BAG_K`, `MASK_RESET`, and `TILE_RESET` are equivalent and none safely exceeds `FULL_SINGLE` | No useful rescue from the tested extra-call policies | Claims that generic sampling or structured decomposition explains rescue | Insufficient precision, poor sampling temperature, bad mask/grid, or persistent model limits | No tested policy produced a practically meaningful safe gain | Verify mechanics and precision, then return to object observability and control-state probes |
| Reset safely exceeds cumulative after neutral repetition penalty, while masked scope does not beat bagging | Harm from the specified cumulative-prefix policy | Spatial restriction as necessary | Prefix length, bad content, order, or error propagation | Reset is better than the named cumulative policy; no spatial-policy advantage was detected | Clean, corrupted, shuffled, and length-matched prefix panel |
| Cumulative safely exceeds reset | Useful prior-row or traversal information outweighs harm | Reset as generally preferable | History harm may exist but be offset by useful content | The cumulative policy helps on this protocol | Separate correct coverage information from raw prefix length and errors |
| Any arm raises `LRR` but fails retention, manual precision, duplicate, validity, prediction-count, or closure guardrails | Nonspecific output expansion or unsafe trade-off | Successful enumeration improvement | A useful subpopulation effect hidden by aggregate harm | The arm rescues some misses but fails the declared safety gate | Diagnose first failure onset; do not promote the policy or architecture |
| Official precision falls while sealed-ledger manual precision holds | Missing official COCO-80 labels on the audited subset | Hallucination as the sole explanation | Crowd-ignore loss, out-of-ontology objects, or box/category error | A measured share of official unmatched predictions are valid audited COCO-80 objects | Replicate ledger reliability and crowd-aware official evaluation |
| Official and sealed-ledger manual precision both fall | Unsupported predictions or localization/category errors | Annotation gaps as the main explanation | Audit incompleteness or ontology mismatch | Precision loss persists after audit augmentation | Inspect arm-blind error taxonomy and object-level raw predictions |
| The same adjudicated objects remain missed by every primary arm | Intervention-resistant residual set | Scope/history as sufficient for those objects | Recognition, control synthesis, parser, prompt, or checkpoint limitation | These tested policies did not rescue the named objects | Same-feature target-specific readout and causal-consumption probe |
| Multiple counterbalanced non-raster orders hurt cumulative but not reset arms | Cumulative decoding is order-sensitive | Purely order-insensitive history | Geometry-sorted training prior versus generic positional/error effects | Cumulative performance depends on traversal order | Compare a random-order-trained checkpoint under the same order panel |
| Halo improves only boundary-near masked objects without safety loss | Masked halo context is useful near boundaries | No context dependence for the selected mask | More visible candidates rather than repaired context | The selected masked halo helps boundary-near objects | Native-tile core-only versus halo comparison |

Forbidden across all rows: a positive result does not prove prior full-image
recognition, a pure post-vision defect, a ledger requirement, an object cursor,
slots, calibrated `STOP`, or a final architecture.

## Assumptions Not Actually Tested

The required ten assumptions are classified independently. “Tests fully” below
means the unit's protocol can directly reject or resolve the assumption within
its stated scope; it does not mean a run has already occurred.

| Assumption challenged | Current classification | Audit judgment |
|---|---|---|
| 1. More local predictions imply the full-image model had already recognized those objects | **Leaves unresolved** | Rescue under a changed pixel input shows conditional detectability, not recognition in the original full-image representation. A same-feature or target-specific readout is required. |
| 2. Masking changes only competition within the decoder | **Leaves unresolved empirically; explicitly rejected as a claim** | The unit correctly admits changed vision activations and boundaries, but no arm holds the encoded visual representation fixed. |
| 3. Absence of resizing controls the vision-side representation | **Tests partially** | It controls original object pixel scale. It does not control token count, padding, positional/coordinate frame, context, or visual features. |
| 4. Reset versus cumulative history isolates only language-history length | **Tests partially** | Spatial input can be matched, but content, errors, useful coverage, traversal position, prompt-token repetition penalty, and compute also change. |
| 5. More predictions imply better enumeration | **Tests partially** | Precision, retention, duplicate, invalid, and closure diagnostics are listed, but they are not yet formal joint gates. |
| 6. Most official false positives are missing annotations | **Tests partially** | A sealed audit can estimate this on the audited subset. The current manual contract and crowd semantics are incomplete, and no wider “most” claim is supported. |
| 7. Geometry-sorted output order is merely superficial | **Tests partially** | One permutation can reveal sensitivity, not its training-derived cause. Existing historical evidence also supports order-conditioned behavior, not an order-free ledger. |
| 8. A positive result implies that a ledger or persistent state is required | **Leaves unresolved by design and correctly forbids the inference** | No ledger intervention is tested; every outcome keeps architecture unpromoted. |
| 9. Equal call count implies equal computation | **Tests fully at the protocol level, pending receipts** | The unit explicitly rejects the assumption and requires realized prompt, image, decode, and call budgets. It must also state that per-object opportunities differ. |
| 10. A decodable representation is causally consumed by the decoder | **Leaves unresolved by this unit and correctly forbids the inference** | The governing decision requires target-specific intervention controls; this unit has no representation-consumption experiment. |

Additional assumptions outside the unit include:

- the chosen mask fill is neutral rather than a visual designation cue;
- one fixed checkpoint and `val200`-derived cohort represent other Qwen3
  Vision-Language checkpoints or dense-scene populations;
- a single seed vector represents the stochastic decoding policy;
- the selected grid and halo are adequate handles for all spatial competition;
- an audit box at `IoU >= 0.50` captures every meaningful partial or occluded
  object;
- persistent misses originate inside the model rather than prompt, parser,
  ontology, or evaluator boundaries;
- equal nominal full-grid token count means equal visual computation;
- natural `STOP` reflects complete visible-object coverage;
- the existing geometry-sorted adapter behaves like a random-order-trained
  adapter under order permutations.

## Minimal Required Corrections

These are bounded protocol and readiness corrections, not an architecture
rewrite.

### 1. Before implementation readiness

1. Rename the four mechanism-sounding effect metrics to the behavior-level
   contrast names in the registry, and narrow all result language accordingly.
2. Add the owning-seed per-object contrast while retaining post-merge bagging as
   the final-policy utility comparator.
3. Freeze the full cumulative-history state transition: accepted-row admission,
   raw/canonical text, ordering, coordinate frame, assistant-message
   serialization, context limit, and cache/reset behavior.
4. Make neutral repetition penalty primary for history attribution and declare
   the non-neutral penalty sensitivity.
5. Freeze one object-level merge, category normalization policy, deterministic
   matcher, metric registry, and effect/equivalence/safety rules.
6. Freeze the pre-execution manual-ledger schema, reviewers, blinding,
   partial/ambiguous/crowd rules, adjudication, identifiers, and hash.
7. Join official cohort records to raw crowd provenance.
8. Replace the single order control with a small counterbalanced panel or
   narrow it to one-order sensitivity; add a native-tile core-only arm or narrow
   the boundary hypothesis.

After these changes, conduct a focused scientific and contract re-review before
authorizing code.

### 2. Before execution readiness

1. Explicitly authorize a bounded research implementation; the current
   `implementation_status: not_authorized` must not be bypassed.
2. Materialize immutable cohort identifiers, dense-stratum thresholds, sample
   counts, selection seed, audit ledger, and all content hashes.
3. Implement sampled generation with explicit `do_sample`, temperature,
   nucleus cutoff, per-request seed, and deterministic replay across batching
   and traversal order.
4. Emit one attempt record before each image/arm/call, then finalize it with raw
   prompt, prompt tokens, visual-token count, generated tokens, parser/closure,
   failure, and realized compute.
5. Run a lightweight synthetic two-by-two grid fixture through masks, tiles,
   coordinate round trips, half-open ownership, cumulative prompts, failures,
   object merge, matching, empty denominators, duplicates, crowd handling, and
   metric/equivalence decisions.
6. Resolve an immutable run identifier and retain repository/worktree identity,
   checkpoint and payload hashes, tokenizer/template/processor identities,
   installed package versions, mask/planner/analyzer code identities, authored
   and resolved configs, invariant receipt, budget receipt, raw metric
   primitives, and terminal status.
7. Run a tiny model mechanics gate only after the synthetic fixture passes.
   Verify nonzero sampled diversity, equal nominal full-grid token count,
   unchanged pixel scale, no prompt truncation, coordinate fidelity, unique
   ownership, seed binding, and failure denominators before any cohort run.

### 3. Before interpretation readiness

1. Freeze superiority, equivalence, and safety/noninferiority thresholds before
   viewing model scores.
2. Report raw owning-seed, raw union, and post-merge object states together.
3. Interpret image-clustered intervals as conditional on the fixed seed vector
   and named audited-sample target.
4. Require joint rescue, retention, manual precision, duplicate, validity,
   prediction-count, and closure gates for a positive enumeration claim.
5. Use `Observed`, `Supported`, `Unresolved`, and `Not claimed` explicitly in
   `results.md`; use `Ruled out` only where the intervention and precision
   justify it.
6. Keep official, audit-augmented, research-alias, crowd, and optional
   pooled-proposal results separately named.
7. Keep every architecture status `not_promoted`. A positive masked result
   authorizes a same-feature post-vision discriminator, not a ledger, slot,
   cursor, state module, training objective, or product change.

## Reusable Research-Workflow Recommendations

### Lifecycle and authority

The research-unit lifecycle is structurally sound. The unit correctly:

- separates a planned protocol from future `results.md`;
- marks implementation as unauthorized and evidence as absent;
- uses a `review.md`-style audit record only when a real audit changes the
  readiness gate;
- keeps artifacts under `outputs/research/` and interpretation under
  `research/`;
- keeps architecture unpromoted for every outcome;
- defers stable implementation contracts to OpenSpec.

This matches the semantic ownership and promotion ladder in
`/data/CoordExp/.worktrees/research-probes/.codex/skills/coordexp-research-knowledge-workflow/references/research-graph-contract.md:6-15`
and `:169-186`. Protocol, execution, evidence acceptance, interpretation,
decision update, mechanism promotion, and architecture promotion are therefore
separated adequately. The unresolved problem is completeness of the frozen
protocol, not the lifecycle model.

### Artifact handles and reproduction

The logical and resolved output-root pattern is appropriate, and the named
checkpoint and historical inference locations currently exist. They remain
handles, not proof. No unit run identifier, invariant receipt, execution
receipt, metric primitives, or bounded result exists.

The proposed receipt family is directionally sufficient after adding:

- exact repository/worktree and code-diff identity;
- runtime/library and pseudo-random generator identity;
- per-call attempt and failure rows;
- exact cumulative prompt/token artifacts;
- mask/tile plan and fill/padding hashes;
- object-merge, matcher, bootstrap, and report code identities;
- immutable official and audit ledger hashes;
- explicit evidence state: locally verified, mechanics-only, metric-bearing, or
  unavailable.

### Reusable implementation ownership after design acceptance

Only after the readiness corrections are accepted:

- place a thin unit launcher, readiness/materialization command, and report
  assembly command under `scripts/research/`;
- place deterministic spatial core/halo planning, mask/tile construction,
  coordinate mapping, half-open ownership, cumulative-prefix state transition,
  object merge, matcher, metric primitives, bootstrap/equivalence logic, and
  receipt validation under tested semantic owners in `src/analysis/`;
- keep checkpoint, cohort, grid, mask fill, arm matrix, hypothesis names,
  thresholds, and verdict rules in the research unit and resolved run config,
  not hard-coded in reusable modules;
- require fixtures for boundary centers, padding, coordinate quantization,
  duplicate chains, equal-IoU assignments, invalid rows, crowd regions,
  empty denominators, batching-invariant seeds, and injected partial failures.

Do not create a broad `src/analysis/` package merely to anticipate future work.
Promote logic only when the accepted design needs it and when a second use
justifies reuse.

### OpenSpec decision

An OpenSpec change is **not justified now**. The protocol is still planned,
implementation is explicitly unauthorized, and the unit has not passed
scientific re-review or a mechanics gate. A research-only implementation can be
authorized later without pretending its hypotheses or thresholds are stable
compatibility contracts.

OpenSpec becomes justified only if the accepted implementation changes a
stable reusable surface—for example, canonical sampled-inference config,
backend request/seed semantics, a durable object-level artifact schema, or a
shared metric interface. Scientific cohort choices, mask style, and route
verdicts must remain outside OpenSpec.

### Reusable principles beyond this unit

1. Match both total budget and the opportunity unit relevant to the causal
   claim.
2. A frozen numeric decode knob is not invariant when it acts on different
   prompt content.
3. Separate policy utility from mechanism localization with raw, pre-merge,
   and post-merge evidence.
4. Hold encoded features fixed before claiming a post-vision owner.
5. Treat reset/cumulative contrasts as prefix-policy interventions until length,
   content, correctness, and order are separated.
6. Seal human reference ledgers before arm outputs or keep proposal-driven
   adjudication explicitly secondary.
7. Require safety/noninferiority gates whenever recall or rescue can rise
   through prediction-count inflation.
8. Preserve crowd and ontology provenance before calling official unmatched
   predictions hallucinations.
9. Use negative results to narrow the tested handle, not to erase a broad
   nonexclusive mechanism.
10. Promote architecture only after independent intervention families agree and
    a novel prediction succeeds.

## Final Recommendation

- **Whether to proceed:** proceed only with a bounded protocol revision followed
  by re-review. After approval, authorize a research-only implementation and
  synthetic mechanics gate. Do not execute the metric-bearing cohort in the
  unit's current form.
- **What must be corrected first:** sampled `FULL_BAG_K` support; neutral and
  controlled history semantics; owning-seed opportunity matching; exact
  cumulative prompts; object merge/matching/metric decisions; sealed manual and
  crowd-aware ledgers; and joint effect/safety thresholds.
- **What must not be built yet:** no ledger, slot representation, object cursor,
  persistent state module, new training objective, final architecture, broad
  reusable framework, or stable inference-contract extension is authorized.
- **What evidence would justify the next research unit:** first, a verified
  five-arm primary panel
  (`FULL_SINGLE`, `FULL_BAG_K`, `TILE_RESET`, `MASK_RESET`, and
  `MASK_CUMULATIVE`) with raw owning-seed and post-merge evidence, neutral
  repetition penalty, sealed audit ledger, crowd-aware official metrics, and
  passed safety gates. If masked restriction then survives bagging and safety
  controls, the next unit should test same-feature post-vision restriction. If
  reset then survives repetition-penalty and prefix-content controls, the next
  unit should isolate clean history length from error-bearing history.

**Final verdict: narrow and re-review.**
