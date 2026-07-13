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

Reviewed unit:
`/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md`
(status `planned`, `evidence_status: none`, no execution artifacts exist). This
review is independent, read-only, and non-normative. It reviews the research
design; it does not defend the unit's authors and does not design the final
architecture. All line numbers refer to the files as read on 2026-07-13 in the
`research-probes` worktree.

## Terminology and Name Registry

Severity classes used by this review:

- **Critical (`P0`)** — execution or interpretation would be fundamentally
  invalid if the unit ran as written.
- **High (`P1`)** — must be resolved before the unit becomes `ready` or before
  its results become interpretable.
- **Medium (`P2`)** — an important ambiguity or reproducibility weakness, but
  not necessarily a launch blocker.
- **Low (`P3`)** — a clarity, maintainability, or presentation improvement.

Finding categories: **scientific-design** (validity of inference),
**research-contract** (compliance with the research-graph and unit lifecycle
contract), **implementation-readiness** (what must exist before code or
execution), **terminology** (naming completeness and clarity).

Model, data, and objective names:

- **Qwen3-VL — Qwen3 Vision-Language**: the pretrained multimodal model family
  under investigation; the frozen checkpoint is an approximately
  two-billion-parameter variant.
- **COCO-80 — Common Objects in Context 80-category ontology**: the closed set
  of reportable object categories for the unit.
- **val200 — Validation-200**: the fixed 200-image validation cohort behind the
  existing inference artifacts.
- **DoRA — Weight-Decomposed Low-Rank Adaptation**: the adapter method used by
  the frozen checkpoint.
- **CE — cross-entropy**: the ordinary next-token training objective.
- **Gaussian Soft-Target Coordinate Cross-Entropy with Ordered
  Cumulative-Distribution Penalty**: the coordinate auxiliary objective used by
  the primary checkpoint — Gaussian soft-target cross-entropy over ordered
  coordinate bins plus a squared cumulative-distribution discrepancy. This is
  the canonical plain-language name and deliberately has no abbreviation.
- **`gaussian_rps`**: a historical artifact-path and configuration token for the
  objective above. It is a provenance string only. This review confirms the
  unit quarantines it correctly and does not promote it into a research term;
  this review likewise uses it only when quoting paths.
- **GT — ground truth**: reference object annotations. Used here only when
  citing the historical "painted ground-truth" probe lineage.
- **`geo_sorted` — geometry-sorted object order**: the validated authored
  top-to-bottom then left-to-right row order used by the checkpoint's training
  data (`docs/SYSTEM_OVERVIEW.md:49`).
- **norm1000 coordinates**: bounding-box coordinates expressed as discrete
  tokens over one thousand bins normalized to the model's input canvas
  (`docs/COORDEXP_SWIFT.md:96`). Consequence used repeatedly below: a tile
  input defines its own coordinate frame.
- **HF `generate` — Hugging Face Transformers generation path**: the decoding
  backend recorded by the prior val200 artifact (`"backend": "hf"` in its
  `run_manifest.json`).

Decode and metric terms:

- **`rp` — repetition penalty**: decode-time penalty applied to the logits of
  token identifiers already present in the sequence; in the standard Hugging
  Face implementation this is a constant once-seen penalty over the full
  sequence, including prompt tokens, not a per-occurrence compounding penalty.
- **`bbox` — bounding box**: the four-coordinate rectangle emitted per object.
- **IoU — bounding-box Intersection over Union**: intersection area divided by
  union area between a predicted and a reference bounding box.
- **LRR — Local Rescue Rate**: the empirical conditional probability that an
  arm detects a reference object missed by the paired Full-Image Single
  Rollout.
- **Retention**: the empirical conditional probability that an arm detects an
  object the paired Full-Image Single Rollout also detected.
- **AP — Average Precision**: the standard precision-recall detection metric;
  secondary in this unit.
- **CI — confidence interval**; the unit's uncertainty tool is the
  **Image-Clustered Bootstrap Confidence Interval**, which resamples images
  while keeping all objects and arm results of an image together.
- **`K`**: the number of spatial cells in the selected grid (sixteen for the
  primary four-by-four grid), and therefore the matched invocation count.
- **STOP — terminal no-more-objects decision**: the model action that ends
  enumeration.
- **`iscrowd` — crowd flag**: the Common Objects in Context annotation flag
  marking a region that stands for many unindividuated instances.
- **out-of-distribution input**: an input whose statistics differ from anything
  the frozen checkpoint saw during training (for example, a canvas that is
  mostly a constant channel-mean fill).

Experimental arm codes (all from the unit's registry; complete names repeated
here because this review uses the codes):

- **`FULL_SINGLE` — Full-Image Single Rollout**: one ordinary rollout over the
  complete image, using the first seed of the frozen seed vector.
- **`FULL_BAG_K` — Full-Image K-Rollout Independent Bagging**: `K` independent
  full-image sampled rollouts merged by the frozen merge policy.
- **`TILE_RESET` — Native-Scale Tile with Per-Tile Reset**: unresized
  core-plus-halo tiles, each decoded from a fresh prompt.
- **`TILE_CUMULATIVE` — Native-Scale Tile with Cumulative History**: the same
  tiles decoded while retaining accepted rows from earlier tiles.
- **`MASK_RESET` — Full-Canvas Masked Region with Per-Region Reset**: full-size
  canvases with one visible core-plus-halo region, each decoded from a fresh
  prompt.
- **`MASK_CUMULATIVE` — Full-Canvas Masked Region with Cumulative History**:
  the same masked canvases decoded while retaining accepted earlier rows.
- **`MASK_RESET_CORE_ONLY` — Full-Canvas Masked Core-Only Reset**: the reset
  masked arm without halo context.
- **`MASK_CUMULATIVE_PERMUTED` — Full-Canvas Masked Cumulative Permuted
  Order**: the cumulative masked arm under a frozen non-raster cell order.

Hypothesis and invariant codes (declared in the unit; operational meanings
restated):

- **`HYPOTHESIS_SPATIAL_SCOPE` — Spatial-Scope Competition**: simultaneous
  object candidates across the full visual field suppress locally valid
  continuation paths.
- **`HYPOTHESIS_HISTORY_HORIZON` — History-Horizon Instability**: a longer
  autoregressive text prefix degrades later object retrieval or continuation.
- **`HYPOTHESIS_TILE_LOCAL_COMPUTATION` — Tile-Local Computation Explanation**:
  tile gains come from fewer image tokens or locally re-encoded visual frames;
  deliberately a residual family in this unit.
- **`HYPOTHESIS_CONTEXT_BOUNDARY` — Partition and Boundary Context**: object
  cuts or lost surrounding context explain core-only versus core-plus-halo
  differences.
- **`HYPOTHESIS_GEOMETRY_ORDER` — Geometry-Sorted Traversal Prior**: raster
  cell order helps because it agrees with geometry-sorted training.
- **`HYPOTHESIS_ANNOTATION_GAPS` — Dense-Scene Annotation Gaps**: new unmatched
  predictions are real but unlabeled reportable objects.
- **`HYPOTHESIS_RESIDUAL_LIMIT` — Residual Recognition or Control-State
  Limit**: objects missed by every condition remain outside reliable
  recognition or object-specific decoder control.
- **`INVARIANT_NO_RESIZE_SCALE` — No-Resize Scale Invariant**: every input
  preserves original object pixel scale; violation invalidates the unit rather
  than informing any hypothesis.

Estimand names (from the unit): **Local Rescue Rate for arm A**, **Retention
for arm A**, **Spatial-Scope Effect** (`LRR(MASK_RESET) − LRR(FULL_BAG_K)`),
**Masked History Effect** (`LRR(MASK_RESET) − LRR(MASK_CUMULATIVE)`), **Tile
History Effect** (`LRR(TILE_RESET) − LRR(TILE_CUMULATIVE)`), **Local-Frame
Effect** (`LRR(TILE_RESET) − LRR(MASK_RESET)`).

Historical path token noted in passing: `pvci` appears in historical
painted-ground-truth experiment paths; those records do not declare its
expansion in the files read for this audit. It is treated here purely as a
path token, never as a term.

## Executive Verdict

- **Is the unit scientifically valuable?** Yes. It attacks the single most
  common confound in tiled-inference folklore — that "tiling helps" bundles
  spatial restriction, history reset, repeated sampling, token-count change,
  and scale change — and it adds the two controls most published tiling
  comparisons omit: a matched-call full-image bagging arm and an
  equal-token-budget masked-canvas arm. The claim boundaries are honest and
  consistent with the standing decisions (`let-architecture-emerge-from-hypothesis-gates`,
  `require-target-specific-causal-consumption`).
- **Is it currently ready for implementation?** No. Five High findings must be
  folded into the protocol first; three of them (crowd-region matching
  semantics, cumulative-history injection contract, merge-policy co-primary)
  change what the metric and runner code must do.
- **Is it currently ready for execution?** No, by its own declaration
  (unit.md:259, unit.md:276-277: `planned` until the cohort/decode readiness
  amendment exists) and by the corrections below (bagging temperature
  principle, repetition-penalty scope receipt).
- **Can its primary comparisons distinguish the intended mechanisms?** Yes at
  the *family* level, after the named corrections: `MASK_RESET` versus
  `FULL_BAG_K` separates structured spatial restriction from generic
  resampling at matched call and token budget, and `MASK_RESET` versus
  `MASK_CUMULATIVE` separates history retention from history reset at matched
  spatial input. They cannot — and the unit correctly does not claim to —
  localize the effect to the vision tower versus the language tower, or
  separate sub-mechanisms inside each family (candidate-path suppression
  versus output-length rationing; prefix length versus prefix content). Two
  decode-mechanics couplings (repetition-penalty scope, merge asymmetry)
  currently threaten even the family-level attribution and are the core of the
  P1 list.
- **Single largest remaining uncertainty:** whether `FULL_BAG_K` will be a
  genuinely strong, fairly-scored control. The headline Spatial-Scope Effect
  is only as meaningful as this control; today its strength hangs on an
  unprincipled temperature choice, an unspecified merge policy that does
  asymmetric work across arms, and an undeclared repetition-penalty scope.

**Verdict: approve after named corrections.**

## Severity-Ranked Findings

### Critical (`P0`)

None. No element of the design would make execution or interpretation
*fundamentally* invalid as pre-registered: the unit is still `planned`, its own
readiness gates block execution before the missing pieces are frozen, and the
issues below are all repairable with bounded edits. Residual risk at this
level: if the unit were executed today by an agent that read only the
eight-arm matrix and the decode list, the P1 items would silently become
interpretation-invalidating; the protection is procedural (the readiness
amendment), not intrinsic.

### High (`P1`)

**P1-1 — Repetition-penalty scope mechanically couples with both primary
contrasts.** [scientific-design]

- File/lines:
  `/data/.../unit.md:208-211` (repetition penalty "frozen for all arms",
  causal effect declared unidentifiable), `unit.md:331-339` (the frozen-factor
  list names the repetition-penalty *value* but not its *scope semantics*),
  `unit.md:44-45` (registry example `rp=1.10`).
- Observed issue: under the recorded decoding backend (HF `generate`; see the
  prior val200 `run_manifest.json`), the standard repetition-penalty processor
  penalizes every token identifier present anywhere in the current sequence —
  including prompt tokens — as a constant once-seen penalty. Two consequences
  the unit does not name: (a) in `MASK_CUMULATIVE` / `TILE_CUMULATIVE`, the
  injected accepted rows pre-penalize wrapper tokens
  (`<|object_ref_start|>`, `<|box_start|>`, …) and every already-used norm1000
  coordinate bin *at call start*, while reset arms begin with a clean seen-set;
  (b) within any single call, once-seen coordinate-bin penalties accumulate
  over the emitted list, so long-output arms (`FULL_SINGLE`, each `FULL_BAG_K`
  rollout) carry them across the whole image while `MASK_RESET` clears the
  seen-set every call. In dense scenes with axis-aligned repeated structure
  (tables, crowds — exactly the target stratum), repeated coordinate bins are
  common, so this is not a negligible pressure.
- Impact: a positive Masked History Effect is confounded with
  "penalized-prompt mechanics", and a positive Spatial-Scope Effect is
  confounded with "per-call seen-set reset mechanics". In the worst pattern —
  `MASK_RESET` above both `MASK_CUMULATIVE` and `FULL_BAG_K` — the entire
  headline panel is reproducible by decode mechanics with no model-internal
  competition or history mechanism at all. The unit's caveat at
  unit.md:208-211 acknowledges rp is unidentified as a *factor* but not that
  it is *differentially loaded onto the primary contrasts*.
- Strongest competing explanation (that this finding protects against):
  "spatial scope and history effects" that are actually artifacts of the
  repetition-penalty seen-set boundary.
- Smallest correction: in the readiness amendment (unit.md:329-339): (1)
  record the installed penalty-scope semantics as a receipt (does the penalty
  see prompt/injected-history tokens?); (2) either set the unit's frozen
  repetition penalty to `1.0` for all arms, or predeclare a paired
  `rp ∈ {1.0, 1.10}` sensitivity panel for `FULL_BAG_K`, `MASK_RESET`, and
  `MASK_CUMULATIVE` on the audited subset with `1.0` primary for mechanism
  attribution. Note this does not contradict unit.md:208-211, which already
  points a later study back to `1.0`; it moves the minimum of that control
  into this unit because the contrast — not just the factor — is loaded.
- Blocks: interpretation (of both primary contrasts); execution readiness for
  the decode amendment.

**P1-2 — The frozen merge policy does asymmetric work across arms, and the
post-merge primary rescue metric makes the headline sensitive to a nuisance
parameter.** [scientific-design]

- File/lines: `unit.md:354-356` (primary rescue/retention use post-merge
  predictions), `unit.md:74-76` (merge policy declared only as a placeholder),
  `unit.md:64-67` and `unit.md:314` (`FULL_BAG_K` applies "the same merge
  policy used by the spatial arms"), `unit.md:294-295` (unique core
  ownership).
- Observed issue: "same merge policy" is nominal, not operational, symmetry.
  Spatial arms deduplicate primarily by *ownership* (each location has exactly
  one owning call), so cross-call duplicates are structurally impossible for
  them and the merge step has little to do. `FULL_BAG_K`'s unique recall, by
  contrast, is *manufactured entirely inside the merge step*: every real
  object arrives up to `K` times with sampling jitter, adjacent to same-class
  neighbors. Any IoU-threshold merge in a dense same-class scene must then
  choose between leaving duplicate residue (precision cost only, harmless to
  recall under one-to-one matching) and chain-merging distinct adjacent
  instances (which deletes true bag rescues). The second failure mode acts
  asymmetrically: it deflates `FULL_BAG_K` unique recall in exactly the
  people-heavy and tableware-heavy strata the unit targets, inflating the
  headline Spatial-Scope Effect.
- Impact: the primary contrast becomes a partial function of an arbitrary
  frozen nuisance parameter (the duplicate-merge threshold and representative
  selection rule), in a direction that favors the unit's lead hypothesis.
  Freezing the policy before results prevents fishing but does not remove the
  structural asymmetry.
- Strongest competing explanation: an observed positive Spatial-Scope Effect
  that is actually merge-collapse suppression of bag rescues.
- Smallest correction: (1) add pre-merge Local Rescue Rate under the same
  one-to-one matcher as a predeclared co-primary or mandatory sensitivity view
  (one-to-one matching already prevents duplicate inflation, so pre-merge
  rescue is well-defined; the unit already retains pre-merge rows,
  unit.md:298-300, 354-356); (2) add a mechanics counter for merge events that
  fuse two predictions matched to *different* reference objects, reported per
  arm. Interpretation of the headline requires the post-merge and pre-merge
  views to agree in sign.
- Blocks: interpretation (primary contrast); implementation readiness (the
  metric module must emit both views).

**P1-3 — Bagging strength is left to an unprincipled temperature choice; a
weak bag is a straw-man control that manufactures the headline result.**
[scientific-design, implementation-readiness]

- File/lines: `unit.md:64-67` (nonzero temperature required),
  `unit.md:341-347` (predeclared nonzero temperature; deterministic-policy
  renaming rule), `unit.md:446-447` (gate 10: count unique raw rollout
  strings).
- Observed issue: the unit correctly rules out degenerate deterministic
  "bagging", but the only diversity evidence required is the count of unique
  raw rollout strings — strings can differ by one sampled token while the
  prediction sets are effectively identical. No principle constrains the
  temperature choice, and no gate requires the bag to be a *live* control. A
  low predeclared temperature makes `FULL_BAG_K ≈ FULL_SINGLE`, from which
  `MASK_RESET > FULL_BAG_K` follows almost automatically.
- Impact: the central discriminator between `HYPOTHESIS_SPATIAL_SCOPE` and
  generic resampling can be silently weakened at design time, without any
  post-hoc misconduct, by one number nobody justified.
- Strongest competing explanation: a "spatial-scope win" that is actually
  "resampling was never given a fair diversity budget".
- Smallest correction: in the readiness amendment: (1) predeclare the
  temperature and nucleus cutoff with a one-paragraph justification; (2) add a
  bag-diversity mechanics gate at the prediction-set level (for example, the
  distribution over images of unique post-merge objects contributed by
  rollouts beyond the first, or mean pairwise prediction-set overlap between
  rollouts), with a predeclared floor below which the arm is renamed a
  weak-diversity control exactly as unit.md:344-347 already does for the
  zero-temperature case; (3) optionally predeclare a two-point temperature
  sensitivity for `FULL_BAG_K` and `MASK_RESET` only.
- Blocks: execution readiness; interpretation of the primary contrast.

**P1-4 — `TILE_CUMULATIVE` history has no defined coordinate frame, and no
choice of frame is semantically coherent.** [scientific-design]

- File/lines: `unit.md:62-64` (cumulative history definition),
  `unit.md:106-107` and `unit.md:316` (arm definition/matrix row);
  frame fact: `docs/COORDEXP_SWIFT.md:96` (norm1000 coordinates are normalized
  to the input canvas, so each tile is its own frame).
- Observed issue: accepted rows from earlier tiles were emitted in *those
  tiles'* norm1000 frames. Appending them verbatim to the next tile's prompt
  mixes incompatible frames; remapping them to the global canvas frame gives
  coordinates that are wrong in the current tile's frame; remapping into the
  current tile's frame puts most prior boxes outside the representable
  coordinate range. Every option injects geometry the model must misread. The
  masked arms do not have this problem (one shared canvas frame throughout),
  which is an additional, unstated reason the masked pair is the right primary
  history contrast.
- Impact: the Tile History Effect (unit.md:398-399) measures
  "frame-incoherent history confusion", not history horizon; any agreement or
  disagreement between Tile History Effect and Masked History Effect is
  uninterpretable as generalization evidence.
- Strongest competing explanation: an apparent tile-history penalty caused
  purely by frame-corrupted prompt geometry.
- Smallest correction: either (a) define the frame policy explicitly in the
  unit and demote `TILE_CUMULATIVE` to a mechanics/exploratory arm whose
  estimand is dropped from the primary list, or (b) drop the arm. Option (a)
  preserves the matrix at trivial cost; the masked pair already carries the
  history question coherently.
- Blocks: interpretation (of Tile History Effect only); none of the headline
  panel.

**P1-5 — The object-matching predicate is not yet fully executable: crowd
regions, ambiguous audit objects, and out-of-ontology predictions have no
defined semantics.** [scientific-design, research-contract]

- File/lines: `unit.md:358-373` (matching predicate), `unit.md:263-266`
  (crowd/overlap summaries recorded at cohort construction, then never used in
  the metric contract), `unit.md:194-196` (non-COCO-80 objects "are
  background, even if Qwen can name them"), `unit.md:269-272` ("manual review
  may tag annotation ambiguity" with no downstream semantics).
- Observed issue: three inputs that dense Common Objects in Context scenes are
  guaranteed to produce have no metric treatment: (a) `iscrowd` reference
  regions — are they recall denominators, ignore regions, or false-positive
  sources for predictions that overlap them?; (b) audit-tagged ambiguous or
  partially visible objects — do they enter numerators, denominators, neither?
  (c) syntactically valid predictions whose normalized class is outside
  COCO-80 — false positives, or excluded as background? The people-heavy
  stratum makes (a) first-order: standard Common Objects in Context practice
  treats crowd regions as ignore regions, and choosing anything else can swing
  precision and rescue substantially. (c) is arm-asymmetric in expectation:
  sparse masked canvases plausibly shift the model's reporting prior toward
  non-COCO objects.
- Impact: the answer to "is the matching predicate fully executable?" is
  currently *no*; two implementers would produce different primary numbers
  from identical raw rows.
- Strongest competing explanation (if unfixed): arm differences in the
  people-heavy stratum driven by crowd-handling and ontology-filter choices
  rather than by any hypothesis.
- Smallest correction: add three sentences to the matching contract:
  (1) `iscrowd` reference regions are ignore regions — excluded from all
  rescue/retention/recall denominators; unmatched predictions overlapping a
  crowd region at or above a frozen threshold are neither true nor false
  positives, unless matched to an individuated audit addition;
  (2) the audit ledger schema carries per-object flags
  `{original, audit-added, ambiguous→ignore, crowd-individuated, truncated}`,
  and `ignore`-flagged objects join neither numerator nor denominator;
  (3) valid rows with out-of-ontology normalized classes are counted as false
  predictions for precision and excluded from matching (or the explicit
  alternative), with their rate reported per arm.
- Blocks: implementation readiness (metric code) and execution readiness (the
  gate-11 fixture should also cover a crowd case and an ignore case).

### Medium (`P2`)

**P2-1 — The reset-versus-cumulative contrast tests clean injected history,
not the "error-bearing" own prefix the hypothesis names; the falsifier is
over-broad.** [scientific-design]

- File/lines: `unit.md:186` (falsifier: "Cumulative matches or exceeds reset
  after spatial scope is fixed"), `overview.md:67-72` ("long and error-bearing
  autoregressive prefix", "own-prefix errors").
- Issue: cumulative history is built from *accepted* rows — parser-cleaned,
  teacher-injected text. A null result therefore falsifies only the
  clean-history-length component; own-prefix-error mechanisms (documented in
  the painted-ground-truth lineage's own-prefix panel) remain untouched.
- Impact: `HYPOTHESIS_HISTORY_HORIZON` could be declared "ruled out" while its
  most plausible sub-mechanism was never manipulated.
- Smallest correction: narrow the hypothesis's falsifier wording to the
  clean-history horizon, and name noisy/self-generated-history injection as a
  later discriminator.
- Blocks: interpretation wording only.

**P2-2 — Cumulative masked history refers to content that is currently masked
out; the history contrast bundles an input-consistency novelty.**
[scientific-design]

- File/lines: `unit.md:62-64`, `unit.md:110-111`.
- Issue: at train time, prefix rows always describe visible content. In
  `MASK_CUMULATIVE`, every injected row points at a gray region. Reset versus
  cumulative therefore contrasts "no history" against "history referencing
  invisible content", not "short versus long history" alone. Partially
  mitigated by the planned slope analyses versus realized prompt length
  (unit.md:425).
- Smallest correction: name this as a bundled confound inside
  `HYPOTHESIS_HISTORY_HORIZON`; optionally use halo-visible prior objects (the
  only coherent subset) as a diagnostic stratum.
- Blocks: interpretation wording only.

**P2-3 — "Accepted rows" and the history-injection contract are undeclared.**
[research-contract, implementation-readiness]

- File/lines: `unit.md:62-64` (uses "accepted rows" without an acceptance
  predicate), `unit.md:331-339` (frozen-factor list omits the history
  injection template and acceptance rule).
- Issue: acceptance could mean parser-valid rows, ownership-filtered rows, or
  merged rows; injection could be assistant-prefix continuation or a user-turn
  listing. These choices change semantics materially (ownership-filtered
  acceptance means a halo-detected, disowned object is *not* in history and
  will be re-attempted by its owner; injection role changes both the learned
  prior and the repetition-penalty exposure of P1-1).
- Smallest correction: add "cumulative-history acceptance predicate and
  injection template (role, ordering, concatenation)" to the resolve-and-hash
  list at unit.md:331-339.
- Blocks: implementation readiness.

**P2-4 — Cohort-planning wording permits leakage of prior model behavior into
image selection.** [research-contract]

- File/lines: `unit.md:246-249` (existing rollout "used for cohort planning
  and baseline sanity only") versus `unit.md:263-266` (dense stratum "never
  from arm scores").
- Issue: "cohort planning" with the existing rollout could include ranking or
  picking images by where step-4887 missed objects — selection on a correlated
  realization of the very conditioning event (`FULL_SINGLE` misses) that
  defines the rescue denominators. Paired contrasts survive such selection,
  but the declared population ("dense by annotation and metadata") would be
  silently replaced by "dense and known-hard for this checkpoint".
- Smallest correction: one sentence: the existing rollout may size and
  sanity-check the cohort but must not rank, select, or exclude images;
  selection features come only from annotations and image metadata.
- Blocks: execution readiness (cohort amendment wording).

**P2-5 — The audit ledger's freeze boundary versus the post-hoc blinded
adjudication is ambiguous.** [research-contract]

- File/lines: `unit.md:273-275` (exhaustive audit ledger materialized during
  cohort construction), `unit.md:443` (gate 8: blinded manual audit "for new
  unmatched predictions" — necessarily post-execution), `unit.md:87-91`
  (Audit-Augmented Local Rescue Rate uses "the frozen dense audit ledger").
- Issue: if post-hoc adjudication of unmatched predictions can *add* objects
  to the ledger, the freeze is broken and additions are prediction-guided —
  arms emitting more boxes seed more reference objects (a real asymmetry). If
  it cannot, the unit should say what gate 8 feeds (presumably
  `HYPOTHESIS_ANNOTATION_GAPS` reporting and manual precision) and confirm the
  audit-augmented denominator is fixed before any arm output is seen.
- Smallest correction: declare: the audit-augmented ledger is frozen
  pre-execution; blinded post-hoc adjudication feeds only annotation-gap
  reporting; if ledger escapes are discovered, publish a separately named
  amended ledger and report both.
- Blocks: interpretation (audit-augmented metrics).

**P2-6 — No minimum detectable effect or audited-subset size target.**
[scientific-design]

- File/lines: `unit.md:276-277` (counts deferred to the amendment, no power
  language), `unit.md:377-379` (bootstrap defined, no width target).
- Issue: the headline is a difference of *conditional* probabilities over
  missed objects on a manually audited subset; with a small subset the paired
  confidence intervals may span every interesting effect size, making
  "approximately equals" outcomes in the outcome map (unit.md:465-475)
  undefined in practice.
- Smallest correction: predeclare a minimum paired missed-object count derived
  from the existing rollout's miss rate, plus a target confidence-interval
  half-width that operationalizes "approximately equals".
- Blocks: execution readiness (cohort amendment).

**P2-7 — The primary ledger for the headline panel is not named.**
[research-contract]

- File/lines: `unit.md:409-414` (headline is the audited-subset panel; both
  ledgers are emitted; no primary named).
- Issue: two headline numbers per contrast (official versus audit-augmented)
  invite post-hoc selection.
- Smallest correction: one sentence naming the Audit-Augmented Local Rescue
  Rate as primary on the audited subset, with the official-annotation view as
  the labeled secondary.
- Blocks: interpretation.

**P2-8 — The per-call seed derivation is unspecified and the
`FULL_SINGLE`-in-bag identity is assumed but never receipted.**
[implementation-readiness]

- File/lines: `unit.md:68-71` (seed vector indexed by cell identity — per
  image or shared across images?), `unit.md:341-343` (`FULL_SINGLE` uses the
  first seed of the same vector).
- Issue: (a) whether the effective seed is `f(base, cell)` or
  `f(base, image, cell)` changes cross-image correlation of sampling noise;
  (b) the design implies bag rollout 1 replays `FULL_SINGLE` exactly, which
  makes bag Retention ≈ 1 by construction — but kernel nondeterminism can
  break seed-replay identity silently.
- Smallest correction: declare the seed derivation function in the amendment,
  and add a mechanics receipt comparing bag-call-1's token stream to
  `FULL_SINGLE` per image (match or documented divergence).
- Blocks: execution readiness.

**P2-9 — A null masked result is ambiguous unless mask harm is checked first;
the outcome map interprets it one-sidedly.** [scientific-design]

- File/lines: `unit.md:471` (outcome row: `MASK_RESET ≈ FULL_BAG_K` while
  `TILE_RESET > FULL_BAG_K` routes to token-budget/local-encoding
  explanations), `unit.md:453-454` (falsification signature reads equality as
  "generic sampling, not spatial scope"), `unit.md:302-304` (mask-artifact
  clause covers only visible-artifact extremes).
- Issue: the channel-mean fill is an out-of-distribution canvas; masking can
  *suppress recognition of visible objects* (context removal, boundary
  effects) by roughly the amount scope restriction helps, producing a null
  that says nothing about spatial scope. The design already contains the
  diagnostic — Retention of `MASK_RESET` on core-interior `FULL_SINGLE` hits —
  but no rule requires consulting it before accepting the null.
- Smallest correction: add one predeclared rule: a null or negative
  Spatial-Scope Effect is interpretable against `HYPOTHESIS_SPATIAL_SCOPE`
  only if `MASK_RESET` Retention on objects wholly inside cores is high (a
  frozen floor); otherwise the outcome is "masking harms recognition —
  scope question unresolved".
- Blocks: interpretation (null branches of the outcome map).

### Low (`P3`)

**P3-1 — Gate 3 names three equal-token arms; the arm matrix requires five.**
[research-contract] `unit.md:436-437` versus `unit.md:322-325`. Harmonize the
gate to include `MASK_RESET_CORE_ONLY` and `MASK_CUMULATIVE_PERMUTED`. Blocks:
none.

**P3-2 — Wrapped multi-line artifact paths are not copy-executable handles.**
[terminology, research-contract] `unit.md:238-242`, `unit.md:246-249`,
`unit.md:494-497`, `unit.md:503-506` embed literal newlines/indentation inside
`text` fences. State the join rule or provide single-line forms in the
amendment. Blocks: none.

**P3-3 — "Generic sampling, not spatial scope" overstates the negative.**
[scientific-design] `unit.md:453-454`. Equal rescue at matched call budget
leaves per-call scope effects open (each masked call concentrates its budget
on one cell; the bag gets `K` whole-image chances per object). The safe
behavior-level wording is "no evidence that spatial restriction adds rescue
beyond matched-call resampling at equal budget" — which is exactly what the
routing decision needs. Blocks: none.

**P3-4 — Retention is not comparable across bag and spatial arms.**
[scientific-design] `unit.md:185` (prediction: structured arms rescue "while
retaining `FULL_SINGLE` true positives"), `unit.md:341-343`. Because the bag
contains `FULL_SINGLE`'s seed, bag Retention ≈ 1 by construction (given seed
replay), and the bag has `K−1` informative fresh rollouts versus `K` for
spatial arms. Label Retention an arm-specific safety metric, not a comparative
mechanism signal. Blocks: none.

**P3-5 — The unit's optimal bipartite matcher diverges from the existing
frozen greedy matcher used by all prior evidence.** [implementation-readiness]
`unit.md:363-365` (maximize count, then total IoU) versus
`src/vis/matching.py:88-92` (greedy, match IoU `0.50`, duplicate IoU `0.30`).
Legitimate, but new code is required (the gate-11 fixture is the right guard),
and recall/precision values are not commensurable with prior
painted-ground-truth tables — say so once. Blocks: none.

**P3-6 — Precision's treatment of invalid rows is unstated.**
[scientific-design] `unit.md:368-369` excludes malformed rows from matching;
`unit.md:422` reports invalid rate separately; whether precision's denominator
is valid-rows-only is left implicit. One sentence fixes it. Blocks: none.

**P3-7 — No multiplicity language across the estimand grid.**
[scientific-design] `unit.md:393-427`: six estimands × two ledgers × two IoU
thresholds × several strata. The declared single headline (unit.md:411-414)
mitigates; add one sentence that all non-headline views are descriptive.
Blocks: none.

**P3-8 — The eight-by-eight conditional gate is fuzzy and the halo snap rule
can silently produce core-only tiles.** [implementation-readiness]
`unit.md:281-283` ("retaining enough visual support" undefined),
`unit.md:291-293` (halo `25%` snapped to the patch/merged-token grid — for an
eight-by-eight grid on a typical val200 image the nominal halo is below one
merged-token cell, and the snap direction is unstated, so it may snap to
zero). Define the numeric support gate and the snap direction (round up, with
a minimum one-cell halo). Blocks: none (conditional arm only).

**P3-9 — No planned invocation/cost table.** [research-contract]
`unit.md:306-320` implies roughly `1 + 7K` calls per image per grid plus the
audited-subset arms; the unit-body contract requires a budget
(`research-graph-contract.md:127`). Budget *receipts* are specified
(unit.md:444-445) but the planned totals are not. Add a small planned-call
table to the amendment. Blocks: none.

**Terminology findings: none material.** The registry (unit.md:26-144) is the
most complete in the research tree: every arm, hypothesis, metric, and legacy
path token is declared; `gaussian_rps` is explicitly quarantined as a
historical provenance token (unit.md:224-226, 236) and mapped to the canonical
plain-language name **Gaussian Soft-Target Coordinate Cross-Entropy with
Ordered Cumulative-Distribution Penalty** (unit.md:36-41), which is used for
all scientific statements (unit.md:253-257). It is not promoted into a
research term anywhere in the unit or overview. Residual terminology risk:
"accepted rows" and "merge policy" are declared names whose *operational*
content is still deferred (covered by P2-3 and P1-2), and the historical path
token `pvci` remains undeclared in the historical records it appears in
(out of scope for this unit).

## Hypothesis-by-Hypothesis Audit

### `HYPOTHESIS_SPATIAL_SCOPE` — Spatial-Scope Competition (unit.md:120-122, 185; overview.md:59-65)

- Motivation: dense full-image rollouts under-enumerate while per-object
  capability is strong; competition among simultaneous candidates (and STOP)
  is a plausible proximate cause.
- Identifiable mechanism, as instrumented: *restricting visible spatial
  evidence at fixed image-token budget and matched call budget adds unique
  recall beyond resampling.* That is a genuine causal contrast of procedures.
- Strongest competing explanations: (a) matched-call resampling suffices —
  controlled by `FULL_BAG_K` (the unit's own control, and the reason this
  design is better than tiling folklore); (b) repetition-penalty seen-set
  reset (P1-1) — currently uncontrolled; (c) merge asymmetry (P1-2) —
  currently uncontrolled; (d) *output-length rationing under a learned
  list-length prior*: each response spends a bounded row budget on its most
  salient candidates; masking forces the budget onto different regions, while
  bag rollouts re-spend it on the same salient set. (d) is not excludable by
  this design and does not need to be for the routing decision — but it means
  a positive result supports "scope restriction helps" without certifying the
  specific "continuation-path suppression" story in the hypothesis prose.
- Testable prediction: as declared (unit.md:185), directionally correct and
  conservative — the bag has `K` whole-image opportunities per object versus
  one owning call for the masked arm, and more decoded tokens in total, so a
  masked win at matched calls is a strong result.
- Falsifier: `FULL_BAG_K` matches the structured arms (unit.md:185). Valid
  once P1-3 guarantees the bag is a live control and P2-9 guards the
  mask-harm null.
- Remaining confound after corrections: vision-side versus post-vision locus
  (explicitly conceded, unit.md:170-176; overview.md:126-131), and (d) above.
- Allowed conclusion on a positive result: behavior-level scope-restriction
  benefit; route to the smallest routing/scope intervention (unit.md:161-162).
  Forbidden: any language-tower localization, any slot/ledger implication.

### `HYPOTHESIS_HISTORY_HORIZON` — History-Horizon Instability (unit.md:122-124, 186; overview.md:67-72)

- Motivation: prior own-prefix evidence shows prefix state shapes behavior
  strongly; long prefixes could degrade retrieval or continuation.
- Identifiable mechanism, as instrumented: *clean, teacher-injected history at
  matched spatial input reduces rescue relative to reset.* Narrower than the
  prose, which names "error-bearing" prefixes and own-prefix errors (P2-1).
- Strongest competing explanations: (a) repetition-penalty prompt scope
  (P1-1) — currently the sharpest; (b) history referencing masked-out content
  (P2-2); (c) learned list-length/count prior — a long injected list raises
  STOP probability because training sequences of that length are near their
  end; this is a *history-content prior*, not retrieval instability, and the
  contrast cannot separate them; (d) geometry-order compatibility — controlled
  by `MASK_CUMULATIVE_PERMUTED`.
- Testable prediction: reset above cumulative for both input families with a
  worsening gap by realized prompt length (unit.md:186) — the slope adds a
  dose-response signature that the binary contrast lacks; good.
- Falsifier: cumulative matches or exceeds reset — over-broad as written
  (P2-1): it rules out clean-history harm only.
- Remaining confound: (a)-(c) above until P1-1 is fixed and the wording is
  narrowed; also note ownership already removes the main *benefit* history
  could deliver (cross-region duplicate suppression), which stacks the deck
  toward reset ≥ cumulative — acceptable, but the unit's existing caveat
  (unit.md:203-204) should be read as covering this direction too.
- Allowed conclusion on a positive result: clean-history retention is harmful
  at matched scope under the frozen decode policy; route to compact
  state/short-horizon experiments (unit.md:163-164). Forbidden: "the model
  needs a ledger", "own-prefix errors are the cause".

### `HYPOTHESIS_TILE_LOCAL_COMPUTATION` — Tile-Local Computation Explanation (unit.md:124-128, 187; overview.md:74-79)

- Motivation: separate "tiles help because of the visual computation" from
  "tiles help because of scheduling".
- Identifiable mechanism: only as a *residual family* — the unit says so
  explicitly (unit.md:126-128), which is the honest choice. The family
  contains at least: fewer image tokens, positional-encoding change, context
  removal, padding, and one member the unit does not name: *coordinate-emission
  renormalization* — tile outputs are norm1000 in the tile frame, so object
  extents occupy much larger normalized spans than the full-image training
  distribution. A tile gain or loss can live entirely in the coordinate head's
  response to that shift. Worth adding to the family list (one clause at
  unit.md:124-128).
- Prediction/falsifier: `TILE_RESET > MASK_RESET` with `MASK_RESET ≈
  FULL_BAG_K` raises the family; `MASK_RESET > FULL_BAG_K` at identical token
  budget demotes it (unit.md:187). Sound at family granularity.
- Remaining confound: irreducible within this unit; the declared later
  orthogonal discriminator is the correct handling.
- Allowed conclusion: family-level attribution only; never "fewer tokens is
  the cause".

### `HYPOTHESIS_CONTEXT_BOUNDARY` — Partition and Boundary Context (unit.md:129-131, 188)

- Motivation: boundary cuts and lost context are the classic tiling artifacts.
- Identifiable mechanism: halo contribution, via `MASK_RESET` versus
  `MASK_RESET_CORE_ONLY` with the boundary-distance stratification
  (unit.md:424-425). Clean pairing (same fill, same frame, same budget).
- Competing explanation: halo benefit that is actually *more visible pixels of
  the owned object itself* (large objects protruding past the core), not
  "context"; the area and boundary-distance strata mostly separate this.
- Falsifier: no boundary-local benefit or duplicate-union-only gain
  (unit.md:188). Fine.
- Remaining confound: objects larger than core-plus-halo fail structurally in
  every spatial arm; interpret via the area stratum, and expect spatial-arm
  Retention losses concentrated there (connects to P3-4).
- Allowed conclusion: halo width matters/does not matter for boundary objects
  under this fill; nothing about general context reasoning.

### `HYPOTHESIS_GEOMETRY_ORDER` — Geometry-Sorted Traversal Prior (unit.md:132-134, 189; overview.md:94-99)

- Motivation: the adapter was trained on validated top-to-bottom/left-to-right
  row order (`docs/SYSTEM_OVERVIEW.md:49`); raster traversal plus cumulative
  history reconstructs that order in the prompt.
- Identifiable mechanism: order-compatibility of *cumulative* arms, via the
  frozen permutation (`MASK_CUMULATIVE_PERMUTED`). Reset arms are order-free
  by construction (no cross-call state), so restricting the permutation test
  to the cumulative arm is correct, not an omission.
- Competing explanation: a permuted-order penalty that is really a
  prompt-length × cell-content interaction — one frozen permutation cannot
  fully cross cell identity with traversal position; the planned
  prompt-length-slope analysis is the partial remedy.
- Falsifier: order-insensitivity within uncertainty (unit.md:189). Fine.
- Remaining confound: single permutation; single checkpoint trained only
  geometry-sorted (unit.md:197-199) — declared.
- Allowed conclusion: traversal-order compatibility does/does not modulate
  cumulative gains for this checkpoint. Forbidden: anything about randomly
  ordered training.

### `HYPOTHESIS_ANNOTATION_GAPS` — Dense-Scene Annotation Gaps (unit.md:134-136, 190; overview.md:101-107)

- Motivation: dense Common Objects in Context annotations are known-incomplete;
  precision drops must not be auto-read as hallucination.
- Identifiable mechanism: blinded audit versus official-annotation
  discrepancy, kept in separately named ledgers (unit.md:375-377) — good.
- Competing explanation: audit acceptance bias — human adjudicators shown
  candidate boxes tend to confirm them. The pre-execution exhaustive ledger is
  the right defense *if* its freeze holds (P2-5); blinding to arm identity
  (gate 8) removes arm-differential bias but not the generic
  prediction-anchoring bias, which inflates audit-augmented metrics for all
  arms symmetrically — a level shift, mostly harmless to paired contrasts.
- Falsifier: blinded audit labels the new predictions unsupported
  (unit.md:190). Fine.
- Remaining confound: `iscrowd` and ambiguity semantics (P1-5) sit exactly on
  this hypothesis's evidence path.
- Allowed conclusion: label-hole accounting is/is not required before training
  against apparent false positives (unit.md:474-475). Forbidden: "the model
  discovers unlabeled objects" as a general capability claim.

### `HYPOTHESIS_RESIDUAL_LIMIT` — Residual Recognition or Control-State Limit (unit.md:136-138, 191; overview.md:108-113)

- Motivation: whatever no intervention rescues bounds all scope/history
  stories.
- Identifiable mechanism: persistent-miss identities shared across
  `FULL_BAG_K`, `TILE_RESET`, `MASK_RESET` (unit.md:426-427). This is an
  *observational residue*, not an intervention; correctly framed.
- Competing explanation: shared misses caused by shared protocol limitations
  (mask fill harming the same small objects everywhere; matching threshold
  excluding the same blurry instances) rather than model limits. The manual
  visibility check ("manually visible objects", unit.md:462-463) is the
  partial control.
- Falsifier: structured scope or reset reliably rescues them (unit.md:191).
- Remaining confound: recognition versus control-synthesis is explicitly not
  separable here (overview.md:111-113) — properly conceded.
- Allowed conclusion: a bounded residual set exists and routes back to
  observability/control-synthesis work (unit.md:166-168).

### `INVARIANT_NO_RESIZE_SCALE` — No-Resize Scale Invariant (unit.md:142-144; overview.md:81-87)

- Correctly classified as a protocol invariant, not a hypothesis — its
  violation invalidates the unit rather than updating any belief
  (unit.md:142-144). This answers review objective A.6 in the unit's favor: no
  protocol invariant is dressed up as a hypothesis, and no hypothesis is
  secretly an invariant. Verification is executable (processor receipts,
  `do_resize=false`, unchanged object pixel dimensions, gates 1-2 at
  unit.md:433-435).
- Residual risk: padding (unit.md:288-290) changes canvas statistics without
  changing object scale; the invariant governs scale only and the unit says
  so.

## Experimental-Arm Audit

**`FULL_SINGLE` — Full-Image Single Rollout (unit.md:100-101, 313).**
Changes: nothing (reference condition), but note it is a *sampled* rollout
under the unit's decode policy, not the historical deterministic production
rollout. Invariant: everything. Estimates: the conditioning event (misses) and
the paired baseline. Cannot estimate: anything mechanistic alone. Necessary:
yes. Note: report a side-by-side sanity comparison against the existing
deterministic val200 artifact (unit.md:246-252 already permits this) so the
motivating phenomenon is visibly reproduced under the new decode policy.

**`FULL_BAG_K` — Full-Image K-Rollout Independent Bagging (unit.md:102-103,
314).** Changes: adds `K−1` sampled rollouts and a merge. Invariant: input,
prompt, token budget per call. Estimates: the generic resampling rescue rate
at matched calls — the control that makes the headline meaningful. Cannot
estimate: anything about *why* resampling rescues. Necessary: absolutely — it
is the design's best idea. Fragilities: temperature strength (P1-3), merge
burden (P1-2), penalty mechanics (P1-1), Retention ≈ 1 by construction
(P3-4).

**`TILE_RESET` — Native-Scale Tile with Per-Tile Reset (unit.md:104-105,
315).** Changes: token count, positional grid, padding, visual context,
coordinate-emission frame — everything except object pixel scale. Invariant:
pixel scale (receipted). Estimates: the ceiling of local decomposition; the
Local-Frame Effect against `MASK_RESET`. Cannot estimate: any single member of
the tile-local family (declared residual). Necessary: yes, as the bridge to
tiling folklore and the family separator.

**`TILE_CUMULATIVE` — Native-Scale Tile with Cumulative History
(unit.md:106-107, 316).** Changes: adds cross-frame history with no coherent
coordinate semantics (P1-4). Estimates today: frame-incoherent-history
confusion. Cannot estimate: history horizon. Necessary: no — demote to
exploratory or drop; the masked pair carries the history contrast.

**`MASK_RESET` — Full-Canvas Masked Region with Per-Region Reset
(unit.md:108-109, 317).** Changes: visible content only, at fixed canvas,
token grid, and coordinate frame — the cleverest arm in the matrix. Invariant:
image-token budget, coordinate statistics, prompt. Estimates: the headline
scope-restriction effect. Cannot estimate: vision-side versus post-vision
locus (masking changes vision-tower activations globally; conceded at
unit.md:200-202, 170-176); scope benefit versus mask harm on a *null* (P2-9).
Necessary: yes — primary.

**`MASK_CUMULATIVE` — Full-Canvas Masked Region with Cumulative History
(unit.md:110-111, 318).** Changes: adds coherent same-frame history.
Invariant: input distribution relative to `MASK_RESET`. Estimates: the
clean-history Masked History Effect (with P1-1/P2-1/P2-2 caveats). Necessary:
yes — primary.

**`MASK_RESET_CORE_ONLY` — Full-Canvas Masked Core-Only Reset
(unit.md:112-113, 319).** Changes: removes halo. Estimates: halo/context
contribution and boundary sensitivity. Necessary: yes as the boundary
ablation; secondary priority is correct (unit.md:483-485).

**`MASK_CUMULATIVE_PERMUTED` — Full-Canvas Masked Cumulative Permuted Order
(unit.md:114-116, 320).** Changes: traversal order only, with cell-bound seeds
retained (unit.md:68-71) — a well-constructed control. Estimates:
order-compatibility of cumulative gains. Cannot estimate: full cell × position
interaction (one permutation). Necessary: yes, cheap and targeted.

Matrix-level judgment: the eight arms are close to minimal for the declared
questions; only `TILE_CUMULATIVE` fails to earn its interpretation. The
equal-token requirement for the five full-canvas arms (unit.md:322-325) and
the separate reporting for tile arms are the right comparability spine.

## Metric and Matching Audit

- **Matching semantics** (unit.md:358-373): alias-normalized class equality,
  IoU ≥ `0.50` candidates, one-to-one bipartite matching maximizing count then
  total IoU, deterministic tie-break, no post-hoc confidence filter (there are
  no scores in compact rows, so this also removes a degree of freedom), frozen
  `0.75` sensitivity. Well-specified except the P1-5 gaps (`iscrowd`,
  ambiguity flags, out-of-ontology classes) and the unstated side of the
  tie-break ("source order" of predictions, references, or both). The gate-11
  fixture (unit.md:448-450) is an excellent, rare control — extend it with a
  crowd-region case and an ignore case.
- **Local Rescue Rate** (unit.md:81-83, 386-389): per-object conditional on
  paired `FULL_SINGLE` misses; correct estimand for "rescue". Inflation
  channels: repeated opportunities (bag `K` versus mask 1 per object — makes a
  masked win conservative and a masked null ambiguous, P3-3/P2-9); permissive
  matching in crowds (IoU `0.50` boxes near a missed object can "rescue" by
  luck — symmetric across arms but scales with each arm's local box density;
  visible through per-arm precision, which the unit already mandates);
  merge-position of the metric (P1-2 — compute both pre- and post-merge).
- **Retention** (unit.md:386-389): right safety metric for spatial arms;
  cross-arm comparison is structurally unfair to nobody except the reader who
  treats bag Retention as informative (P3-4). The core-interior Retention
  floor should be promoted into an interpretation gate (P2-9).
- **Precision**: manual and official precision are mandated per arm
  (unit.md:417-419); invalid-row denominator needs one sentence (P3-6);
  out-of-ontology handling needs a rule (P1-5).
- **Duplicates**: pre/post-merge duplicate components retained
  (unit.md:421-422); one-to-one matching prevents duplicate recall inflation
  (unit.md:368-369) — correct. Add the cross-instance merge-collapse counter
  (P1-2).
- **Invalid rows**: never matched, counted, and attempted-image denominators
  preserved (gate 7, unit.md:442) — correct and consistent with the
  painted-ground-truth lineage's lessons.
- **Official versus audit-augmented ledgers**: separately named metrics, never
  pooled (unit.md:375-377) — exemplary. Freeze boundary needs P2-5; primary
  ledger needs P2-7.
- **Uncertainty**: 10,000 image-clustered bootstrap replicates, image-level
  pairing, frozen seed (unit.md:377-379, 92-94) — the correct statistical
  unit for object-level nesting under paired arms. Missing: a width target
  that operationalizes "approximately equals" (P2-6) and a one-line
  multiplicity posture (P3-7).

## Outcome-to-Belief Matrix

For each result pattern (effects read as paired differences with
image-clustered confidence intervals excluding zero unless "≈"):

**1. `MASK_RESET` > `FULL_BAG_K` and `MASK_RESET` > `MASK_CUMULATIVE`.**
Supported: scope restriction at fixed token budget adds rescue beyond
matched-call resampling; clean history retention is harmful at matched scope.
Weakened: generic-sampling sufficiency; history-harmless. Unresolved
alternatives: repetition-penalty mechanics for both legs (until P1-1);
vision-side versus post-vision locus; output-length rationing versus
path-suppression inside the scope family. Strongest allowed claim: the
behavior-level double result plus routing per unit.md:469. Next smallest
discriminator: the unit's own — a post-vision spatial scope on *cached
identical visual features* (unit.md:469), which is also the E.3/E.4 answer:
masking does **not** locate the defect after the vision tower, because the
masked pixels change vision-tower activations everywhere; only a
same-visual-feature intervention (one full-image vision pass, scope imposed
after it) can support post-vision localization.

**2. `MASK_RESET` > `FULL_BAG_K`, `MASK_RESET` ≈ `MASK_CUMULATIVE`.**
Supported: scope restriction without history sensitivity. Weakened:
history-horizon as the primary bottleneck (clean-history component only,
P2-1). Unresolved: own-prefix-error history effects. Allowed: routing without
state changes (unit.md:470). Next: spatial routing probe; noisy-history
injection later.

**3. `MASK_RESET` ≈ `FULL_BAG_K`, `TILE_RESET` > `FULL_BAG_K`.** Supported:
the tile-local computation family. Weakened: decoder-side scope accounts —
*only if* the P2-9 mask-harm gate passes; otherwise unresolved. Unresolved:
which family member (declared residual; note coordinate-frame renormalization
as a member). Allowed: investigate token budget / local encoding / crop frame
(unit.md:471). Next: token-budget-matched crop comparisons or feature-cached
scope.

**4. `FULL_BAG_K` ≈ `MASK_RESET` ≈ `TILE_RESET` (all rescue similarly).**
Supported: generic resampling is the operative rescue mechanism. Weakened:
scope and history as primary bottlenecks (at equal budget; per-call effects
remain open, P3-3). Unresolved: whether any structure helps at *larger*
budgets. Allowed: unit.md:472 routing (control synthesis / observability).
Next: none in this family; return to designation/control experiments.

**5. Reset > cumulative in masked arms, but `MASK_RESET` ≈ `FULL_BAG_K`.**
Supported: history retention harms at matched scope (with P1-1 resolved).
Weakened: scope competition. Unresolved: prompt-penalty mechanics if P1-1
unfixed; content-versus-length inside the history family. Allowed:
unit.md:473 routing (state compression before visual selectors). Next: history
content controls (wrong-rows, shuffled-rows) at fixed length — cheap and
decisive for content-versus-length.

**6. Official precision falls while blinded manual precision holds.**
Supported: annotation incompleteness accounting must precede any training
against apparent false positives (unit.md:474-475). Weakened: hallucination
readings of dense-scene false positives. Unresolved: audit acceptance bias
level shift. Allowed: build the incomplete-label evaluation unit first.

**7. The same manually visible objects are missed by every arm.** Supported:
a residual recognition-or-control limit bounds this seam
(`HYPOTHESIS_RESIDUAL_LIMIT`). Unresolved: recognition versus
control-synthesis (explicitly out of scope, overview.md:111-113). Allowed:
route back to observability/control-state work (unit.md:166-168). Next: the
target-specific consumption probes already mandated by
`require-target-specific-causal-consumption`.

**8. Spatial arms lose `FULL_SINGLE` true positives (low Retention) even away
from boundaries.** Supported: masking/tiling harms recognition of visible
content (fill or context damage). This pattern reclassifies any accompanying
null as "unresolved", per P2-9, and blocks deployment-flavored conclusions
about decomposition. Next: fill-design or context ablation, pre-registered in
a new run identifier (the unit's stop rule at unit.md:485-487 already forbids
post-hoc fill shopping — good).

## Assumptions Not Actually Tested

Status of the ten challenged assumptions (tests fully / partially / leaves
unresolved):

1. **More local predictions imply the full-image model had already recognized
   those objects.** Partially tested. Rescue proves the checkpoint can detect
   the object under restricted input; it does not show the full-image forward
   pass computed that identity. The unit never *states* the stronger claim,
   and its routing ("smallest routing or scope intervention", unit.md:161-162)
   only needs the weaker one; latent full-image recognition would need
   feature-level probes (the "visual observability" branch, unit.md:166-168).
2. **Masking changes only competition within the decoder.** Not assumed —
   explicitly disclaimed (unit.md:200-202, 174-176). The locus question is
   deliberately left unresolved; correct posture.
3. **No resizing means the vision-side representation is controlled.** Not
   assumed for tiles (unit.md:199) or masks (unit.md:200-202). The invariant
   controls *object scale* only, and receipts make that executable. Unresolved
   by design.
4. **Reset versus cumulative isolates only language-history length.** Not
   fully true and only partially acknowledged: permuted order and
   prompt-length slopes address parts; repetition-penalty scope (P1-1),
   masked-content reference (P2-2), and clean-versus-own-prefix content (P2-1)
   remain bundled. Partially tested after corrections.
5. **More predictions imply better enumeration.** Rejected by design:
   one-to-one matching, mandatory precision/duplicate/invalid reporting
   (unit.md:368-369, 417-422), consistent with the painted-ground-truth
   lineage's precision-collapse lesson. Tested fully at the metric level.
6. **Most false positives are missing annotations.** Treated as a hypothesis
   with a blinded falsifier (`HYPOTHESIS_ANNOTATION_GAPS`), not an assumption.
   Tested (subject to P2-5 freeze hygiene).
7. **Geometry-sorted output order is merely superficial.** Not assumed; the
   permuted cumulative arm tests order compatibility. Partially tested (one
   permutation, one geometry-sorted checkpoint; unit.md:197-199).
8. **A positive result implies a ledger or persistent state is required.**
   Explicitly forbidden, repeatedly (unit.md:161-162, 203-204, 476;
   overview.md:53-56). Not assumed anywhere. The unit is fully aligned with
   `let-architecture-emerge-from-hypothesis-gates`.
9. **Equal call count implies equal computation.** Explicitly disclaimed
   (unit.md:77-80, 349-350) with realized-budget receipts (gate 9,
   unit.md:444-445). Honest as long as reports always carry the realized
   budgets next to the matched-call label.
10. **A decodable representation is causally consumed.** Out of this unit's
    scope (it is purely behavioral) and already governed by the standing
    decision `require-target-specific-causal-consumption`. Leaves unresolved,
    correctly.

Additional assumptions outside the unit's scope that its conclusions will
silently condition on:

- **Decode-policy transfer**: all effects are measured at a sampled, frozen
  decode policy; transfer to the deterministic production decode is untested.
- **Single checkpoint, single objective family**: geometry-sorted training
  with Gaussian Soft-Target Coordinate Cross-Entropy with Ordered
  Cumulative-Distribution Penalty; the pure cross-entropy replication is
  explicitly reserved (unit.md:253-257).
- **Fill-design point**: one channel-mean fill; conclusions are conditional on
  it (the unit's own inconclusive-on-artifact rule, unit.md:302-304, guards
  only visible extremes).
- **Grid geometry**: four-by-four with `25%` halo is one point in
  decomposition space; effects may be non-monotonic in cell size.
- **One-to-one IoU `0.50` as the operational meaning of "detects"**: the
  `0.75` view is the only sensitivity.
- **val200 provenance**: the cohort inherits whatever selection produced
  val200 in the first place.

## Minimal Required Corrections

Bounded edits only; no architecture or protocol redesign.

**Before implementation readiness (metric/runner code may be written after
these):**

1. Add crowd/ignore, ambiguity-flag, and out-of-ontology semantics to the
   matching contract, and extend the gate-11 fixture with a crowd and an
   ignore case (P1-5).
2. Declare the cumulative-history acceptance predicate and injection template
   in the frozen-factor list (P2-3).
3. Specify pre-merge Local Rescue Rate as a co-primary/sensitivity output and
   the cross-instance merge-collapse counter in the metric contract (P1-2).
4. Define the tile-history frame policy or demote/drop `TILE_CUMULATIVE` and
   remove the Tile History Effect from the primary estimands (P1-4).
5. Declare the seed derivation function over (image, cell) (P2-8).

**Before execution readiness (the `planned → ready` amendment):**

6. Predeclare the sampling temperature and nucleus cutoff with justification,
   and add the prediction-set-level bag-diversity gate (P1-3).
7. Record the repetition-penalty scope semantics as a receipt and either set
   the unit's penalty to `1.0` or predeclare the paired
   `{1.0, 1.10}` sensitivity for the three headline arms (P1-1).
8. Tighten the cohort wording: the existing rollout may size and sanity-check
   but not select; add the minimum paired missed-object count and the
   confidence-interval width target that defines "approximately equals"
   (P2-4, P2-6).
9. Declare the audit-ledger freeze boundary and the role of post-hoc blinded
   adjudication (P2-5); name the Audit-Augmented Local Rescue Rate as the
   headline's primary ledger (P2-7).
10. Add the bag-call-1 versus `FULL_SINGLE` replay receipt and the planned
    invocation table (P2-8, P3-9); harmonize gate 3 with the five-arm
    token-budget requirement (P3-1).

**Before interpretation readiness (may be edited any time before results are
read):**

11. Narrow `HYPOTHESIS_HISTORY_HORIZON`'s falsifier to the clean-history
    component and name the masked-content-reference confound (P2-1, P2-2).
12. Add the mask-harm Retention floor as a precondition for interpreting any
    null Spatial-Scope Effect, and soften the "generic sampling, not spatial
    scope" signature to its behavior-level form (P2-9, P3-3).
13. Label Retention as arm-specific, note the bag's `K−1` informative-call
    asymmetry, and note non-commensurability with prior greedy-matcher metrics
    (P3-4, P3-5); state the invalid-row precision denominator and the
    descriptive status of non-headline views (P3-6, P3-7).

## Reusable Research-Workflow Recommendations

1. **Control arms need their own strength gates.** A matched-budget control
   (here, bagging) should carry a mechanics gate proving it is a *live*
   control (prediction-set diversity floor), exactly as treatment arms carry
   invariant receipts. "The control existed" is not "the control was strong".
2. **Freeze decode-mechanics *semantics*, not just values.** Whenever a
   contrast manipulates prompt content or length, penalty scope, stop
   handling, and cap semantics are part of the contrast, not background;
   record them as receipts from the installed generation path, per the
   upstream-boundary rule in the repository guide.
3. **When an aggregation policy does different work per arm, expose
   pre-aggregation co-primaries and asymmetry counters.** Nominally "the same
   merge policy" is not operational symmetry.
4. **Conditional rescue estimands should ship with per-arm opportunity
   accounting** (calls with a chance at each object), so budget-level and
   per-opportunity readings cannot be conflated.
5. **Human reference ledgers get an explicit freeze boundary.** Pre-execution
   exhaustive ledgers are references; post-hoc blinded adjudication feeds
   hypotheses; ledger changes fork a named version.
6. **Nulls need predeclared interpretability preconditions.** Any intervention
   that can plausibly harm (masking, cropping) should pre-register the
   harm-check (here, core-interior Retention) that a null must pass before it
   counts against a hypothesis.
7. **The terminology-registry discipline in this unit — including quarantining
   legacy path tokens like `gaussian_rps` behind a canonical plain-language
   name — is worth adopting as the template for all new units**, alongside its
   deterministic matching fixture (gate 11), which should become a standing
   pattern for every metric-bearing unit.

## Final Recommendation

- **Proceed?** Yes — **approve after named corrections**. The unit is the
  right next discriminator for this investigation, its arm matrix is close to
  minimal, and its claim boundaries already forbid every overreach the
  standing decisions worry about.
- **Correct first:** the five P1 items (repetition-penalty scope on the
  primary contrasts; pre-merge co-primary plus merge-collapse counters;
  bag-diversity/temperature principle; `TILE_CUMULATIVE` frame policy or
  demotion; crowd/ambiguity/ontology matching semantics), folded into the
  already-planned readiness amendment, plus the P2 execution items (cohort
  wording, power target, ledger freeze, primary ledger, seed/replay receipts).
- **Do not build yet:** reusable runners or analysis modules under
  `scripts/research/` or `src/analysis/` beyond what the unit itself
  authorizes (it authorizes nothing; unit.md:514-517), any merge/matching
  library promoted as a stable contract, any OpenSpec change (nothing here is
  a stable compatibility-sensitive contract yet — deferral is correct per the
  promotion ladder, `research-graph-contract.md:169-184`), and — per the
  unit's own outcome map — any slot, ledger, cursor, or architecture work
  regardless of result (unit.md:476).
- **Evidence that would justify the next research unit:** a mechanics-clean
  run in which (a) the primary panel passes all eleven gates plus the
  corrections above, and (b) either `MASK_RESET` separates from `FULL_BAG_K`
  with agreeing pre-/post-merge signs and a surviving repetition-penalty
  sensitivity — justifying the cached-visual-feature post-vision scope unit —
  or reset separates from cumulative under the same conditions — justifying a
  history-content-versus-length unit; or (c) all arms converge, justifying the
  return to object-specific control synthesis already queued by the decision
  graph. A result that fails the bag-diversity gate or the mask-harm floor
  justifies only a repaired rerun, not a new unit.
