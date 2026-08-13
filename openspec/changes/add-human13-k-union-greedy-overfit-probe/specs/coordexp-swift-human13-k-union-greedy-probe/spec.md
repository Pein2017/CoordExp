## Purpose

Defines the bounded mechanical behavior needed to consolidate already observed,
IoU-valid Human-13 K-sampled owner rows into clean-greedy training probes without
weakening blind-data defaults or promoting research assumptions into production.

## ADDED Requirements

### Requirement: Research-Unit-Bound Probe Authorization

The probe SHALL bind its panel, Source checkpoint, prompt and wrapper,
tokenizer, sampling recipe, matcher, target-row rule, arm matrix, and artifact
root to the owning Human-13 research unit. The ordinary StateBank blind-image
policy SHALL remain unchanged; only an experiment-local reader bound to the
exact panel hash and explicit `overfit_only` purpose MAY admit the panel.

The probe SHALL support a plan-only materialization path that performs no model
load, decode, forward, backward, optimizer mutation, checkpoint write, or GPU
allocation. No implementation test, plan generator, or review command SHALL
implicitly launch a model job.

#### Scenario: Exact overfit identity is materialized

- **WHEN** the declared panel hash, thirteen image identities, Source identity,
  and research-unit identity match the approved binding
- **THEN** the probe SHALL emit a dry-run plan with those identities and zero
  model or optimizer actions
- **AND** the generic blind-image policy SHALL remain unchanged.

#### Scenario: Blind cohort is requested through the generic path

- **WHEN** the same image identifiers are presented to the ordinary StateBank
  loader or the experiment-local binding is missing or mismatched
- **THEN** admission SHALL fail before packing or model execution
- **AND** no ordinary configuration flag SHALL convert the failure into a
  general blind-cohort exception.

### Requirement: Explicit Batch-Four K Sampling

For each image, sampled discovery SHALL consist of exactly sixteen independent
`n=1` requests with explicit seeds `21001` through `21016`, temperature `0.4`,
top-p `0.95`, repetition penalty `1.10`, and maximum new tokens `512`. The
collector SHALL submit four successive physical request batches of size four,
preserve request-to-seed identity, and restore canonical seed order in the
frozen ledger.

Sampling settings SHALL be separate from the clean-greedy evaluation settings.
Clean-greedy evaluation SHALL use one HF request at physical batch size one and
the Source-matched repetition penalty `1.0`.

#### Scenario: One image completes K discovery

- **WHEN** the collector receives all four declared request batches
- **THEN** the frozen ledger SHALL contain one and only one trajectory for each
  explicit seed `21001..21016`
- **AND** no trajectory identity SHALL depend on `n>1` child-index seed
  expansion.

#### Scenario: Request batch is incomplete or reordered

- **WHEN** a seed is missing, duplicated, served under a different request
  setting, or cannot be mapped to its explicit request
- **THEN** ledger finalization SHALL fail for that image
- **AND** a partial K set SHALL NOT be interpreted as owner absence.

#### Scenario: Cache telemetry is unavailable

- **WHEN** the vLLM runtime does not expose encoder, prefix, or cache-hit
  counters
- **THEN** the collector SHALL record the telemetry as unavailable without
  blocking a complete scientific ledger
- **AND** the receipt SHALL NOT claim that possible cache reuse was observed.

### Requirement: Frozen Owner and Prefix Ledger

Before any update, the probe SHALL freeze raw generated token identifiers,
parser and truncation status, chronological duplicate classification,
one-to-one same-category owner assignment at IoU at least `0.50` over retained
non-duplicate rows, Source-greedy owners, K-union owners, K-hit greedy misses,
K-miss owners, selected native target rows, residual order, raw prefixes,
duplicate-cleaned treatment prefixes, and duplicate-event provenance.

The probe SHALL scan complete predicted rows in every raw Source and K
trajectory in generation order, retain the earliest row, and classify every
later row as a duplicate when its class-agnostic prediction-to-prediction IoU
with an already retained row is greater than `0.95`. A duplicate row SHALL be
ineligible for owner assignment, owner-set support, Source replay, positive
target selection, and A4 candidates even if matching without the duplicate
rule could assign it to a distinct GT owner. It SHALL be removed from the
synthetic treatment prefix while remaining in raw provenance and duplicate
unlikelihood. The probe SHALL concatenate exact retained token spans without
decode and re-tokenize. Unmatched non-duplicate, malformed, and geometry-
invalid spans SHALL remain context-only and SHALL receive no positive or
negative target unless another explicit requirement names one.

#### Scenario: Duplicate-cleaned prefix is built

- **WHEN** a raw prefix contains one retained complete row and a later complete
  row above the duplicate IoU threshold
- **THEN** the clean prefix SHALL omit only the later row's exact token span
- **AND** the raw prefix and the removed row's original decision state SHALL
  remain available for duplicate unlikelihood.

#### Scenario: K-miss owner is absent from all sampled trajectories

- **WHEN** a trusted GT owner is not assigned in Source greedy or any complete
  K trajectory
- **THEN** the owner SHALL remain in the K-miss set
- **AND** it SHALL contribute neither a Stage-1 positive nor a negative target.

#### Scenario: Later near-identical row could match another dense GT owner

- **WHEN** a later row exceeds the frozen duplicate threshold but the
  one-to-one matcher could otherwise assign it to a distinct GT owner
- **THEN** the chronological duplicate classification SHALL win
- **AND** manifest finalization SHALL prove that the row occurs in no owner,
  replay, target, or candidate-positive set.

### Requirement: Terminal-Safe Minimal Objective Family

The probe SHALL implement only the objective family needed by Frozen Source,
the full-GT capacity control, A0, A1, A3, A4, A7, A8-prime, and conditionally
A6. All Stage-1 positive row objectives SHALL mask the final chat terminal.
Source-body replay SHALL supervise trusted Source-owner row bodies only and
SHALL mask terminal, unmatched, malformed, and unresolved spans.

Owner-mean row CE SHALL mean the mean target-token CE within each complete row,
followed by the mean over eligible physical owners in the complete planned
panel step. It SHALL NOT silently substitute field-balanced schema/coordinate
weighting.

#### Scenario: Partial owner set is supervised

- **WHEN** at least one K-miss owner remains outside the Stage-1 target set
- **THEN** no treatment, Source replay, or full-H chain SHALL select the chat
  terminal as a positive target
- **AND** terminal MAY appear only as an unsupervised context token or a
  mechanically selected non-target competitor.

#### Scenario: H1 owner segments share one panel update

- **WHEN** multiple K-hit owners from one or more images are represented as
  independent H1 segments
- **THEN** their owner-mean objective SHALL be computed at the same parameter
  state and accumulated before exactly one panel optimizer update
- **AND** sequential per-owner AdamW updates SHALL NOT be represented as an
  equivalent implementation.

### Requirement: Frozen Initial Update Contract

The initial sixteen-exposure screen SHALL train only the language-tower DoRA
payload. Vision, multimodal aligner, token embeddings, and base language
weights SHALL remain frozen. Every updated arm SHALL use torch AdamW with
learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay,
global gradient clip `1.0`, and cosine-with-warmup scheduling with zero warmup
and a sixteen-update horizon.

After each family applies its declared normalization, the H/replay/duplicate
coefficients SHALL be `(0,1,1)` for the A0 shared no-H background control,
`(1,1,1)` for A1/A3/A4/A6/A8-prime, `(1,0,1)` for A7, and `(1,0,0)` for the
separate full-GT capacity control. Active families SHALL NOT be renormalized to
keep the coefficient sum constant. A later 100-update screen SHALL restart
Source and fresh AdamW with its own 100-update schedule and SHALL NOT claim
optimizer continuation from the short run.

#### Scenario: Arm configuration omits or changes one frozen value

- **WHEN** a resolved initial-screen config has an unset or mismatched
  trainable surface, optimizer value, scheduler horizon, clipping value, or
  family coefficient
- **THEN** materialization SHALL fail before model execution
- **AND** A7 and A0 SHALL be the only declared coefficient ablations.

### Requirement: Once-Per-Image Any-Valid Row Mass

A4 SHALL score exact-token-deduplicated, complete, row-terminated native K-hit
candidate rows from one image as one logically atomic objective. It SHALL compute the
negative log probability of their prefix-free union once per image exposure,
not once per owner. It SHALL report candidate weights and effective owner count
and SHALL NOT describe length-normalized row energy as probability.

The logical objective MAY span several physical packs only through an exact
fixed-parameter two-pass implementation: first score every candidate without
gradients, compute one fp32 global per-image softmax over the complete candidate
set, then replay the identical candidates with detached global weights and
accumulate all gradients before one AdamW step. Chunk-local union losses,
optimizer changes between score and replay, incomplete candidate coverage, or
score/replay identity drift are forbidden.

#### Scenario: One easy and two hard candidate rows are scored

- **WHEN** one image has three distinct complete K-hit candidate rows
- **THEN** A4 SHALL emit one union-mass loss for the image
- **AND** it SHALL report all three normalized candidate weights and
  `1 / sum(weight^2)`.

#### Scenario: Logical candidate group exceeds one physical pack

- **WHEN** every candidate segment fits the 12,000-token bound but their
  complete logical group does not fit one physical forward
- **THEN** A4 SHALL use exact two-pass global score/gradient replay across
  deterministic isolated packs at one parameter state
- **AND** it SHALL emit one logical union loss and one global weight vector per
  image, not one loss per pack.

#### Scenario: One candidate segment or global binding is invalid

- **WHEN** an individual candidate segment exceeds 12,000 tokens, the complete
  set cannot be enumerated, or score/replay candidate identities differ
- **THEN** A4 SHALL fail before optimizer mutation
- **AND** it SHALL NOT approximate the missing mass or renormalize per chunk.

### Requirement: Coherent Full-Residual Bottleneck Objective

A8-prime SHALL use exactly the same frozen native rows and coherent full-
residual order as A1. For every non-terminal target token in that chain it
SHALL compare the target logit with the highest non-target logit, treat the
competitor identity as a stop-gradient selection, and compute fp32
`relu(margin_required - target_margin)`. It SHALL average within row and then
over owners, include row wrappers and row terminators, exclude the final chat
terminal, and contain no CE continuation term.

The positive required margin SHALL be frozen before updates from the maximum
aligned packed-versus-HF target-margin drift plus `1e-4`. A8-prime SHALL be
mechanically blocked when aligned finite scores are unavailable or the required
margin exceeds `0.5`.

#### Scenario: Target is already a stable strict argmax

- **WHEN** a coherent-chain target margin is at least the frozen required
  margin
- **THEN** that token site SHALL contribute zero A8-prime gradient.

#### Scenario: Two H1 rows diverge at one shared prefix

- **WHEN** two valid native H rows have different next tokens at the same exact
  prefix
- **THEN** the probe SHALL NOT construct independent singleton top-1 margins
  for both siblings at that state
- **AND** A8-prime SHALL remain a single coherent full-residual path.

### Requirement: Uncapped Duplicate-Event Unlikelihood

Every frozen later-duplicate event in Source greedy and all sixteen artifact-
valid K trajectories SHALL contribute one fp32 unlikelihood term at the final
coordinate token that closes its duplicate box, conditioned on the original
raw prefix and preceding tokens from that duplicate row. The term SHALL be
mathematically `-log(1-p(token))` but SHALL be evaluated stably as
`softplus(z_target - logsumexp(z_non_target))`; direct subtraction from a
rounded softmax probability is forbidden. No duplicate event MAY be capped,
sampled, or discarded because of its ordinal position. The panel objective
SHALL first average events within image and then average eligible images.

Duplicate-free output SHALL be reported as a goal and SHALL NOT be a hard arm
promotion gate.

#### Scenario: One image contains many later duplicates

- **WHEN** a frozen raw trajectory contains any number of complete later rows
  above the duplicate threshold
- **THEN** every such row SHALL have one selected final-coordinate
  unlikelihood event
- **AND** the artifact SHALL report raw event count, consumed event count, and
  zero capped events.

#### Scenario: Duplicate target probability saturates in fp32 softmax

- **WHEN** the selected duplicate token is separated from all non-target
  logits by at least thirty nats
- **THEN** the stable unlikelihood term and its gradient SHALL remain finite
- **AND** the arm SHALL NOT stop because `1 - softmax` rounded to zero.

### Requirement: Isolated Panel-Step Packing and Arm Parallelism

The probe SHALL use the accepted no-padding varlen packing and multimodal
position reset for each logical segment under `global_max_length=12000`.
Independent segments SHALL be deterministically ordered by descending encoded
length with stable identity tie-breakers and first-fit into physical packs. A1,
A8-prime, and full-GT SHALL keep one coherent segment per image; A4 SHALL keep
one image's candidate group logically atomic while permitting its exact
two-pass candidate segments to occupy several physical packs.

All packs in one panel exposure SHALL use the complete panel denominator and
accumulate before exactly one AdamW update. Packing SHALL be reported only as a
padding and launch optimization; no image-encoder, prefix-KV, or forward-FLOP
reuse MAY be claimed without direct measurement.

Each training arm SHALL run as an independent one-rank Accelerate process on
one GPU. The launcher SHALL support at most eight concurrent arm processes and
SHALL NOT introduce unequal-rank DDP or cross-rank candidate scheduling.

#### Scenario: Independent H1 segments span several packs

- **WHEN** one panel exposure requires multiple no-padding packs
- **THEN** every pack SHALL use isolated attention and reset positions
- **AND** the optimizer SHALL remain unchanged until all packs have contributed
  their globally normalized gradients.

#### Scenario: Eight GPUs are available

- **WHEN** up to eight applicable arm runs are authorized together
- **THEN** the launcher SHALL assign at most one independent one-rank arm
  process per GPU
- **AND** no arm SHALL share parameters, optimizer moments, or output roots
  with another arm.

### Requirement: Bounded Census, Vertical Slice, and Outcome Artifacts

The probe SHALL provide a dry-run materialization, a separately authorized
full-panel model-derived discovery/freeze gate, a no-update trie/bottleneck
census after target selection and before training, and one-image production-
shaped vertical slice. The discovery gate SHALL acquire or identity-verify all
thirteen Source HF clean-greedy outputs and all 208 explicit K requests before
sealing the canonical manifest. A partial manifest SHALL be mechanics-only and
SHALL NOT choose a training image, A6 applicability, or A8-prime margin. The
census SHALL not change target rows, owner sets, residual order, or arm
weights. The vertical slice SHALL cover
materialization, packing, forward/backward, one update, checkpoint write/read,
batch-size-one HF clean greedy, declared owner matching, and final projection.

Artifacts SHALL remain compact: one frozen manifest/ledger, resolved arm plans,
the existing compact training run files, raw clean-greedy outputs, one owner-
outcome table, and bounded performance counters for pack count, packed tokens,
zero padding, utilization, GPU seconds, wall time, and peak memory. The change
SHALL NOT add a second evidence journal, redundant seal hierarchy, cache layer,
or general candidate-tree framework.

#### Scenario: No-update census completes

- **WHEN** the frozen H-row trie and coherent full-residual chain are scored
- **THEN** the census SHALL report first non-argmax site, minimum strict margin,
  token-role counts, ties, viable-child status, and cross-surface drift
- **AND** the selected rows and order SHALL remain byte-identical.

#### Scenario: Census processor skeleton is cloned

- **WHEN** an image-aware encoded skeleton is cloned for packed or HF scoring
- **THEN** its exact image identity, prompt boundary, and owner-row token
  metadata SHALL remain present and value-identical
- **AND** missing dynamic metadata SHALL fail before a model forward rather
  than producing a partial census artifact.

#### Scenario: A6 donor prefix is materialized

- **WHEN** an eligible A6 target has earlier frozen duplicate rows in its donor
  trajectory
- **THEN** the materialized donor prefix SHALL remove exactly those earlier
  duplicate token spans and byte-match the sealed clean-prefix binding
- **AND** the runner SHALL reject any raw-prefix fallback before model load.

#### Scenario: Census run does not reach complete publication

- **WHEN** packed/HF scoring, alignment, finite checks, or receipt publication
  fails
- **THEN** no A8 margin SHALL be inferred from the plan or partial state
- **AND** the next attempt SHALL use a new immutable output root.

#### Scenario: Training is requested before full-panel discovery freezes

- **WHEN** any target-dependent update is requested with missing Source rows,
  fewer than sixteen unique K seeds for any image, a partial manifest, or
  unsealed A6/A8 applicability
- **THEN** the runner SHALL fail before forward or optimizer execution.

#### Scenario: Non-conclusion-critical telemetry is missing

- **WHEN** optional cache or profiler telemetry is unavailable but all identity,
  loss, packing, checkpoint, decode, and matcher evidence is complete
- **THEN** the vertical slice MAY remain mechanically valid
- **AND** the missing telemetry SHALL be marked unavailable rather than causing
  another audit or receipt family.

#### Scenario: Work outside the authorized successor is requested

- **WHEN** execution requests A2, A5, K-miss supervision, online refresh, a
  100-update continuation, checkpoint promotion, or another undeclared arm
- **THEN** the workflow SHALL stop before model execution
- **AND** the current authorization SHALL apply only to the fresh census and
  A4/A6/A8-prime missing-arm successor.

### Requirement: Scope-Limited Outcome Projection

The analyzer SHALL first apply the same chronological class-agnostic pred-pred
IoU duplicate exclusion to original-prompt clean-greedy outputs, count later
rows as burden with no owner credit, and then project retained rows onto the
frozen Source, K-hit, and K-miss owner sets using the declared one-to-one same-
category IoU matcher. It SHALL report per-image results, legacy-twelve and
image-2299 separately, then pooled results: K-hit gained, Source retained and
lost, K-miss incidental gain, final unique owner set, rows/tokens, fixed-budget
and natural-stop coverage, duplicate, unmatched, malformed, invalid and cap-
stop burden, and common-owner IoU.

Teacher-forced loss, fixed-prefix margin, trie membership, and exact-token
reproduction SHALL remain diagnostics and SHALL NOT promote an arm without an
original-prompt clean-greedy owner gain.

#### Scenario: Owner exchange occurs

- **WHEN** an arm adds one K-hit owner but loses one Source-greedy owner
- **THEN** the analyzer SHALL report the gain and loss separately
- **AND** it SHALL NOT represent the result as an unqualified net improvement.

#### Scenario: Clean greedy emits a new IoU-valid box outside the native row bank

- **WHEN** the output matches a frozen GT owner under the declared matcher even
  though its exact coordinates were not a selected native training row
- **THEN** the analyzer SHALL award the owner credit
- **AND** training-time finite-bank membership SHALL NOT constrain final metric
  credit.
