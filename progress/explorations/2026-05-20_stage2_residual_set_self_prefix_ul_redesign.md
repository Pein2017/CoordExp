# Stage-2 Residual-Set Self-Prefix / UL Redesign Decisions

Source discussion was split from `2026-05-19_unified_teacher_forcing_objective_architecture_decisions.md` to keep the Stage-2 residual-set and UL-mining design reviewable.

## 2026-05-20 Stage-2 Residual-Set Self-Prefix Correction Principle

## Decision

The next Stage-2 objective design should be residual-set conditioned at every
grammar-valid self-prefix.

For a rollout prefix `h_t`, define:

```text
G_t = full GT object set for the image
E_t = GT objects already matched by the current self-prefix
R_t = G_t \ E_t
C_t = ValidNext(h_t, R_t)
```

The training target at that prefix is the valid next-token set induced by
`R_t`, not a canonical next object, not a sorted/tail false-negative insertion
list, and not a multi-target merge over multiple rollout trajectories.

The per-token objective is:

```text
L_t = -log P(C_t | h_t)
    = -log sum_{v in C_t} p(v | h_t)
```

K rollout attempts should be interpreted as K independent self-prefix
correction samples:

```text
D_stage2 = union_k correction_samples(rollout_k)
```

They should not be collapsed into one cross-prefix multiple-target objective or
one empirical pseudo-label distribution. Different rollout prefixes create
different autoregressive hidden-state contexts, so their target supports should
be optimized as separate samples.

Approved residual-set membership:

```text
E_t = completed, grammar-valid, uniquely GT-matched emitted objects
R_t = G_t \ E_t
A_t = active compatible GT candidates under the current partial object prefix
```

The active partial object should not be inserted into `E_t` early. A description
prefix, `box_start`, or partial bbox coordinate sequence only narrows `A_t`.
The object is removed from `R_t` only after the entry is complete, grammar-valid,
and uniquely mapped to a GT object id.

Consequences:

- Matched TP: once the full object entry is uniquely matched, its GT id enters
  `E_t` and is unavailable at the next boundary.
- FP or unmatched object: no GT id is removed, so the next correction still uses
  the previous residual set.
- Duplicate: matching an already emitted GT id does not remove a new object; the
  next boundary target is still induced by remaining objects or EOS.
- Partial active prefix: `E_t` is unchanged; only `A_t` and valid next-token
  support are narrowed.

Approved GT mapping contract for completed emitted objects:

```text
matchable(pred, gt)
    = canonical_desc_id(pred) == canonical_desc_id(gt)
      and bbox_iou(pred, gt) >= tau_match

assignment
    = deterministic one-to-one assignment over matchable pairs
      using IoU as score and stable tie-breaks

completed emitted object enters E_t
    only if the assignment selects a unique pred -> gt pair
```

The residual-set builder should not use geometry-only matching. A high-IoU
prediction with the wrong description/class is an FP or unmatched completed
object for residual-set accounting and must not remove a GT object from `R_t`.
For closed-set COCO-style labels, `canonical_desc_id` should be the exact
canonical class id/name. For future open-vocabulary data, this requires an
explicit canonicalization layer rather than an implicit text-similarity match.

Approved active-candidate transition rule:

```text
At object boundary:
    A_t = R_t

After description/text prefix tokens:
    A_t = candidates whose canonical serialized description has that token prefix

After box_start:
    A_t is unchanged, and the grammar state moves to bbox slot x1

After coordinate token at slot s:
    A_t = candidates whose coord_s is compatible with the emitted coordinate token

After full bbox:
    the completed object may enter E_t only through the approved GT mapping
```

The valid next-token set and the state transition must be consistent. If `x1`
selects one of several same-description objects, later `y1/x2/y2` positions
must be conditioned on the filtered candidate set, not on the original
same-description group. This prevents coordinate mixing across object
hypotheses.

First-version coordinate state transition should use exact coordinate-token
compatibility. Coordinate neighborhoods may be retained as an explicit
ablation/future extension for valid-set construction, but they should not
silently loosen `A_t` transitions unless a separate
`coord_transition_tolerance` contract is introduced.

Approved correction anchoring rule:

```text
CorrectionEvent.anchor_prefix
    = earliest grammar-valid prefix immediately before the selected actionable
      error decision

CorrectionEvent.target_state
    = residual-set state at that anchor prefix
```

Stage-2 correction samples should be anchored at the boundary where the model is
about to make the wrong decision, not after the full wrong object has already
been emitted.

Examples:

- Matched TP object repair: anchor at the object-internal position immediately
  before the wrong or uncertain token, such as description continuation,
  `box_start`, or one of `x1/y1/x2/y2`.
- FN / premature stop: anchor at the EOS boundary before EOS and target valid
  continuation from `R_t`.
- Duplicate: anchor at the object boundary before the duplicated object starts;
  target remaining object continuation or EOS, excluding emitted GT ids.
- FP: anchor at the object boundary before the spurious object starts; target
  remaining object continuation or EOS.

First version should default to one first actionable correction event per
rollout sample. This avoids letting a bad rollout emit many highly correlated
post-error samples. An `all_events` mode may be kept as a future ablation, but
it is not the clean default.

Approved duplicate-loss boundary:

Do not restore duplicate-specific training loss in the first residual-set
redesign. Duplicate handling may remain as parsing, guarding, clustering,
quarantine, and diagnostics, but not as a loss family. If duplicate behavior
becomes the dominant failure after the cleaner residual-set objective is
validated, a duplicate-specific mechanism can be reconsidered explicitly.

Approved aggressive unlabeled-object direction:

K rollout consensus may be used to mine unlabeled objects. When all considered
rollouts independently produce a consistent unmatched object cluster
(`consensus_ratio = 1.0`) and the cluster passes the agreed guards, the object
may be promoted for positive supervision. This promoted object must remain
provenance-separated from labeled GT:

```text
labeled TP
    matched to dataset GT

UL-promoted positive
    consensus-mined from K rollouts, not present in dataset GT
```

The builder may treat a UL-promoted positive as TP-like for supervision, but
metrics and diagnostics must report it separately from labeled GT TP to avoid
inflating recall or hiding annotation-noise assumptions.

Clarified UL supervision semantics:

Cross-rollout clustering is only the admission test for whether an unmatched
object should be promoted to UL-positive supervision. It should not collapse the
members into a single canonical pseudo-GT trajectory for loss computation.

For a promoted UL cluster:

```text
rollout k member bbox
    acts as the bbox target inside rollout k's own correction sequence

cross-rollout agreement
    decides whether the object is eligible to be promoted
```

Thus desc and bbox can both receive positive supervision, but each loss sequence
remains within its own rollout/self-prefix context. The admission test must be
strict: canonical description must match, and the promoted members must be
geometrically close across rollouts under a carefully specified cluster
criterion.

Approved UL cluster admission geometry:

First version should use a strict complete-link style gate rather than average
IoU or connected components:

```text
UL cluster valid iff:
    same canonical desc_id
    one member per valid rollout
    support_ratio == 1.0
    every cross-rollout member pair passes geometry_close()

geometry_close(a, b):
    IoU(a, b) >= tau_iou_min
    and center_distance(a, b) <= tau_center * median_diag(a, b)
    and area_ratio(a, b) <= tau_area_ratio
    and aspect_ratio(a, b) <= tau_aspect_ratio
```

This avoids chain-merging nearby crowded instances where `A` is close to `B`
and `B` is close to `C`, but `A` and `C` are not mutually close enough to be
the same object. Thresholds should be configurable and conservative in the
first version.

Approved UL observability requirement:

Whenever UL clusters are found, accepted, rejected, or quarantined, the pipeline
must record metrics and reviewable artifacts. UL-promoted supervision requires
an audit trail because it introduces positive supervision beyond labeled GT.

At minimum, record per-cluster evidence:

```text
image_id
desc_id / desc_text
support_rollout_ids
support_ratio
member boxes per rollout
pairwise geometry stats
max IoU to labeled same-desc GT
max IoU to emitted matched objects
cluster decision: promoted / rejected / quarantined
decision reason
anchor prefix or object boundary metadata when available
```

The artifact should be usable by later visualization/review tooling to inspect
whether UL-promoted objects are real unlabeled objects, hallucinations, or
duplicate bursts. Metrics should keep UL-promoted positives separate from
labeled GT TP.

Approved UL artifact granularity:

Scalar UL metrics/counters should always be available when UL mining is
enabled. Detailed cluster evidence should be materialized only when monitor
dumps, smoke/debug dumps, or an equivalent artifact flag is enabled, to avoid
unbounded training-output growth.

The first artifact should be JSONL:

```text
ul_clusters.jsonl
    one row per discovered UL cluster
    include promoted, rejected, and quarantined clusters
    include member rollout ids, member boxes, geometry stats, overlap stats,
    decision reason, and anchor/boundary metadata when available
```

Visualization should be a downstream consumer of this JSONL plus image paths,
not an always-on training side effect. This keeps the training loop auditable
without forcing per-step image rendering.

Approved rollout-local promoted-UL residual universe:

```text
G_labeled = dataset GT objects

G_ul(k)
    = UL-promoted objects available for rollout/self-prefix sample k
      using rollout k's own promoted member desc+bbox

G*_k = G_labeled union G_ul(k)

E_t
    = completed emitted objects matched to labeled GT ids
      or rollout-local UL-promoted ids

R_t = G*_k \ E_t
```

UL-promoted positives may be TP-like for the local Stage-2 correction loss, but
they must not silently rewrite the dataset GT. Their ids, metrics, and artifacts
must remain provenance-separated from labeled GT. A future semi-automatic
annotation workflow may export reviewed UL clusters to expand missing COCO
annotations, but that should be a separate explicit pseudo-label/review dataset
pipeline rather than a hidden side effect of online training.

Approved UL lifecycle / namespace:

```text
ul_candidate
    unmatched cross-rollout cluster discovered from K rollouts

ul_promoted_local
    consensus_ratio = 1.0 and strict geometry gate passes
    eligible for the current Stage-2 sample's G*_k and local desc+bbox
    supervision
    not dataset GT
    not labeled-recall evidence

ul_export_candidate
    reviewable artifact row, e.g. ul_clusters.jsonl, suitable for later
    visualization or human/rule review

pseudo_gt_reviewed
    reviewed annotation source that may enter an explicit offline dataset
    expansion pipeline with provenance/manifest support
```

Online Stage-2 training may use `ul_promoted_local` for loss. Default eval
metrics should remain labeled-GT metrics, with UL statistics reported
separately. Only `pseudo_gt_reviewed` may behave like expanded dataset GT in
future processed-data builds.

Approved UL loss weight:

`ul_promoted_local` supervision should have its own configurable provenance
weight rather than being silently equivalent to labeled GT. First-version
default:

```text
lambda_ul_promoted = 0.5
```

Labeled GT remains weight `1.0` by default. The IR/loss layer should preserve
the provenance and effective weight so diagnostics can report labeled-GT and
UL-promoted contributions separately. Ablations may set
`lambda_ul_promoted = 1.0`, but that should be explicit.

Approved UL consensus denominator:

UL consensus should use eligible/valid rollouts as the denominator:

```text
K_requested = requested rollout count
K_valid = rollouts that are generated, parseable, grammar-valid, and eligible
          for UL mining
support_ratio = support_rollouts / K_valid
```

Promotion requires:

```text
K_valid >= min_ul_valid_rollouts
support_rollouts == K_valid
support_ratio == 1.0
```

First-version default should be conservative and configurable:

```text
min_ul_valid_rollouts = 3
```

Invalid or ineligible rollouts should be counted and reported separately by
reason. They should not lower the UL consensus ratio, but they may prevent
promotion if `K_valid` falls below the minimum.

Approved per-rollout UL pre-dedup:

Before cross-rollout UL clustering, each rollout should deduplicate its own
unmatched objects so one rollout contributes at most one vote to a candidate
UL cluster.

First-version rule:

```text
within each rollout:
    group unmatched completed objects by canonical desc_id
    cluster same-desc near-duplicate boxes using strict geometry_close()
    keep one representative per local cluster for UL mining
    suppress the other local members from UL voting

representative:
    earliest object in rollout order
```

The earliest-in-rollout representative keeps the UL vote aligned with the
trajectory's first decision to emit that region. Suppressed same-rollout
members should be recorded as duplicate-like diagnostics and must not increase
cross-rollout support.

Approved repeated-object boundary semantics:

Same-rollout duplicate-like members should not be treated as positives and
should not receive a duplicate-specific loss in the first version. When such a
member identifies the first actionable error in a rollout, it may create a
standard residual-set boundary correction sample:

```text
provenance = repeated_object_boundary
anchor = prefix immediately before the repeated object starts
target = ValidNext(prefix, R_t)
```

The correction sequence is cut before the repeated object is visible to the
loss. The repeated object's desc/bbox are not used as targets. The objective is
positive-only residual-set valid-set marginal: encourage remaining valid object
continuations or EOS. Duplicate branches are simply absent from the valid set.

This is distinct from duplicate unlikelihood. A future duplicate-specific loss
would add an explicit negative term at the same boundary, such as penalizing the
duplicate branch or first-divergence token. That mechanism remains out of scope
for the first redesign and must not be reintroduced implicitly.

Approved first-actionable event selection:

When one rollout contains multiple potential correction events, the first
version should select the earliest actionable anchor in sequence order:

```text
selected_event = candidate event with smallest anchor token/boundary position
```

Do not prioritize by error family, such as FN over duplicate over FP. The
Stage-2 correction target should identify the first point where the self-prefix
deviates from the valid residual-set or selected-object path. This preserves the
principle of cutting the correction sample exactly before the model's wrong
decision.

If multiple provenances explain the same anchor, emit one correction sample with
combined provenance tags rather than multiple competing sequences. Later errors
after the first actionable event should be diagnostics only in the default
first-version mode.

Approved builder ordering:

UL consensus mining should run before per-rollout correction event selection.
First-version order:

```text
1. generate K rollouts
2. filter K_valid grammar-valid eligible rollouts
3. match completed objects to labeled GT with desc-gated geometry assignment
4. collect unmatched completed objects
5. run per-rollout same-desc pre-dedup
6. run cross-rollout complete-link UL consensus
7. build rollout-local G*_k = labeled GT + ul_promoted_local(k)
8. walk each rollout's self-prefix against G*_k
9. select the earliest actionable correction event
10. emit the correction sample
```

This prevents real unlabeled objects from being selected too early as FP
boundary corrections. UL mining is a K-rollout group-level admission step;
correction event selection is a rollout-local residual-state step.

Approved UL effect on continue/EOS:

`ul_promoted_local` objects participate in rollout-local residual-set
continue/EOS decisions after they enter `G*_k`.

```text
R_t = G*_k \ E_t

if R_t contains labeled GT objects:
    valid continuation should beat EOS for labeled remaining objects

if R_t contains only UL-promoted local objects:
    valid continuation should still beat EOS in this local training sample
```

Diagnostics must keep the remaining-object provenance visible:

```text
premature_stop_labeled_remaining
premature_stop_ul_remaining
premature_stop_mixed_remaining
```

The loss may use provenance weights such as `lambda_ul_promoted = 0.5`, but UL
membership in `G*_k` should still affect whether EOS is locally valid.

Approved rollout-local UL emission accounting:

When walking rollout `k`, a completed emitted object may match either a labeled
GT id or a `ul_promoted_local` id that belongs to rollout `k`.

```text
completed emitted object matches ul_promoted_local(k)
    -> enters E_t
    -> is removed from later R_t
```

This prevents the local residual set from asking the model to generate the same
promoted UL object again after it has already been emitted. Cross-rollout
agreement only admits the UL object; it does not replace rollout-local sequence
state. Each rollout should match and account for its own promoted member box in
its own self-prefix walk.

Approved completed-object event precedence:

Completed emitted objects should be classified with mutually exclusive
precedence:

```text
1. labeled_gt match
2. ul_promoted_local match
3. repeated_object_boundary
4. fp_boundary
```

A first occurrence that matches `ul_promoted_local(k)` is a TP-like promoted
positive for rollout `k`; it should enter `E_t` and must not also produce an FP
or repeated-object correction event. A later same-rollout object that overlaps
an already emitted labeled/UL object may become a `repeated_object_boundary`
candidate.

Approved divergence-only trie supervision:

Trie/residual-set valid-set marginal should be used only while the current
prefix still has multiple valid compatible continuations. Once the active
branch is committed to a single labeled GT or rollout-local UL-promoted object,
the inner objective falls back to hard teacher-forcing CE for that selected
path.

```text
ambiguous object boundary / desc / coordinate branch:
    residual-set valid-set marginal

committed labeled GT branch:
    selected-object hard path CE, weight 1.0

committed ul_promoted_local branch:
    selected-object hard path CE, weight lambda_ul_promoted
```

For UL bbox supervision, the target is the promoted member bbox from the same
rollout/self-prefix sequence. Cross-rollout clustering decides admission only;
it does not average or canonicalize bbox targets for loss.

Approved branch commitment rule:

Let `A_t` be the active compatible candidate set under the current object
prefix.

```text
|A_t| > 1:
    branch is ambiguous
    use residual-set/trie valid-set marginal

|A_t| == 1:
    branch is committed
    use selected-object hard teacher-forcing CE for subsequent tokens

|A_t| == 0:
    branch is invalid
    stop this correction path and classify the current position as the
    first actionable error when appropriate
```

Commitment happens as soon as the active prefix uniquely identifies a labeled GT
or rollout-local UL-promoted object. It does not wait for full object
completion. For repeated same-description objects, this often means description
tokens remain ambiguous, `x1` may be a valid-set branch point, and later bbox
slots fall back to hard CE after the coordinate prefix narrows `A_t` to a
singleton.

Approved transition-failure anchor rule:

When consuming a self-prefix token would make the active candidate set empty,
the correction sample should anchor immediately before that invalid token and
target the valid set at that position.

```text
before token:
    A_t is non-empty
    C_t = valid next tokens under current A_t / R_t

consume token:
    A_{t+1} becomes empty

correction:
    anchor = prefix before the token
    target = C_t
```

Examples:

- Wrong description token: anchor before that text token and target valid next
  description tokens.
- Wrong coordinate token: anchor before that coordinate token and target the
  valid coordinate set or selected-object coordinate.
- Wrong schema token inside an otherwise valid object path: anchor before that
  schema/control token and target the expected schema token.

Only object-level branches that should not exist at all, such as FP or repeated
object entry, should anchor at the object boundary before object start. This
keeps token-level path errors local rather than pushing every error back to the
object boundary.

Approved unified correction-event structure:

Transition failures and boundary failures should share one reusable
`CorrectionEvent` abstraction rather than each error family inventing a
separate event type.

Recommended first-version fields:

```text
CorrectionEvent:
    rollout_id
    anchor_token_pos
    anchor_boundary_index optional
    grammar_state
    role
    provenance_tags
    residual_state
    active_candidates_before
    valid_token_ids_before
    observed_bad_token_id optional
    selected_target_id optional
    loss_weight
```

Error-specific meaning should live in provenance tags, for example:

```text
transition_failure
desc_transition_failure
coord_transition_failure
schema_transition_failure
repeated_object_boundary
premature_stop
fp_boundary
ul_promoted_local
```

Loss builders should not rediscover FN, FP, duplicate, or UL semantics. They
should consume objective-facing fields such as role, valid token ids, selected
target id, provenance weight, and anchor position. Provenance remains available
for diagnostics, artifact rows, filtering, and ablations.

Approved event-to-IR layering:

Use a two-layer abstraction:

```text
CorrectionEvent
    Stage-2 residual builder / diagnostics layer
    owns rollout provenance, residual state, UL/repeated/FP/FN labels,
    anchor choice, artifacts, and debug evidence

SupervisionAtom / TeacherForcingTargetIR
    objective-generic loss layer
    owns token role, valid_token_ids, selected_token_id, loss_weight,
    coverage weights, logit_position, and target_position
```

Stage-2-specific algorithms compile `CorrectionEvent` objects into
`SupervisionAtom` objects. Loss modules should consume only
`TeacherForcingTargetIR`; they should not know or reimplement FN, FP, duplicate,
or UL mining semantics. This keeps Stage-1, Stage-2, offline pseudo-label, and
future reviewed pseudo-GT supervision compatible with the same loss abstraction.

Approved next-token logits alignment contract:

Compiling a `CorrectionEvent` into `SupervisionAtom` must use causal LM
next-token alignment:

```text
logit_position = target_position - 1
```

For a teacher-forced sequence:

```text
input_ids = [prompt..., y0, y1, y2, ...]
```

the model logits at `target_position - 1` supervise token `y_t` at
`target_position`. The loss must never use a hidden state that has already seen
the target/bad token to predict that token.

Examples:

```text
anchor before invalid token:
    target_position = absolute input position of the invalid-token slot
    logit_position = target_position - 1

anchor before EOS:
    target_position = EOS slot
    logit_position = token before EOS

anchor before object start:
    target_position = first object-continuation token slot or EOS-decision slot
    logit_position = previous emitted token
```

The IR validator and tests must treat this as a hard invariant:

```text
atom.logit_position + 1 == atom.target_position
target_position points to the supervised token slot
logit_position is inside the non-padding causal context
positions do not cross prompt/assistant boundaries incorrectly
```

This alignment is especially high-risk for Stage-2 self-prefix correction and
must be tested before relying on smoke-run metrics.

Approved corrected-sequence target positions:

`target_position` should refer to the supervised token slot in the
teacher-forced corrected training sequence, not to the raw rollout's bad token
as a target.

```text
observed_bad_token_id
    provenance / diagnostic only

selected_token_id
    token placed in the corrected teacher-forced sequence for roll-in

valid_token_ids
    valid target support at that corrected slot

target_position
    absolute input position of the corrected supervised token slot
```

For transition failures, the raw rollout generated a bad token at the failing
slot, but the training sequence should replace that slot with a selected valid
token from the corrected path. For repeated-object, FP, or premature-stop
events, the raw wrong continuation is cut away and the corrected sequence starts
from the anchor with a valid continuation or EOS. The raw bad token or wrong
continuation may be stored in provenance/artifacts, but the loss should run on
the corrected teacher-forced sequence.

Approved corrected roll-in branch selection:

When a corrected sequence must choose one path through an ambiguous valid set,
the default roll-in policy should sample a valid branch with a deterministic
seed rather than selecting the sorted/canonical first candidate.

```text
loss support:
    all valid next tokens

teacher-forced corrected roll-in:
    one seeded-random valid branch
```

The selected token/path is used only to define the teacher-forced hidden-state
context after the ambiguous position. The atom's valid-token support should
still include all valid alternatives at the ambiguous position.

First-version policy:

```text
rollin_policy = random_valid_branch
base_seed = 17
derived_seed = f(base_seed, sample_id, rollout_id, event_index)
```

A stable-first policy may exist only as a debug/repro fallback, not as the
default training behavior, because it would reintroduce sorted/canonical order
prior into the corrected hidden-state trajectory.

Approved roll-in resampling policy:

Corrected roll-in branch sampling should be deterministic at the correction
event level in the first version:

```text
same base_seed + sample_id + rollout_id + event_index
    -> same selected corrected branch
```

Default:

```text
rollin_resample_policy = fixed_event
```

This keeps unit tests, artifact review, teacher-forcing alignment checks, and
smoke-run comparisons reproducible. Future ablations may add `per_epoch` or
`per_step` resampling as data augmentation, but those should not be the default
while the residual-set and UL mining semantics are being validated.

Approved local correction span default:

In the first version, one `CorrectionEvent` should compile to one local
correction span, not automatically to the entire corrected suffix.

Default span policy:

```text
schema / boundary / description transition:
    one event -> one primary SupervisionAtom

coordinate transition:
    one event -> bbox_tail_from_anchor
```

For coordinate correction, the bbox group is a local structured unit. Once a
coordinate correction is anchored, supervise the remaining coordinate slots in
that bbox:

```text
anchor before x1:
    supervise x1, y1, x2, y2

anchor before y1:
    supervise y1, x2, y2

anchor before x2:
    supervise x2, y2

anchor before y2:
    supervise y2
```

This preserves precise anchoring while preventing Stage-2 from only repairing
the earliest coordinate slot, such as repeatedly fixing `x1` while leaving
`y1/x2/y2` under-corrected. The span should not extend beyond the current bbox
or into the next object unless a future explicit mode enables that behavior.

Within the coordinate span, the normal divergence/commitment rule still applies:

```text
ambiguous coordinate slot:
    residual-set valid-set marginal

after A_t becomes singleton:
    selected-object hard teacher-forcing CE
```

For matched labeled GT, the coordinate hard-CE atoms use GT bbox slots. For
`ul_promoted_local`, they use the promoted member bbox from the same rollout and
the configured UL loss weight. Future ablations may explicitly test broader
spans such as:

```text
span_policy = first_object
span_policy = until_commit
span_policy = full_corrected_suffix
```

but these should not be first-version defaults.

Approved dynamic commitment within coordinate spans:

Within `bbox_tail_from_anchor`, each coordinate slot should be supervised
according to the current active candidate set after the corrected roll-in tokens
chosen so far:

```text
|A_t| > 1:
    coordinate slot uses residual-set valid-set marginal

|A_t| == 1:
    coordinate slot uses selected-object hard CE

|A_t| == 0:
    invalid corrected roll-in state; builder should reject or regenerate
```

The first coordinate slot does not force the entire remaining tail to be hard
CE unless it actually commits the branch. If repeated same-description objects
share `x1`, `y1` may remain a valid-set branch point. Once a corrected roll-in
token narrows `A_t` to a singleton, all later slots in the bbox tail use that
selected object's coordinate targets.

Approved residual-state-machine action abstraction:

`ValidNext` and state transition should not be implemented as separate pieces
that must be manually kept in sync. They should be two views of one
residual-state machine.

Recommended abstraction:

```text
ResidualStateMachine.valid_actions(state) -> list[ValidAction]

ValidAction:
    token_id
    role
    next_state
    candidate_subset_after
    selected_object_id optional
    provenance optional
```

Then:

```text
valid_token_ids = {action.token_id for action in valid_actions}
selected_action = seeded_sample(valid_actions)
selected_token_id = selected_action.token_id
next_state = selected_action.next_state
```

Corrected roll-in should sample only from `ValidAction` records whose
`next_state` is already transition-validated. Therefore, a corrected roll-in
that makes `A_t` empty is impossible by construction, not a normal runtime case.

Defensive guardrails are still allowed:

```text
strict mode:
    raise on invalid action / empty next_state

non-strict smoke/debug mode:
    drop sample, increment invalid_builder_state, and dump compact diagnostics
```

The guardrail is for catching implementation bugs or artifact corruption. It is
not part of the intended algorithmic path.

Approved token-level `ValidAction` scope:

First-version `ValidAction` should represent one valid next token, not a
higher-level object action.

Recommended fields:

```text
ValidAction:
    token_id
    token_role
    next_state
    candidate_subset_after
    selected_object_id optional
    object_provenance optional
    coord_slot optional
    action_tags optional
```

Object-level semantics such as branch commitment, labeled-vs-UL provenance,
EOS, repeated-object diagnostics, and candidate filtering should be represented
through `next_state` and metadata/provenance. Keeping `ValidAction` token-level
keeps it directly compatible with `SupervisionAtom.valid_token_ids`,
`selected_token_id`, and next-token teacher-forcing alignment.

Approved EOS/stop as token-level action:

EOS/stop should also be represented as an ordinary token-level `ValidAction`
rather than a separate special branch.

```text
if R_t is empty at an object boundary:
    valid_actions = [STOP action]

if R_t is non-empty:
    valid_actions = valid object-continuation actions
    STOP is not valid
```

The STOP action should use the existing `TokenRole.STOP` role and the configured
Qwen3-VL stop token:

```text
token_id = <|im_end|>
token_role = STOP
next_state = terminal
action_tags = {eos, true_stop}
```

The padding token `<|endoftext|>` must not be used as EOS/stop semantics. A raw
rollout that emits `<|im_end|>` while `R_t` is non-empty is a premature-stop
correction event whose valid actions are object-continuation tokens, not STOP.
Continuation-vs-EOS diagnostics or optional margins are extra measurements or
losses; they should not change the core token-level action contract.

Approved STOP/continuation exclusivity:

In the core residual-state-machine target, STOP and object-continuation actions
are mutually exclusive at object boundaries.

```text
R_t is empty:
    valid_actions = {STOP}

R_t is non-empty:
    valid_actions = {valid object-continuation tokens}
    STOP is absent
```

Do not include STOP and continuation in the same core valid set and rely on the
loss to learn a tradeoff. EOS/continue calibration may be measured or added as
an optional margin, but the canonical valid-set target should remain:

```text
remaining objects exist -> continue
no remaining objects -> STOP
```

This keeps `premature_stop` well-defined as a raw STOP token emitted while STOP
is not a valid action.

Approved continuation-vs-STOP margin default:

Continuation-vs-STOP margin should be optional and disabled by default in the
first version:

```text
lambda_continue_margin = 0.0
```

The canonical objective should first validate residual-set valid actions,
STOP/continuation exclusivity, rollout-local UL residual sets, and earliest
anchor correction without adding a trainable STOP margin. This follows prior
evidence that simply loosening or biasing EOS behavior can harm decoding.

Diagnostics should still report:

```text
log P(valid_continue)
log P(STOP)
valid_vs_stop_margin
premature_stop_labeled_remaining
premature_stop_ul_remaining
premature_stop_mixed_remaining
```

Ablations may enable:

```text
lambda_continue_margin > 0
continue_margin_m = configurable
```

The margin, when enabled, is an extra term. It must not change the core
`valid_actions` contract.

## Rationale

This decision rejects the weaker interpretation that Stage-2 should only add FN
description categories earlier into the edited target. A false negative is not
primarily an object to append. It is evidence that the residual set remains
non-empty under the current self-prefix.

The stronger rule is:

```text
At any grammar-valid self-prefix with a well-defined residual state,
the valid target support is induced by the remaining GT object set.
```

Consequences by rollout outcome:

- Matched TP object repair: once the active object branch is committed, bbox and
  object-internal tokens use selected-object hard-path likelihood.
- Ambiguous description or coordinate branch: if multiple objects in `R_t`
  remain compatible with `h_t`, train valid-set marginal over their legal next
  tokens.
- FN or under-generation: if the model attempts EOS while `R_t` is non-empty,
  train valid continuation from `R_t` and optionally measure or ablate a
  continuation-vs-EOS margin.
- True stop: if `R_t` is empty, the valid target is EOS/stop.
- Duplicate or FP: already emitted matched objects are excluded from `R_t`.
  Their future desc/coord branches are therefore invalid without requiring a
  separate duplicate-aware or FP-aware loss family.

The design keeps trie-like construction, but changes its meaning. The required
primitive is a residual-set valid-set builder:

```text
residual_set_valid_set = ValidNext(h_t, R_t)
```

The canonical objective should not be called `trie-multiple-target` when that
term means merging anchor/explorer/repaired candidates from K rollouts. That
candidate-group semantics is a legacy/comparator interpretation, not the new
Stage-2 canonical math.

Likewise, the canonical objective should not be called `FN-aware`. FN, FP,
duplicate, and matched-TP are builder provenance and diagnostics. The loss
primitive is residual-set valid-set marginal plus optional remaining-aware
continue/stop calibration.

## Consequence

The Stage-2 redesign should separate three layers:

```text
CorrectionEvent
    provenance from rollout parsing/matching:
    matched_tp, fn_continue, duplicate_boundary, fp_boundary, true_stop, etc.

ResidualSetState
    emitted matched GT ids, remaining GT ids, active compatible candidates,
    and grammar-valid self-prefix position.

SupervisionAtom
    objective-facing target atom with token role, valid_token_ids,
    selected_token_id for roll-in/control-flow, optional coverage weights,
    loss weight, and provenance.
```

Loss modules should not rediscover FN/FP/duplicate semantics. They should only
consume atom-local fields such as allowed token roles, valid token ids, selected
token id, coverage weights, and loss weight.

Terminology for the new Stage-2 algorithm discussion:

- Use `self-prefix correction sample`.
- Use `residual set` for remaining GT object state.
- Use `residual-set valid-set marginal` for `-log P(C_t | h_t)`.
- Use `remaining-aware continue/stop` for EOS/continuation behavior.
- Use `selected-object hard path CE` for committed object internals.
- Do not use `trie-multiple-target` as a canonical Stage-2 objective name.
- Do not use `FN-aware loss` as a canonical objective name.

Approved first-version eligibility boundary:

```text
self-prefix correction sample
    = grammar-valid compact_full prefix
    + completed emitted objects
    + at most one active object prefix with known slot/state
```

The active state must be uniquely attributable to an object boundary,
description continuation, box start, one bbox coordinate slot, or EOS boundary.
Malformed schema, unparseable object boundaries, broken coordinate arity,
unknown token-role state, prompt/template mismatches, and prefixes whose active
object state cannot be uniquely determined should be dropped by the runtime
guardian or handled by a future separate format-repair route. They should not be
mixed into the residual-set objective, because `R_t` and `C_t` would no longer
have exact semantics.

This is a planning decision, not current runtime behavior. It intentionally
differs from the current Stage-2 AB contract, which still describes edited
anchor targets, false-negative insertion, and one merged teacher-forced forward.
If adopted as stable behavior, the Stage-2 OpenSpec and runbook must be updated
or superseded.

## Evidence

- Scope: `none-yet`
- Discussion handles:
  - `docs/AGENT_INDEX.md`
  - `docs/catalog.yaml`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `openspec/specs/stage2-ab-training/spec.md`
  - `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
- Current implementation handles inspected during the discussion:
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/target_builder.py`
  - `src/trainers/stage2_two_channel/trie_supervision.py`
  - `src/trainers/teacher_forcing/modules/stage2_trie_ce.py`
  - `src/trainers/stage2_two_channel/teacher_forcing_adapter.py`
  - `src/training/teacher_forcing/ir.py`
  - `src/training/teacher_forcing/probabilities.py`
  - `src/training/teacher_forcing/validation.py`
- Current behavior conflict to resolve before implementation:
  - `openspec/specs/stage2-ab-training/spec.md` currently states that Channel-B
    builds the final positive target from the anchor clean sequence and uses one
    merged teacher-forced forward over the edited anchor target.

## Open Follow-Up Questions

The new grilling loop should resolve, in order:

1. what counts as a grammar-valid self-prefix eligible for residual-set
   correction versus malformed-prefix schema repair or sample drop;
2. how to identify `E_t` when a partially emitted object is currently active
   but not yet completed;
3. whether object completion removes a candidate from `R_t` only after full bbox
   completion, or whether earlier commitment should create a separate active
   candidate state;
4. how to handle prefix states where `<|object_ref_start|>` is legal but the
   first differentiating error occurs later in description or coordinate slots;
5. which diagnostics and smoke ablations are sufficient before promoting this
   from progress decision to OpenSpec contract.
