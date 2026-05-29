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

Superseded local coordinate-repair span:

The earlier `bbox_tail_from_anchor` plan is deleted for Stage-2 v1. It would
have anchored correction inside raw-rollout coordinate slots, but that conflicts
with the later decision to treat bbox geometry as a commitment gate rather than
a coordinate-repair target.

Do not implement local coordinate repair/refinement in the first version:

```text
no raw-rollout x1/y1/x2/y2 correction anchor
no bbox_tail_from_anchor correction event
no nearest-GT coordinate repair
no invalid-bbox coordinate repair
no low-IoU coordinate refine
```

Stage-2 v1 coordinate supervision exists only inside constructed teacher-forced
continuations:

```text
constructed correction suffix:
    selected remaining object bbox tokens

optional clean GT stabilizer stream:
    same typed-trie / singleton coord targets if enabled
```

Bbox geometry is a commitment gate, not a coordinate-repair target:

```text
legal + desc exact + IoU >= 0.75:
    commit object and update remaining set

otherwise:
    uncommitted dirty prefix
    semantic remaining unchanged
    train the next valid boundary/continuation from the residual set
```

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

## 2026-05-21 Dirty-Prefix Recovery / K-Rollout UL Mining Update

This section records the follow-up grilling loop after the Stage-2 smoke run
against the `et-rmp-ce-ckpt-3660+` / `checkpoint-3664` base. It supersedes the
earlier first-version assumption that Stage-2 should keep only one first
actionable correction event per rollout and only grammar-clean prefixes.

The updated principle is:

```text
Stage-2 correction uses dirty textual prefixes with clean semantic residual state.

textual prefix:
    preserve the model's actual rollout token history when boundaries are reliable

semantic residual state:
    only committed supervision objects update emitted / remaining state

loss:
    do not imitate rollout-prefix tokens;
    attach expert correction atoms at eligible next-token positions
```

K rollout terminology is now:

```text
K rollouts = K equal rollout_attempts
```

Canonical terms:

```text
rollout_attempt
rollout_id
decode_mode
generation_config / sampling_seed
```

Deprecated / legacy terms:

```text
anchor
explorer
anchor_rollout
explorer_rollout
```

The new IR should not use anchor/explorer naming. Legacy fields may be read only
by an adapter that converts them into equal `rollout_attempts[]`.

OpenSpec and implementation plans for the new path should fully remove
anchor/explorer as concepts. The only allowed reference is in offline legacy
artifact adapters or migration notes:

```text
legacy anchor/explorer fields -> rollout_attempts[]
```

Canonical runtime/config should not expose anchor/explorer options, metrics, or
target-builder concepts. New training runtime must emit `rollout_attempts[]`
directly. If training runtime still emits anchor/explorer fields, treat it as
incomplete migration. The legacy adapter is offline-only.

Default attempt generation for the first version:

```text
K = 4
attempts:
    1 greedy rollout_attempt
    3 sampling rollout_attempts

all attempts:
    equal semantic status
    same generation_config by default
    seed variation for sampling attempts
```

Greedy is just `decode_mode=greedy`, not an anchor.

Attempt metadata should include:

```text
rollout_id
decode_mode
sampling_seed optional
generation_config_hash
```

Diagnostics should preserve decode-mode slices so sampling noise and useful
exposure-bias states are separable:

```text
clean_success_rate_by_decode_mode
invalid_rate_by_decode_mode
promoted_ul_support_by_decode_mode
dirty_correction_events_by_decode_mode
```

Stage-2 v1 should use offline prepared rollout JSONL, not online generation
inside the training loop:

```text
1. rollout preparation job:
       model checkpoint -> K rollout_attempts per sample
       store response_token_ids, raw_text, decode_mode, seed, config hash

2. stage2 target builder / training:
       read prepared JSONL
       parse rows, dedup attempts, promote UL, assemble target IR
       train correction objective

3. monitor / review:
       write compact step monitors and ul_clusters.jsonl
```

This is round-based offline DAgger-like correction data. It preserves the core
idea that each round trains on self-prefix states from the current or recent
policy, but keeps generation out of the training loop for reproducibility.
Future refresh can be explicit and round-based:

```text
round 0: generate rollouts from checkpoint A
round 1: train correction checkpoint B
round 2: regenerate rollouts from checkpoint B
```

The prepared rollout record must carry enough information for target IR replay:

```text
response_token_ids required for new prepared data
raw decoded text
decode_mode
sampling_seed optional
generation_config_hash
sample/image provenance
```

It does not need to store absolute assistant spans from the rollout-time
processor. Those positions depend on the current tokenizer/processor/chat
template and should be recomputed during training assembly:

```text
not required:
    rollout-time absolute assistant_start
    rollout-time full input span

optional debug fields:
    response_token_count
    raw_text_sha256
    generation_prompt_hash
    tokenizer_name_or_hash
```

Training assembly must reconstruct prompt + assistant context in the current
environment and validate:

```text
assistant response span is located
response_token_ids are the intended assistant prefix
logit_position is valid
target_position = logit_position + 1 when target_position is represented
positions do not cross prompt / assistant boundaries
```

Do not implement fully online generate-while-training in the first version.
Offline rounds keep dirty-prefix parsing, UL consensus, logits-position checks,
and ablations replayable.

Raw text is required for diagnostics, span recovery, and human review, but it is
not the canonical training prefix. New prepared rollout JSONL should fail/drop
when `response_token_ids` are missing:

```text
strict new-data mode:
    missing response_token_ids -> drop sample + diagnostic

explicit legacy fallback mode:
    re-encode raw_text
    dirty_prefix_reencoded = 1
    allow smoke / legacy ablation only
```

This protects the causal logits-position contract from tokenizer drift,
special-token handling differences, and legacy newline re-encoding changes.

### Implementation Abstraction

Do not implement the redesign as a long case-by-case target builder. Use a
layered pipeline where each layer owns one semantic question:

```text
1. Row Segmentation
2. Row Classification
3. Semantic State Scan
4. Correction Atom Extraction
5. Target Sequence Assembly
```

Target Sequence Assembly must use a template boundary adapter rather than
hand-authored schema strings:

```text
TemplateBoundaryAdapter:
    resolve the Stage-1 assistant template / serialization policy
    render full assistant text for chosen supervision objects
    tokenize with the same tokenizer / processor contract
    expose object spans, separator spans, terminal/stop spans
    slice the constructed suffix from a requested boundary state
    validate that prefix + suffix does not duplicate deterministic schema
```

Stage-2 may choose the remaining supervision objects, object order, suffix-start
state, and correction atoms. It must not manually author schema fragments such
as newline, `<|object_ref_start|>`, `<|box_start|>`, or `<|im_end|>` by string
concatenation outside the adapter.

The core idea is that special cases such as invalid bbox, FP, duplicate burst,
wrong description, early EOS, and truncation should fall out of these reusable
layers rather than being independently reimplemented in loss code.

#### 1. Row Segmentation

Input:

```text
prompt_ids
rollout_attempt response_token_ids
rollout decoded text for span recovery / diagnostics
compact_full grammar markers
```

Output:

```text
RowSpan[]
```

`RowSpan` should describe token-level boundaries, not supervision meaning:

```text
row_index
start_token
end_token optional
boundary_reliable
is_complete_row
is_middle_malformed_resynced
is_trailing_incomplete
raw_text_span
desc_span optional
box_start_span optional
coord_spans optional
next_row_start optional
```

Segmentation policy:

```text
complete row:
    can be scanned and may become committed or uncommitted

middle malformed span with later reliable <|object_ref_start|>:
    can remain dirty context and allow suffix resync

trailing incomplete object:
    drop from its <|object_ref_start|> and return to the last stable boundary

unresynchronizable suffix:
    diagnostics-only after the last stable boundary
```

This layer must not inspect GT/UL matching. It only answers whether object-row
boundaries are reliable.

The row segmentation layer is also the only layer that should directly depend
on raw rollout text for boundary recovery. Downstream layers should consume
token spans and structured row observations rather than re-splitting text.

#### 2. Row Classification

Input:

```text
RowSpan
earlier legal predictions in the same rollout_attempt
```

Output:

```text
RowObservation
```

`RowObservation` should describe row-local facts:

```text
structural_status:
    complete
    malformed_resynced
    trailing_incomplete
    unparseable

desc_status:
    normalized_desc
    invalid_desc
    missing_desc

geometry_status:
    legal_bbox
    invalid_bbox
    missing_bbox
    wrong_coord_arity

duplicate_status:
    duplicate_burst
    not_duplicate
```

Duplicate burst is row-local / rollout-local:

```text
duplicate if:
    current row is legal positive-area bbox
    earlier row is legal positive-area bbox
    normalized desc matches
    pred-vs-pred IoU >= 0.95
```

This layer must guard all IoU computations:

```text
IoU is defined only for legal positive-area bbox:
    x1 < x2
    y1 < y2
```

Invalid bbox rows bypass IoU-based modules and only contribute dirty context and
diagnostics.

#### 3. Semantic State Scan

Input:

```text
RowObservation stream
remaining supervision set = GT + promoted UL
```

Output:

```text
ResidualScanState at each stable boundary
ScanDecision for each row
```

This is the only layer that can update emitted / remaining state:

```text
try_commit(row, remaining):
    if row is not structurally complete:
        return uncommitted
    if row bbox is not legal positive-area:
        return uncommitted
    if row is duplicate_burst:
        return uncommitted

    candidates =
        remaining objects with
        normalized_desc match
        and IoU(row.bbox, object.bbox) >= 0.75

    if candidates empty:
        return uncommitted

    return committed(best candidate by IoU desc, center distance asc, stable id asc)
```

Committed rows:

```text
remove exactly one GT or promoted UL id from remaining
do not rewrite rollout tokens
do not create loss on rollout-prefix tokens
```

Uncommitted rows:

```text
remaining unchanged
dirty_context becomes true after the row is in prefix
row tokens remain masked
```

This layer unifies invalid geometry, legal low-IoU rows, FP, duplicate burst,
wrong-desc overlap, and failed UL candidates as commitment failures with
different provenance.

Dataset-description vocabulary is a required dataset contract for this layer:

```text
dataset_desc_vocab:
    supplied by dataset adapter / view metadata / resolved config
    not hard-coded in Stage-2 target builder
    unavailable -> fail fast
```

GT supervision objects must use descriptions in `dataset_desc_vocab`. A GT desc
outside the vocab is a hard data-contract error, not an object-drop condition.

Rollout descriptions outside the vocab are model-output errors:

```text
rollout desc outside dataset_desc_vocab:
    uncommitted dirty prefix if boundary reliable
    no commitment
    no pending UL candidacy
    diagnostic: rollout_desc_out_of_vocab
```

Promoted-UL clusters outside the vocab are rejected, not hard errors:

```text
status = rejected_desc_out_of_vocab
```

Diagnostics should record:

```text
dataset_desc_vocab_id
dataset_desc_vocab_size
dataset_desc_vocab_hash
```

Do not dump the full vocab list into every monitor sample by default.

`dataset_desc_vocab_id` should come from dataset/view metadata. Stage-2 target
building should not invent or hard-code dataset names. If an id is missing but
the vocab/hash exists, diagnostics may use a deterministic fallback id:

```text
dataset_desc_vocab_id = "desc_vocab:<hash_prefix>"
```

The fallback id is diagnostic-only.

Within one run, the mapping must be stable:

```text
dataset_desc_vocab_id -> dataset_desc_vocab_hash
```

If the same id appears with different hashes in one run, fail fast.

First version requires one desc vocab per batch/run. Mixed-vocab batches are not
supported for this Stage-2 objective:

```text
multiple dataset_desc_vocab_hash values in one batch/run:
    fail fast
```

Promoted UL uses the same dataset desc vocabulary contract as GT; no separate UL
vocab exists in v1.

Vocab contract should be logged once at run level / resolved config level:

```text
dataset_desc_vocab_id
dataset_desc_vocab_size
dataset_desc_vocab_hash
```

Per-sample monitor payloads should only include vocab fields for hard errors,
rejected clusters, or diagnostic exceptions.

`dataset_desc_vocab_hash` is defined as:

```text
vocab_items =
    sorted(normalize_desc(desc) for desc in dataset_desc_vocab)

payload =
    json.dumps(vocab_items, ensure_ascii=False, separators=(",", ":"))

dataset_desc_vocab_hash =
    sha256(payload.encode("utf-8")).hexdigest()
```

The hash is order-insensitive with respect to the source vocab container and is
based on normalized canonical desc strings.

Normalized-desc collisions are hard errors:

```text
if len(set(normalized_vocab)) != len(raw_vocab):
    fail fast
```

Example:

```text
"potted  plant" and "potted plant"
    -> both normalize to "potted plant"
```

This would break exact canonical mapping.

Description normalization is also used before tokenizing supervision descs:

```text
canonical_desc = normalize_desc(raw_desc)
desc_token_ids = tokenizer.encode(canonical_desc, add_special_tokens=False)
```

Constructed suffix rendering must use `canonical_desc`, not raw dataset
whitespace.

Rollout desc normalization:

```text
normalize_desc(rollout_desc) == "":
    desc_status = invalid_desc_empty
    row uncommitted
    not pending UL
    dirty prefix if boundary reliable
```

For hard errors and rejected clusters, diagnostics should include:

```text
sample_id
image_id
object_id or cluster_id
raw_desc
normalized_desc
dataset_desc_vocab_id
dataset_desc_vocab_hash
```

Normal rows should not dump raw/normalized desc pairs by default.

#### 4. Correction Atom Extraction

Input:

```text
RowSpan / RowObservation stream
ResidualScanState before and after stable boundaries
constructed-suffix trie state
```

Output:

```text
SupervisionAtom[]
```

Universal rule:

```text
At each eligible next-token decision point:
    expert target = ValidNext(prefix_state, current remaining supervision set)
```

The actual rollout token may be wrong. The correction atom supervises the expert
next-token distribution at that causal position; it does not imitate the actual
token.

Event examples under this rule:

```text
remaining nonempty + model stops:
    target valid continuation

remaining empty + model continues:
    target EOS singleton

wrong-desc row:
    target earliest valid description-trie child before the divergent desc token
    unless it is a spatially matched label-conflict case, which uses reduced
    weight and separate provenance

legal FP / low-IoU / invalid geometry / duplicate:
    row is uncommitted, remaining unchanged, later boundary target uses same
    remaining set
```

First-version bbox policy:

```text
do not create bbox-coordinate internal repair atoms
do not correct x1/y1/x2/y2 divergence in rollout prefix
do not use nearest-GT bbox repair
```

Description divergence is allowed because it is an object identity choice.
Bbox coordinate divergence is not corrected in the first version; bbox quality
only affects commitment.

Correction atom extraction must not redo upstream semantics:

```text
do not reparse raw rollout text
do not redo bbox legality parsing
do not redo duplicate detection
do not redo GT/UL matching
do not recompute remaining set
do not run UL promotion
```

It consumes:

```text
RowSpan
RowObservation
ScanDecision
ResidualScanState
remaining supervision objects exposed by scan state
constructed suffix trie state
```

Wrong-description rows may get description divergence correction atoms:

```text
remaining = {person, horse}
actual row = <|object_ref_start|>dog...

position:
    token immediately before the first divergent desc token
role:
    desc
token_type:
    text
valid_token_ids:
    description-trie children for current remaining set
```

However, wrong-desc rows are split by localization evidence:

```text
ordinary wrong-desc:
    no desc-agnostic spatial match to a remaining supervision object
    may use normal context weight

spatial_wrong_desc_conflict:
    legal bbox
    desc does not match
    desc-agnostic IoU with a remaining GT / promoted UL >= commit_iou_threshold
    may create the same earliest-divergence desc atom
    but with reduced label-conflict weight
    row remains uncommitted
    remaining state is unchanged
    record review diagnostics
```

Rationale: a high-IoU wrong-description prediction may indicate annotation noise
or a dataset category disagreement rather than a pure model error. It should not
receive the same supervision strength as a clear wrong-desc hallucination.

Default label-conflict weighting:

```text
label_conflict_weight = 0.25

ordinary wrong-desc:
    weight = context/source weight

spatial_wrong_desc_conflict:
    weight = context/source weight * label_conflict_weight
```

This weight is separate from `fallback_loss_weight`; fallback expresses dirty
context confidence, while `label_conflict_weight` expresses possible annotation
or category-noise uncertainty.

Diagnostics:

```text
spatial_wrong_desc_conflict_count
spatial_wrong_desc_conflict_loss
spatial_wrong_desc_conflict_pred_desc
spatial_wrong_desc_conflict_nearest_gt_desc
spatial_wrong_desc_conflict_iou
```

Spatial wrong-description conflicts are never UL candidates in v1:

```text
if legal proposal has high IoU with any GT/promoted UL but desc differs:
    classify as spatial_wrong_desc_conflict
    do not add to pending UL candidates
    do not promote to UL even with cross-rollout consensus
    keep review diagnostics only
```

This avoids creating two supervised labels for the same spatial object. Future
semi-automatic dataset repair may export these cases as a label-conflict review
surface, but they should not enter the Stage-2 residual supervision set.

They also do not update emitted / remaining state:

```text
spatial_wrong_desc_conflict:
    desc correction atom weight *= label_conflict_weight
    row remains uncommitted
    remaining supervision set unchanged
```

Do not introduce a third "spatially emitted but semantically uncommitted" state
in v1.

Do not create a separate `label_conflict_review.jsonl` by default. Keep these
cases in compact diagnostics:

```text
monitor_dumps/step_*.json:
    spatial_wrong_desc_conflict_count
    capped examples for review

ul_clusters.jsonl:
    include related GT conflict info only when a UL candidate/cluster is rejected
    because of cross-desc or near-GT conflict
```

A dedicated label-conflict review artifact can be added later if these cases
become a primary semi-automatic dataset repair workflow.

If the row description is compatible but the bbox is invalid or IoU-low, do not
create a coordinate or box-start correction atom in the first version. The row
is uncommitted and later boundary targets are derived from the unchanged
remaining set.

Every atom uses causal next-token indexing:

```text
logit_position = prefix_last_token_index
logits at token i predict token i+1
```

#### 5. Target Sequence Assembly

Input:

```text
prompt_ids
kept raw rollout prefix ids
remaining supervision set at suffix start
SupervisionAtom[]
```

Output:

```text
training input_ids
labels / masks
canonical atom registry
diagnostics
```

Assembly rules:

```text
input_ids =
    prompt_ids
    + raw rollout response_token_ids up to the kept stable prefix
    + constructed continuation suffix ids

rollout prefix labels:
    -100

constructed suffix:
    all remaining supervision objects
    random order with seed 17-derived deterministic RNG
    GT and promoted UL shuffled together
    EOS appended if it fits
```

Do not truncate rollout prefix in the first version. If prompt plus raw rollout
prefix plus constructed suffix exceeds length budget, drop the sample and record
overlength diagnostics. Constructed suffix may be truncated only at complete
object boundaries; never include a partial target object.

All supervision atoms enter one canonical registry:

```text
key = logit_position
same position + same target:
    merge provenance
same position + conflicting target:
    record conflict diagnostic
    deterministic keep one canonical atom
```

Default correction sample granularity:

```text
one rollout_attempt -> one training sequence
```

Within that sequence, collect all eligible non-conflicting correction atoms:

```text
before uncommitted rows
after dirty rows when recovery is valid
before early EOS
before first over-generation row when remaining is empty
constructed suffix positions
```

Do not split a rollout_attempt into one sample per correction event in v1. If no
active atom remains after filtering/merging, skip the sample and record the
reason. This keeps K rollouts as K self-prefix contexts while allowing each
context to expose multiple next-token correction points.

Each `SupervisionAtom` owns both token-type and inner valid-set targets:

```text
logit_position
target_position optional
token_type
role
valid_token_ids
selected_token_id optional
weight
target_kind
provenance list
```

Do not create separate independently mergeable type atoms and inner atoms.

`logit_position` is the canonical runtime field consumed by the loss module.
Older prose may use `position_index` as a synonym, but implementation should not
introduce a separate peer field with that name. `target_position` is for
validation and debugging:

```text
if target_position is not None:
    logit_position + 1 == target_position
```

`SupervisionAtom` contract:

```text
token_type:
    schema
    text
    coord

role:
    object_boundary
    desc
    box_start
    x1
    y1
    x2
    y2
    eos

target_kind:
    valid_set
    singleton
    eos
```

`role=object_boundary` is used for continue decisions:

```text
remaining nonempty:
    role = object_boundary
    valid_token_ids = {<|object_ref_start|>}
```

`role=eos` is used for stop decisions:

```text
remaining empty:
    role = eos
    valid_token_ids = {<|im_end|>}
```

Do not add a separate `object_ref_start` role.

`selected_token_id` is optional. It is used for constructed-suffix roll-in,
validation, hard-singleton consistency, and diagnostics. The loss computation
uses `valid_token_ids`.

`weight` is the final scalar event weight exposed to the loss module. Upstream
target builders combine source/context factors before creating the atom:

```text
GT clean:
    1.0

GT dirty:
    fallback_loss_weight

UL clean:
    0.5

UL dirty:
    0.5 * fallback_loss_weight
```

Loss modules must not recompute these components.

Atom weight scales the full atom loss:

```text
L_atom =
    atom.weight * (
        lambda_type  * L_type
      + lambda_inner * L_inner
    )
```

The type/inner tradeoff is global; atoms do not carry separate type and inner
weights.

Stage-2 v1 keeps the standalone token-type loss enabled by default. Do not turn
off type exclusivity to tolerate ambiguous or malformed segments:

```text
lambda_type = 1.0
lambda_inner = 1.0
```

Principle:

```text
prefer dropping or resynchronizing malformed segments
over weakening schema/text/coord type supervision
```

Type loss owns global schema/text/coord mass. Inner valid-set loss consumes
conditional within-type targets and must not own or recompute the type loss.

Loss computation should be unified through valid-token marginal:

```text
L_inner = -log sum_{v in valid_token_ids} p(v)
```

Hard CE and EOS are singleton valid-set special cases. `target_kind` is for
validation, diagnostics, and metric grouping.

When a sequence contains multiple correction atoms, normalize within the
rollout_attempt sequence by atom weights:

```text
sequence_loss =
    sum_i atom_weight_i * atom_loss_i
    / max(eps, sum_i atom_weight_i)

batch_loss =
    mean(sequence_loss over retained rollout_attempt sequences)
```

This keeps all eligible correction atoms active without letting unusually dirty
or long rollout attempts dominate simply because they contain more atoms. Still
report atom-level diagnostics:

```text
active_atom_count
atom_weight_sum
raw_atom_loss_sum
sequence_loss
```

Do not add a second clean/dirty/UL bucket normalization layer in v1. Clean,
dirty, GT, UL, EOS, and continuation differences are expressed by atom weights;
they are grouped only for diagnostics:

```text
loss_clean_atoms
loss_dirty_atoms
loss_ul_atoms
loss_eos_atoms
atom_count_by_provenance
atom_weight_sum_by_provenance
```

Loss modules are row-category agnostic. They consume only:

```text
logits
SupervisionAtom.logit_position
SupervisionAtom.token_type
SupervisionAtom.valid_token_ids
SupervisionAtom.weight
token type groups
global loss coefficients
```

They must not know about invalid geometry, FP, duplicate burst, pending UL,
promoted UL, dirty prefixes, early EOS, or truncation. Those categories remain
in provenance and diagnostics only.

Description trie construction:

```text
desc_token_ids = tokenizer.encode(canonical_desc, add_special_tokens=False)
```

The canonical desc string is normalized only with:

```text
strip
collapse repeated whitespace to one space
```

All description target token ids must belong to the text token group. If a GT or
promoted-UL supervision object's description tokenization contains schema,
coord, EOS, PAD, or disallowed special tokens, record
`invalid_description_tokenization` and reject that supervision object; if no
supervision objects remain, drop the sample. COCO should not hit this path.

For COCO first version, promoted UL descriptions must be in
`dataset_desc_vocab`. The promoted UL desc is the exact normalized cluster desc;
all members must share it exactly.

Valid-token sets are sets, not multisets:

```text
duplicate text/coord token ids:
    deduplicate
    no multiplicity weighting
```

### Universal Principles

The implementation should be organized around these principles:

```text
Preserve observed state, supervise expert action.
    Rollout tokens are context, not imitation labels.

Only committed rows change semantic state.
    Committed = legal bbox + exact desc + IoU >= 0.75 + not duplicate burst.

Expert target is always a function of current remaining supervision set.
    remaining nonempty -> valid continuation
    remaining empty -> EOS

Dirty context changes weight, not target semantics.
    clean context -> weight 1.0
    dirty context -> fallback_loss_weight

Bbox errors are commitment failures, not coordinate-repair targets.
    No bbox internal repair in first version.

One logits position has one canonical supervision atom.
    Merge duplicates, diagnose conflicts.
```

### Evidence From Current Smoke Artifacts

Evidence scope is smoke / monitor-dump only, not a full training-distribution
claim.

Primary artifact:

```text
output/stage2_ab/smoke/coco80_view_stage2_trie_ce_overfit_probe/
  tail_append_train8_noeval_64steps/
  smoke_64steps-coco80_view-overfit_train8-noeval-stage2_trie_ce-tail_append-zero_fp-compact_full-et_rmp_ce_ckpt3664/
  v1-20260521-020538/
```

The inspected invalid-bbox sidecar is:

```text
monitor_dumps/invalid_bbox_points/invalid_bbox_points_summary.json
```

Important observations:

```text
current train8 / 64-step stage2_trie smoke:
    monitor samples inspected: 80
    parse-invalid samples: 15
    complete geometry-invalid row events: 11
    structural malformed row events: 7

geometry invalids:
    concentrated on image_id=49
    desc=person
    x-axis order violation
    nearest GT:
        object_id=49:7
        coco_ann_id=2010752
        bbox=[501,669,521,716]
        width=20
```

Raw COCO for `coco_ann_id=2010752` is valid:

```text
raw xywh=[191.0,334.37,7.16,23.41]
norm1000 xyxy ~= [501,669,520,716]
processed bbox_2d=[501,669,521,716]
```

The repeated invalid predictions were like:

```text
[543,666,520,711]
[539,666,520,717]
[545,668,520,711]
[545,669,520,717]
```

This is not raw-GT corruption. It is a model-learning issue around a crowded,
same-description, very thin person object where `x1` crosses beyond `x2`.

Broader smoke scan over `output/stage2_ab/smoke/**/monitor_dumps/step_*.json`
found strong concentration rather than random invalidity:

```text
monitor step files: 873
monitor samples: 1018
parse-invalid samples: 62
complete geometry-invalid row events: 4706
structural malformed row events: 7676
```

The larger train64 preflight runs showed many invalid rows, but concentrated in
particular unstable configs and crowded/thin/boundary-heavy images or classes.
This supports treating complete geometry-invalid rows differently from
structurally malformed rows.

### Final Row Terminology

Do not use `no-credit` as the canonical term. Use:

```text
committed row
pending UL candidate
uncommitted row
```

Definitions:

```text
committed row:
    legal row matched to a remaining GT object or promoted UL object
    can update emitted / remaining semantic state
    may provide supervision only when it is part of constructed target suffix,
    not when it is historical rollout prefix

pending UL candidate:
    legal unmatched proposal collected from one rollout before K-rollout
    consensus decides whether it becomes promoted UL

uncommitted row:
    invalid geometry
    structural malformed row
    duplicate burst row
    legal FP / low-IoU localization miss
    failed or unpromoted UL candidate
    wrong-desc unmatched row
    does not update emitted / remaining state
```

### Dirty Textual Prefix / Clean Semantic Residual State

The canonical Stage-2 state split is:

```text
rollout textual prefix:
    raw model-generated token history
    use original response_token_ids whenever available
    preserve legacy newline and other emitted token quirks

semantic residual state:
    GT + promoted UL supervision objects not yet committed by online matching
```

Do not rewrite rollout prefix rows into GT rows:

```text
committed TP prefix row:
    keep original rollout desc/bbox tokens
    do not replace with canonical GT tokens
    use it only to update semantic state
```

All self-rollout prefix tokens are context, not imitation targets:

```text
rollout prefix tokens:
    input context: yes
    normal CE labels: -100
    type loss: no, unless an explicit correction atom is attached to that
              logits position
```

Correction atoms may attach to logits positions inside the rollout prefix, but
those atoms supervise expert next-token choices, not the actual generated token
that follows in the rollout.

### Causal Logits Position Contract

Correction-event positions use standard causal-LM next-token convention:

```text
logit_position = prefix_last_token_index
loss reads logits[:, logit_position, :]
logits at token i predict token i+1
```

Examples:

```text
assistant output start:
    logit_position = last prompt token

boundary before object row k:
    logit_position = token immediately before row_k_start

boundary after complete object row:
    logit_position = row_end_token

boundary before rollout EOS:
    logit_position = token immediately before EOS
```

Never use the row-start token index for boundary supervision.

### Matching / Commitment Rules

First-version committed matching is deliberately stricter than normal IoU-0.5
detection matching:

```text
commit_iou_threshold = 0.75
```

The threshold applies to both GT objects and promoted UL objects.

Committed match condition:

```text
row structurally complete
bbox legal: x1 < x2 and y1 < y2
not duplicate burst
normalize_desc(pred.desc) == normalize_desc(target.desc)
IoU(pred.bbox, target.bbox) >= 0.75
greedy online match to current remaining supervision set
```

Description normalization is intentionally minimal:

```text
strip leading/trailing whitespace
collapse repeated whitespace to one space
```

Do not use case folding, underscore/space conversion, synonym matching, alias
matching, or text-similarity matching in the first version.

Online matching is rollout-order greedy:

```text
for each complete legal row in rollout order:
    first check duplicate burst
    then match remaining GT by desc + IoU
    then match remaining promoted UL by desc + IoU
    if committed, remove matched supervision object from remaining set
```

Random-order legal TP sequences are clean contexts. Canonical/sorted order is
not required for clean status.

Rows with same desc but IoU below `0.75` are uncommitted FP/localization misses,
not refinement targets. Rows with desc outside the remaining supervision set are
uncommitted FP dirty-prefix context, not structural errors.

### Bbox Coordinates Are Not Repaired In Stage-2

The first implementation should not do bbox-coordinate refinement:

```text
do not sort x1/x2 or y1/y2
do not clamp invalid boxes
do not use nearest-GT repair
do not apply coord-token trie repair
do not use coord tolerance radius
do not add regression-ish bbox repair
do not require exact coord-token equality for committed TP rows
```

Bbox geometry only decides whether a prediction is good enough to commit:

```text
legal + desc exact + IoU >= 0.75:
    committed

otherwise:
    uncommitted dirty row
    semantic remaining unchanged
```

This supersedes the earlier consideration of bbox-internal coordinate
earliest-divergence correction for invalid geometry rows.

### Invalid / Malformed / Truncation Handling

Complete-but-illegal bbox row:

```text
conditions:
    object row boundary is reliable
    desc and box_start present
    exactly four coord tokens
    geometry invalid: x2 <= x1 or y2 <= y1

handling:
    keep as dirty textual prefix
    do not commit
    do not repair
    do not supervise its internal tokens
    remaining state unchanged
```

Structurally malformed middle span:

```text
examples:
    wrong coordinate arity
    missing desc or missing box_start
    text/tool_call contamination inside object row

if a later <|object_ref_start|> delimiter reliably resynchronizes:
    keep malformed span as dirty context
    mask malformed tokens
    do not create atoms or type loss inside the malformed span
    continue scanning suffix

if no reliable resync:
    suffix is diagnostics-only from the last stable boundary
```

This policy supports dirty-context recovery without weakening schema
supervision:

```text
malformed but resync reliable:
    keep tokens as masked context
    train later reliable correction atoms

malformed and resync unreliable:
    cut back to last stable boundary or drop sample
```

Trailing incomplete object due truncation/no-EOS:

```text
drop whole trailing object span from its <|object_ref_start|>
revert training prefix to the last stable boundary
do not keep the incomplete tail as dirty prefix
```

This applies regardless of how far the trailing object got:

```text
<|object_ref_start|>
<|object_ref_start|>cat
<|object_ref_start|>cat<|box_start|>
<|object_ref_start|>cat<|box_start|><|coord_x1|>
...
```

If no complete object remains after dropping the tail, return to assistant
output start with the full supervision set remaining.

EOS is terminal:

```text
early EOS with remaining nonempty:
    train at the position immediately before EOS
    target valid continuation

do not:
    include EOS as dirty prefix and train after-EOS recovery
```

### Duplicate Burst Definition

Duplicate is a pred-vs-pred phenomenon, not primarily a pred-vs-GT status:

```text
duplicate burst if:
    current row is legal bbox
    same normalized desc as an earlier legal prediction in the same rollout
    IoU(current_bbox, earlier_bbox) >= 0.95
```

Duplicate burst is checked before remaining-GT/UL matching.

Duplicate handling:

```text
uncommitted row
no semantic credit
remaining state unchanged
tokens masked
can remain as dirty textual prefix if boundary reliable
boundary target remains current remaining set or EOS
```

Within-rollout duplicate burst evidence must not provide positive UL consensus.

Within-rollout here means within one `rollout_attempt`.

### Dirty-Prefix Recovery Gate

Dirty-prefix recovery has a small hard quality gate. This is not a new
hyperparameter family; it only prevents template/logits-position corruption:

```text
dirty prefix may be used when:
    prompt / assistant boundary is reliable
    at least one stable boundary can be located
    correction atom logit_position points to a real prefix token
    target valid set is nonempty
    prefix + constructed complete suffix fits max_length
```

Drop the correction sample when:

```text
assistant response start cannot be located
no stable boundary can be located
prefix_last_token / logits position is invalid
constructed suffix cannot fit any complete target object or EOS
desc vocab / token-type contract fails fast
```

The gate does not require every intermediate row to parse cleanly. It preserves
dirty-context recovery while refusing cases where the training position itself
would be ambiguous or wrong.

### K-Rollout UL Promotion

Use collect-then-classify:

```text
Phase 1: aggregate K rollouts
    parse rows
    identify valid object boundaries
    collect legal unmatched non-duplicate proposals as pending UL candidates
    flag invalid/malformed/duplicate diagnostics

Phase 2: promote UL
    cluster pending candidates across different rollouts
    same normalized desc
    cross-rollout cluster IoU >= 0.9
    K_valid >= 2
    support_distinct_rollouts >= 2
    support_distinct_rollouts / K_valid >= 1.0 by default
    no GT conflict
    not based only on within-rollout duplicate burst

Phase 3: build target IR
    supervision set = GT + promoted UL
    GT weight = 1.0
    promoted UL weight = 0.5
    online greedy scan each rollout against this supervision set
```

The canonical aggregate input is:

```text
rollout_attempts[]
```

Legacy adapter rule:

```text
if rollout_attempts[] exists:
    use it directly

else:
    collect known legacy fields deterministically:
        rollout_text
        anchor_rollout_text
        explorer_rollout_text
        corresponding token ids when available

    deduplicate exact same attempts
    assign rollout_id = 0..K-1
    record legacy_source_fields in provenance
```

After adaptation, all attempts are equal. There is no anchor/explorer priority.

Exact duplicate rollout attempts are deduplicated before UL consensus and before
correction training sample generation:

```text
attempt_dedup_key =
    response_token_ids if available
    else exact raw text / canonical byte string
```

Only exact duplicates are removed. Near-duplicate attempts remain distinct
self-prefix samples:

```text
keep as separate attempts:
    slightly different bbox coordinates
    different object order
    same proposal under different tokenization/text
    similar malformed spans with different error positions
```

Rationale: K rollouts are K self-prefix contexts, not an averaging estimator.
Approximate deduplication could delete the exposure-bias states Stage-2 is meant
to repair. Exact duplicates add no new context and must not count as distinct UL
consensus support.

Diagnostics:

```text
rollout_attempt_duplicate_dropped
legacy_source_fields
dedup_source_rollout_ids
```

Non-identical attempts with the same candidate bbox/desc can provide distinct
UL support:

```text
same desc
cluster IoU >= 0.9
distinct rollout_id after exact-attempt dedup
    -> distinct support
```

Pending UL candidates can only come from legal bbox rows:

```text
structurally complete
valid desc
legal bbox
not duplicate burst
not matched to GT
```

Invalid geometry rows and malformed rows are never UL candidates.

Consensus support counts distinct rollout ids:

```text
multiple near-duplicate candidates from the same rollout count as one vote
```

`rollout_id` means the stable id of a deduplicated equal rollout attempt within
the sample/batch context, not an anchor/explorer role.

Use `K_valid` as the consensus denominator:

```text
K_valid = rollout_attempts whose object boundaries are reliable enough to
          extract legal proposals
```

`K_valid` is about reliable proposal extraction, not a clean or fully successful
rollout. An attempt can contain invalid rows, malformed spans with reliable
resync, FP, duplicates, or dirty context and still be K-valid if legal proposal
rows can be extracted reliably. An attempt is K-invalid when object boundaries
are globally unparseable or too unreliable for proposal extraction.

Because K-valid excludes structurally invalid attempts, a default K=4 run can
promote from `3/3` valid attempts if the greedy attempt is invalid:

```text
K_total = 4
K_valid = 3
support = 3
ratio_over_valid = 1.0
ratio_over_total = 0.75
```

This is allowed if all other gates pass, but diagnostics must expose which
attempt decode modes were invalid.

Still report:

```text
K_total
K_valid
K_invalid
ratio_over_valid
ratio_over_total
invalid_attempt_decode_modes
```

Below-threshold clusters are retained for review, not training:

```text
status = candidate_below_threshold
not in supervision set
not committed
not supervised
does not override EOS
```

Promoted UL enters the sample-local residual set:

```text
remaining_supervision_set = GT objects + promoted UL objects
```

Promoted UL can be committed by legal rollout rows under the same desc + IoU
`0.75` rule, with supervision weight `0.5` when used in constructed suffix.

UL representative bbox:

```text
cluster representative = medoid
    member with highest mean IoU to other cluster members

do not average bboxes
```

GT/UL conflict rules:

```text
same-desc UL-vs-GT IoU >= 0.75:
    reject as GT conflict / GT support evidence

same-desc UL-vs-GT 0.30 <= IoU < 0.75:
    reject from training in v1
    status = rejected_near_gt_gray_zone
    keep review artifact

cross-desc UL-vs-GT IoU >= 0.75:
    reject promotion
    keep review artifact

UL-UL same-desc overlap >= 0.9:
    merge or keep one representative before entering supervision set
```

The near-GT gray zone prevents stable localization bias around an already
labeled same-description object from being promoted as a new unlabeled object.
It trades away some near-object UL recall for safer first-version mining:

```text
gt_gray_iou_low = 0.30
commit_iou_threshold = 0.75
```

Do not cap the number of promoted UL objects per sample in v1. Use all promoted
UL clusters that pass the gates, but make high-UL cases visible:

```text
promoted_ul_count
gt_count
promoted_ul_to_gt_ratio
promoted_ul_weighted_mass
promoted_ul_high_count_warning if promoted_ul_count > max(3, 2 * gt_count)
```

This keeps the first version useful for discovering heavily under-labeled images
while preserving diagnostics for stop-calibration or noise issues.

### Correction Events And Weights

The first version should use:

```text
one rollout -> one training sequence
all eligible correction events active
no max active event cap
```

This supersedes the earlier `one first event per rollout` plan.

The sequence is:

```text
input = prompt_ids + raw rollout prefix ids + constructed continuation suffix ids
```

Loss is not ordinary label imitation over the rollout prefix. Instead:

```text
rollout prefix tokens:
    labels = -100

eligible correction event positions:
    custom target IR atoms may read logits at the prefix-last-token position

constructed suffix:
    active target IR labels/loss atoms
```

Eligible correction events include:

```text
before first uncommitted row:
    clean-prefix correction

after uncommitted rows when boundaries are reliable:
    dirty-prefix recovery

clean incomplete / early stop:
    continue remaining set

remaining empty but model continues:
    EOS correction
```

Weights:

```text
clean-prefix correction:
    1.0

clean-incomplete / clean early-stop continue:
    1.0

dirty-prefix recovery:
    fallback_loss_weight

promoted UL object path:
    0.5
```

`fallback_loss_weight` is reused. Do not add a new dirty-recovery weight knob.

Clean-success rollouts are skipped for correction training:

```text
all rows committed
remaining supervision set empty
EOS/stop correct
no dirty/uncommitted rows
```

They are not used as implicit positive-retention samples in the default
correction builder:

```text
clean_success:
    no correction atom
    no constructed suffix
    skip sample
    count diagnostic only
```

Diagnostics:

```text
clean_success_rollout_skipped
clean_success_rate
```

An optional clean GT SFT stabilizer stream may be supported, but it is disabled
by default. The default Stage-2 dataset is all self-prefix correction attempts:

```text
default:
    clean_gt_sft_mix = 0
    train every retained rollout_attempt as a self-prefix correction sample

optional stabilizer:
    mix clean GT random-order samples into training
    use the same typed-trie valid-set semantics
    keep promoted UL out of the clean stream in v1
```

This optional stream is a schema/enumeration stabilizer only. It must not become
the main Stage-2 objective or silently reintroduce sorted-order SFT.

Clean-incomplete rollouts produce correction:

```text
prefix = committed rollout subset
remaining supervision set nonempty
target = remaining continuation
weight = 1.0
```

Remaining-empty over-generation produces EOS correction:

```text
boundary before first extra row:
    target EOS
    weight 1.0 if prefix clean

later dirty boundaries:
    target EOS
    weight fallback_loss_weight
```

### Constructed Continuation Suffix

For the `checkpoint-3660+` / `checkpoint-3664` Stage-2 v1 path, the constructed
continuation suffix should remain checkpoint-compatible with the existing
Stage-1 chat/template contract. Do not invent a second Stage-2 grammar for
newline, object separators, or EOS placement.

```text
v1 checkpoint-compatible template resolution:
    reuse / mirror the existing Stage-1 rendered assistant template
    preserve any template-rendered newline / separator / stop behavior
    do not strip newline from supervised continuations if Stage-1 renders it

future no-newline grammar:
    possible later migration
    not the default for this checkpoint-compatible Stage-2 run
```

If the resolved Stage-1 template renders newline at a position, newline is a
deterministic schema/control token there, not an optional grammar alternative:

```text
newline position:
    token_type = schema
    valid_token_ids = {newline_token_id}
    target_kind = singleton
```

Do not use boundary valid sets such as `{newline, <|object_ref_start|>}` in v1.
That optional grammar would be a separate migration and would complicate the
residual-state and logits-position checks.

Target assembly must still avoid duplicating deterministic schema tokens that
the kept raw prefix already contains. This is a suffix-start-state problem, not
optional grammar:

```text
if kept raw prefix already includes the Stage-1-rendered separator/close token:
    constructed suffix starts after that token

if kept raw prefix ends immediately before the next Stage-1-rendered schema token:
    constructed suffix emits that schema token first

if suffix starts from assistant output start:
    use the full checkpoint-compatible Stage-1 template from the beginning
```

Template-rendered schema remains deterministic, but the builder must not emit
the same deterministic schema token twice. EOS-before/after-separator behavior
must follow the resolved Stage-1 chat/template surface rather than a Stage-2
hard-coded rule.

But dirty textual prefix preserves raw rollout token ids:

```text
do not decode -> normalize -> re-encode the rollout prefix by default
```

Only if original token ids are unavailable may the builder re-encode decoded
text, and it must record a diagnostic:

```text
dirty_prefix_reencoded = 1
```

Continuation object order:

```text
random remaining order
base seed = 17
deterministically derived once per correction sequence from sample / rollout /
suffix-start boundary identity
GT and promoted UL shuffled together
```

Do not force GT-first or UL-tail ordering. Object source affects loss weight,
not order.

The order is fixed for the built correction sequence in v1:

```text
same base_seed + sample_id + rollout_id + suffix_start_boundary
    -> same remaining object order
```

Do not resample suffix object order per epoch or per optimizer step in the first
version. Future augmentation may add explicit multi-view suffix order sampling,
but the default should stay reproducible while self-prefix correction and UL
promotion semantics are being validated.

For active target positions:

```text
valid set = current remaining supervision set
```

For masked rollout-prefix positions:

```text
do not train A/B again even if a clean full-sequence target would have included
trie-marginal alternatives at those earlier boundaries
```

Example:

```text
GT = {A,B,C}
rollout prefix = A,B,EOS

active correction:
    prefix A,B
    remaining {C}
    target C continuation, then EOS after C
```

The `A` and `B` rollout tokens stay masked. Stage-2 does not back-fill their
earlier clean SFT losses in this correction sample.

### Diagnostics And Artifacts

Do not create many new files by default.

UL review artifact:

```text
<run_dir>/monitor_dumps/ul_clusters.jsonl
```

One JSONL row per cluster, including enough information for later visualization:

```text
global_step
sample_id / image_id / base_idx
dataset_name / dataset_id when available
image_path when available
original_width / original_height when available
processed_width / processed_height
coordinate_space = norm1000
cluster_id
status
desc
representative_bbox
all member bboxes
member rollout ids
K_total / K_valid / support / ratios
weight_if_promoted
GT conflict info
nearby_gt_summary
duplicate-burst flags
promotion/rejection reason
recoverable image provenance
```

UL bbox fields should be canonical normalized `xyxy` coordinate tokens /
norm1000 coordinates. The artifact should carry enough image provenance for a
later visualization tool to load the image and convert coordinates, but it
should not generate PNGs by default.

For dirty-prefix correction, do not create a default target-IR dump file.
Instead, add compact counters to existing `monitor_dumps/step_*.json` sample
stats or triage payloads:

```text
committed_gt_rows
committed_ul_rows
pending_ul_candidates
promoted_ul_clusters
uncommitted_invalid_geometry
uncommitted_malformed
uncommitted_duplicate
uncommitted_fp_or_unpromoted
clean_correction_events
dirty_correction_events
eos_targets
continue_targets
truncated_tail_dropped
truncated_tail_token_count
truncated_tail_stage
dirty_prefix_reencoded
```

The implementation should also preserve invalid-geometry diagnostics by image,
desc, axis, and nearest-GT summary for analysis, but this is diagnostic only and
must not become nearest-GT training supervision.

## 2026-05-21 OpenSpec Rewrite Decision And Audit Outcome

Decision: reuse and rewrite the existing OpenSpec change
`openspec/changes/add-stage2-residual-set-ul-correction` instead of creating a
new overlapping change.

Rationale:

- The change name already matches the intended stable contract: Stage-2
  residual-set correction with UL support.
- The previous contents were stale and carried superseded concepts such as
  `bbox_tail_from_anchor`, earliest-only events, and anchor/explorer framing.
- Reusing the change avoids duplicate OpenSpec surfaces while still allowing a
  full rewrite of proposal, design, tasks, and spec deltas.

The rewritten OpenSpec now treats the v1 contract as:

- offline prepared rollout attempts as input;
- K rollout attempts as independent self-prefix samples, not an averaged
  pseudo-label set;
- exact rollout-token dedup;
- template-boundary adapter for Stage-1-compatible rendering/tokenization;
- shared `SupervisionAtom` IR with explicit causal `logit_position`;
- standalone token-type loss plus inner valid-set marginal/hard-path loss;
- action-based valid-set transitions to avoid object-coordinate mixing;
- dirty-prefix recovery by constructing correction suffixes from the semantic
  residual set;
- no raw-coordinate repair and no `bbox_tail_from_anchor`;
- strict UL promotion with cross-rollout same-desc consensus and rollout-local
  member bbox supervision;
- duplicate burst as diagnostic/exclusion only, not unlikelihood;
- spatial wrong-description conflicts as low-weight earliest-divergence desc
  correction when the span is reliable.

Subagent audit outcome:

- First review round found no P0, but flagged config ambiguity, stale
  anchor/explorer language, underspecified `SupervisionAtom`, underspecified
  valid-set transitions, rollout-local UL ambiguity, and missing migration
  tasks.
- The spec was refined to require `prepared_rollout_jsonl`, strict residual-set
  config keys, `ValidAction` transitions, rollout-local UL training bboxes, and
  canonical `monitor_dumps/ul_clusters.jsonl`.
- Second review round found no remaining P0 implementation blockers. One P1
  wording mismatch was fixed by making reliable `spatial_wrong_desc_conflict`
  atoms mandatory; one P1 implementation-alignment concern was converted into
  explicit schema/test/smoke migration tasks.

Verification after refinement:

```bash
openspec validate add-stage2-residual-set-ul-correction --type change --strict --no-interactive
git diff --check -- openspec/changes/add-stage2-residual-set-ul-correction progress/explorations/2026-05-20_stage2_residual_set_self_prefix_ul_redesign.md
```

Both checks passed on 2026-05-21 in
`/data/CoordExp/.worktrees/unified-training-infra-refactor`.
