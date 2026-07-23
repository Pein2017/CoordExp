# Comprehensive Handoff: Qwen3-VL Dense Autoregressive Detection Research Flow

Updated through 2026-07-23.

This is the primary fresh-session entry point for the current Qwen3
Vision-Language (`Qwen3-VL`) dense-enumeration program. It compresses the
research motivation, mechanism evidence, failed and demoted routes, scientific
method, current data and model surfaces, and the next training contract.

Do not reconstruct the current route from the long chat transcript. Read this
file first, then reopen the linked research units and artifacts when a claim or
implementation detail matters.

## Read This First: Final Route Correction

The next treatment is **true order-independent set-level supervision**.

The immediately preceding route drifted toward single-owner and single-row
training:

```text
choose one sampled or geometry-sorted owner
  -> imitate one complete object row
  -> optionally preserve one Source row
```

Those experiments remain valuable. They proved that an owner-directed gradient
is learnable: selected physical owners enter clean greedy rollout far more
often after training. They also exposed the central failure: the model often
gains the selected owner by losing another useful owner. Increasing image
breadth did not repair that mismatch at the frozen training dose.

The next objective must therefore represent the deployment goal directly:

> At a model-produced rollout state, every verified uncovered physical owner is
> a valid next owner. Training must reward the final unique physical-owner set,
> independent of traversal order, without designating one row, owner, or
> trajectory as the sole teacher target.

A same-prefix valid-versus-harmful row comparison may remain an auxiliary
diagnostic. It is not the primary training objective. Token-type gating may
remain a stability mechanism. It must not turn the experiment back into exact
cross-entropy on one selected row.

The source of this correction is:

`/data/CoordExp/.worktrees/research-probes/handoff/from-side-chat.md`

That source discusses two possible set objectives. This handoff selects the
first one for the initial screen: use every admissible strict owner-set
dominance pair, with set-equivalent trajectories treated as positive aliases.
The alternative listwise objective that moves probability mass onto the
observed nondominated set is deferred until the pairwise contract has been
measured and tested.

The current compass and project memory have been updated to reflect it:

- `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/compass.md`
- `/data/CoordExp/.worktrees/research-probes/memories/current.md`

## 1. Workspace and Authority Boundary

### Current checkout

```text
Repository: /data/CoordExp
Worktree: /data/CoordExp/.worktrees/research-probes
Branch: research-probes
HEAD observed while writing this handoff:
8beb8e21d
```

This handoff was finalized after the Pi-worker pilot and the constant-dose
image-breadth screen had been committed separately. At that boundary, the only
remaining dirty group was this set-level route correction, its compass and
memory updates, and the two handoff files. The final commit is expected to
leave the worktree clean. Reinspect live status before new work; do not infer
current dirtiness from this historical snapshot.

### What each surface owns

| Surface | Authority |
| --- | --- |
| `outputs/research/` | Executed artifacts, raw trajectories, receipts, traces, and metric primitives |
| `research/.../experiments/<unit-id>/` | Scientific protocol, evidence handles, executed observations, bounded verdict, and claim boundary |
| `research/investigations/` | Cross-unit synthesis and competing explanations |
| `research/decisions/` | Mutable evidence-backed route choices |
| `research/mechanisms/` | Reusable mechanisms only after independent support and a successful novel prediction |
| `openspec/` | Stable reusable implementation and compatibility contracts, not hypotheses or scientific conclusions |
| `docs/` | Current software and operator behavior |
| `memories/` | Concise cross-session continuity, never formal authority |
| `progress/` | Deprecated legacy evidence; do not create new files there |

An artifact path is a provenance handle, not proof that execution was valid.
Reopen receipts, resolved configurations, model identity, decode policy, and
failure counters before using an artifact for a new conclusion.

## 2. North-Star Objective

Train the pretrained Qwen3-VL model to enumerate the complete supported Common
Objects in Context 80-category (`COCO-80`) physical-object set through one
deterministic greedy autoregressive rollout while preserving:

- the pretrained model's visual and language knowledge;
- strong one-shot object recognition and localization;
- phrase-to-geometry consistency;
- unique physical-owner coverage rather than repeated hits;
- low unsupported-entity hallucination;
- valid object-row syntax and natural termination; and
- the ability to generalize beyond the images that supplied gradients.

The deployment target is one strong deterministic traversal. Sampling and
bagging are exploration, diagnosis, and data-construction tools, not the final
inference policy.

The program intentionally tests the limit of the existing modern multimodal
language model. It does not begin by attaching an external detector, a second
powerful backbone, Detection Transformer-style object queries, or a learned
object-slot bank. Such mechanisms remain valid later branches only if the
native model fails a properly set-aligned training test.

## 3. Model and Data Context

### Primary model

- Base family: Qwen3 Vision-Language 2-billion-parameter model.
- Output: description-first autoregressive object rows containing one object
  phrase and four bounding-box coordinates.
- Coordinates: discrete normalized coordinate tokens from 0 through 999. The
  coordinate order is `x1, y1, x2, y2`, where `x` is horizontal, `y` is
  vertical, `(x1, y1)` is the upper-left boundary, and `(x2, y2)` is the
  lower-right boundary.
- Coordinate values remain normalized to 0 through 999 during data, rollout,
  scoring, and training. Conversion to absolute pixels is only for rendering
  and visual review.
- The 2-billion-parameter model uses tied input-token embeddings and output
  language-model head weights.
- Fine-tuning: Weight-Decomposed Low-Rank Adaptation (`DoRA`) over the language
  tower. The vision tower and multimodal projector remain frozen for the next
  treatment unless later evidence explicitly changes that decision.
- Primary ordering checkpoint: geometry-sorted, description-first, pure
  cross-entropy with token-type gate, step 4,887.

Primary checkpoint:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/
checkpoints/step-4887/checkpoint.json
```

Random-order ablation checkpoint:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_random_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1-20260719T070043Z/
checkpoints/step-4887/
```

The random-order checkpoint is an ablation, not the preferred base. It changed
prefix sensitivity but did not create an order-free covered-set state; on the
best-resolved dense-person image it collapsed toward one habitual successor.

### Dataset scope

- Current ontology: COCO-80 only. Other visible entities are outside the
  reportable class set for this investigation.
- Dense COCO images contain many real but officially unlabeled objects.
- Official unmatched predictions are never automatically hallucinations.
- Entity existence/category and geometry quality must be adjudicated
  separately.

Current validation data:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

Image root:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/
```

Twelve manually refined dense validation images:

```text
16228, 14038, 1584, 13348, 4134, 2685,
5001, 10707, 6040, 7511, 14439, 13923
```

These twelve images are high-quality safety and interpretation evidence. They
must not supply training gradients.

## 4. Core Terminology

- **Physical owner**: one real object instance, independent of class label or
  how many generated rows refer to it.
- **Complete object row**: one syntactically closed generated unit containing
  a description and four bounding-box coordinates.
- **Covered owner set**: verified physical owners already emitted by a prefix.
- **Remaining owner set**: verified reportable owners not yet represented in
  the covered owner set.
- **Trajectory**: one complete autoregressive rollout or a continuation from a
  declared rollout state.
- **Source checkpoint**: the frozen geometry-sorted pure-cross-entropy step
  4,887 checkpoint used as the primary baseline.
- **Greedy rollout**: deterministic next-token argmax decoding.
- **Low-temperature sampled rollout**: stochastic decoding used to explore
  alternative model-supported trajectories.
- **Owner exchange**: a treatment gains one physical owner while losing a
  different owner, leaving final set coverage flat or ambiguous.
- **Unknown prediction**: an unmatched prediction whose physical existence,
  category, or geometry cannot be reliably adjudicated. It receives no
  negative training gradient.
- **Source@B16**: repetition penalty 1.0 with at most sixteen complete object
  rows, or an earlier natural image-end. It is a bounded baseline created
  because unrestricted greedy decoding can enter repeated-row loops.
- **StateBank**: the repository's serialized collection of exact-replay
  training events, provenance, token spans, and supervision metadata. It is an
  implementation container, not a scientific state or memory mechanism.

## 5. First-Principles Formulation

For image `I`, let the verified reportable physical-owner set be:

\[
O(I)=\{o_1,\ldots,o_N\}.
\]

At row boundary `t`, the actual generated prefix is `P_t`. Matching prior rows
to physical owners gives the covered set:

\[
C_t=C(P_t),
\]

and the remaining set:

\[
U_t=O(I)\setminus C_t.
\]

For a continuation or full trajectory `tau`, let:

\[
S(\tau)\subseteq O(I)
\]

be its distinct verified physical-owner set. The desired deployment value is
not the likelihood of one canonical sequence. It is the quality of the owner
set produced by deterministic greedy decoding:

\[
J_{\mathrm{greedy}}(\theta;I)
=
\left|S\left(g_\theta(I)\right)\right|,
\]

subject to phrase, geometry, syntax, hallucination, duplication, and natural-
termination safety.

Traditional teacher forcing optimizes one serialized target:

\[
\mathcal L_{\mathrm{CE}}
=
-\sum_k\log p_\theta(y_k^*\mid I,y_{<k}^*).
\]

This provides dense token supervision but does not identify the latent task
state or policy:

- which uncovered owner may be selected next;
- whether several different owners are equally valid;
- whether phrase and geometry belong to the same physical owner;
- whether a completed row has been committed;
- whether probability is redistributed toward remaining owners; or
- whether stopping means visual completion rather than linguistic completion.

The core mismatch is therefore not “too few token losses.” It is that token
cross-entropy under one ordering does not identify the order-independent set
objective.

## 6. Current Best Mechanism Model

The most economical explanation consistent with the evidence is:

> Qwen3-VL already contains usable visual support for many missed physical
> owners and can causally route that support into object rows. Its weak point is
> converting distributed candidate support into a stable, permutation-
> invariant traversal of the remaining owner set.

The model's active state is not “no memory.” It is a distributed, geometry-
sensitive route state carried by image features, the full generated prefix,
cached language states, and current autoregressive phase. It sometimes
suppresses a covered owner, but also retains row order and path residue,
overwrites access to other owners, and exchanges one valid owner for another.

The current operational chain is:

```text
static visual evidence
  -> prefix-conditioned candidate competition
  -> continue or stop
  -> category or description decision
  -> first discriminative coordinate decision
  -> progressive box completion
  -> completed row perturbs the next route state
```

Continuation, description identity, instance selection, and complete geometry
are not one atomic decision. The first discriminating boundary changes by
scene: it may be a description token, `x1`, `y1`, or a later extent coordinate.

No stable native order-free covered-set ledger, universal latest-row carrier,
portable complete-object vector, or final architecture has been established.
The distributed state may nevertheless be sufficient if training credit is
aligned with the final set.

## 7. What the Experiments Established

### 7.1 Baseline behavioral signature

The geometry-sorted pure-cross-entropy model is usually conservative:

- recall is low in dense scenes;
- unsupported-entity hallucination is relatively rare in normal rollout;
- early termination occurs even when valid owners remain;
- selected scenes enter low-confidence duplication bursts;
- dense and repeated-category scenes degrade more strongly;
- geometry-sorted ordering is modestly stronger than random ordering;
- prefix changes can alter later objects and coordinates; and
- multiple low-temperature rollouts discover complementary real owners.

These symptoms do not prove a single cause. In particular, suppressing the
terminal token only lengthens output; it does not provide evidence for which
owner should appear next.

### 7.2 Visual support and causal control exist

Painting an object boundary on the original image can make the model describe
and localize the marked object even when the text prefix suggests another
object. Anti-copy training showed that a corrupted mark can act as a proposal
while the model returns to the clean image to recover the physical object,
rather than merely copying the rendered rectangle.

The painted-minus-clean visual delta can be injected after visual features are
scattered into ordinary language-model image-token embeddings. Zero-delta
controls are neutral. This proves a real post-vision visual-to-decoder control
surface; it does not prove that a learned cursor or slot is required.

Historical late-middle language-layer probes found bounded phase separation:

- stop-versus-continue control becomes strong around decoder layers 17 through
  21;
- description identity becomes stronger around layers 20 through 23; and
- a late residual replacement can recover the immediate divergent token.

A one-time residual patch did not recover a complete row or all four
coordinates. These layers are useful observation and intervention surfaces,
not a selected fixed “heart-bypass layer.”

Key evidence:

- `/data/CoordExp/.worktrees/research-probes/research/ideas/qwen3-vl-painted-gt-transcription-probe/`
- `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/2026-07-13-to-2026-07-16-weekly-research-report.md`

### 7.3 Attention is not causal explanation

Attention maps often show plausible regional focus, but magnitude does not
prove causal influence on later logits. Hard spatial eligibility, soft bias,
residual replacement, same-feature controls, and no-operation controls showed
that the relevant state is distributed across attention, residual, multilayer,
and phase-specific computation.

Future attention work is justified only when it diagnoses one exact routing
seam or tests a concrete causal intervention. More attention visualization is
not a main research direction.

### 7.4 Masking and tiling reveal competition but are not final policies

No-resize native tiling and full-canvas masking were introduced to test whether
showing only part of the image makes local owners easier to emit without
cheating through magnification.

The masked policy sometimes rescued objects missed by full-image rollout,
including real officially unlabeled objects. It also lost global context,
fragmented objects across regions, duplicated partial instances, and produced
avoidable hallucinations such as implausible categories under local context.

Equal-call full-image bagging was competitive or better after aggregation.
Therefore:

- spatial competition is real in bounded cases;
- global context is also valuable;
- masking is a causal/control baseline, not the accepted inference policy; and
- a mask benefit alone cannot be called a purely language-side failure.

Key unit:

`/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/`

### 7.5 Sampling exposes reachable owner support

Repeated low-temperature full-image sampling reveals physical owners missed by
greedy decoding. On the twelve manually refined images:

- some sampled trajectories conservatively beat greedy at the same row budget;
- several images contain more verified owners in the sampled union than any
  single short trajectory can hold; and
- different trajectories provide complementary owner support.

This rules out “the vision tower cannot represent any missed object” for those
rescued owners. It does not rule out genuine recognition limits for objects
that no controlled route can find.

The current 2,432-image sampled panel contains sixteen trajectories per image:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
trajectory-panel-2432-vllm/production-v2/
```

Executed scope:

- 2,432 images;
- 38,912 sampled trajectories;
- temperature 0.4;
- nucleus probability 0.95;
- repetition penalty 1.0;
- 1,024 generated-token limit;
- natural closure for all recorded sampled trajectories; and
- no matched unbounded greedy trajectory in this panel.

Sampling estimates observed reachability under the sampling policy. It does
not estimate complete mathematical support and does not by itself prove a
greedy rescue effect.

### 7.6 Prefix rows causally change later routing

Complete coherent rows influence later owner selection. The effect is
content-sensitive and often geometry-sensitive:

- at one cup state, only a coherent cup phrase plus left-cup geometry advances
  to the right cup;
- phrase alone, geometry alone, and arbitrary equal-length rows fail;
- same-covered-set prefix permutations can change later physical owners even
  when the final row is held fixed;
- one owner can be locally suppressed after its row is included; and
- natural coordinate variants of the same owner can unlock different later
  owners.

The effect is not a clean set operation. Of three causal same-owner coordinate-
history cases, only one safely adds an owner; two exchange owners. One image
depends mainly on the latest row, while another requires a joint two-row
history. Thus the model has executable route state, but no proven monotonic
“add this owner to the covered set” transition.

Key units:

- `2026-07-15-prefix-state-phrase-geometry-factorial`
- `2026-07-19-same-covered-set-prefix-order-equivalence`
- `2026-07-19-common-object-prefix-permutation-short-horizon`
- `2026-07-19-complete-candidate-row-score-decomposition`
- `2026-07-19-sampled-history-target-reachability-and-complete-row-value`

All are indexed under:

`/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/index.md`

### 7.7 Phrase, owner, and complete geometry are separable

Description-first generation is empirically stronger than geometry-first for
the current Qwen3-VL model, but selecting the description does not uniquely
select the physical owner when several same-class objects exist.

The four coordinates also do not behave as one already-bound box:

- one boundary can be plausible while another is shifted;
- `x1` can select a horizontal corridor without determining full extent;
- shared `x1` candidates may separate at `y1` or `x2`;
- `y2` can jump to a full-canvas or merged-instance extreme; and
- neighboring instances can contaminate different boundaries.

An owner may therefore be discovered correctly while its physical extent is
wrong. Entity discovery, category, and geometry must be evaluated separately.
Do not treat one good coordinate, one wrong coordinate, or one coordinate-token
argmax as a complete instance-binding verdict.

### 7.8 Full-model float32 is required for delicate causal claims

The same prompt and visual features produced batch-shape-dependent coordinate
logits in `bfloat16`. Full-model 32-bit floating point restored practical
batch invariance for the exact recipients. Visual feature replay localized the
split after visual feature extraction.

Use full-model `float32` for exact logit comparisons, residual patching,
candidate-row scoring, and any result where a small coordinate difference
changes the mechanism interpretation. Ordinary high-throughput rollout may use
the production precision after a representative invariance check.

### 7.9 Random ordering did not create an order-free ledger

The random-order-trained checkpoint differs from the geometry-sorted
checkpoint, but not in the hoped-for direction. On the best-resolved dense
person case, random-order training sent almost every prefix condition to one
habitual successor. Geometry-sorted training retained more route diversity and
remains the pragmatic primary baseline.

This means fixed order is not proven to be the root problem, and random order
is not a set-level objective. A model can become order-insensitive by ignoring
the covered set and collapsing to one successor.

### 7.10 Coordinate correction is locally learnable but does not transfer to rollout

First-wrong-coordinate and 256-image coordinate-boundary treatments improved
teacher-forced exact-prefix coordinate margins across all four positions. Clean
free rollout did not improve. Lower learning rate reduced disruption but did
not close the transfer gap.

This separates local optimization from policy value:

```text
better target-coordinate margin at one frozen prefix
does not imply
better owner selection, complete box, or later set coverage
```

Do not scale the coordinate-only treatment unchanged.

### 7.11 Positive row imitation learns the selected owner but exchanges the set

Single-route, multiple-route, Source-preservation, and constant-dose image-
breadth screens established a consistent result:

- selected sampled-route owners become much easier to recover;
- route-family owners enter greedy output more often;
- Source-preservation rows retain the owners they explicitly name at high
  rates; yet
- ordinary owners and owners on non-admitted images are lost; and
- broader image exposure at the same training dose does not establish held-out
  final-set expansion.

In the latest fixed-dose screen, selected Source-missed owners were recovered
approximately 36 to 41 percent of the time, versus approximately 8 to 12
percent for non-selected missed owners on gradient images. This is strong
enrichment consistent with owner-specific uptake, not a same-owner untreated
counterfactual.

At 992 events and 31 optimizer updates, spreading training over 496 images did
not beat concentrating it in 162 nested images on the pre-admission held-out
cohort. The uncertainty intervals include zero, so the bounded conclusion is
“no established breadth advantage at this dose,” not “image breadth can never
help.”

Authoritative result:

`/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md`

The bounded scientific lesson is:

> Owner-directed credit clearly changes behavior, while the current target
> under-specifies final-set preservation. The remaining roles of optimization
> strength, transfer, and native-state accessibility are unresolved.

### 7.12 Repetition penalty is a trajectory intervention

The historical repetition penalty 1.10 setting was a practical duplication
heuristic. A corrected common-cohort comparison shows that it exchanges roughly
65 to 100 owner identities per arm and seed relative to repetition penalty 1.0
and often lowers absolute owner coverage.

It is not a harmless duplicate-only adjustment. Freeze repetition penalty 1.0
for primary scientific comparisons. Report repetition penalty 1.10 separately
when needed; do not use it to rescue a training conclusion.

### 7.13 Duplicate-cleaned trajectory treatment is bounded positive evidence

A separate physical-owner duplication unit compared duplicate-cleaned
trajectory imitation with equal-update Source-only controls. Relative to its
six-update control, the complete recipe gained four matched unique owners and
removed 51 strict duplicate candidates on the 256-image training panel. On 240
images that supplied no training event it was plus three owners and minus 48
strict duplicate candidates; on the twelve human-refined images it was plus
three owners and minus nine duplicate candidates.

This does not establish a uniformly safe mechanism. The controls repeated
Source rows and did not match event composition, owner exposure, or training
data semantics. Image `10707` lost a laptop and developed a repeated remote
row, and treatment-only versus control-only owner identities remained large.
The result reinforces two requirements for set-level work: use gained,
retained, and lost-owner accounting, and distinguish complete-recipe efficacy
from attribution to one loss component.

Evidence:

`/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/results.md`

## 8. Current Hypothesis Register

| Hypothesis | Current belief | What would change it |
| --- | --- | --- |
| Many greedy-missed owners are visually unavailable | Rejected for sampled- or cue-rescuable owners; still possible for owners never reached under any controlled condition | Persistent misses under sampling, local views, visual cues, and controlled support reads |
| The main problem is policy concentration and route dependence | Leading explanation | A set-aligned objective failing despite reliable support and correct optimization, or a carrier intervention succeeding under matched supervision |
| The model has a stable order-free covered-set ledger | Not established | Same-covered-set permutations with symmetric omission controls producing stable final-set completion and reactivating every omitted owner |
| The model has no usable state for coverage | Also not established | An oracle carrier outperforming native prefix state after matched set-level training would strengthen this claim |
| One latest row is the coverage carrier | Rejected as universal | One case is latest-row dominated; another requires joint earlier history |
| One coordinate commits the object | Rejected as universal | The discriminative coordinate changes by scene and candidate geometry |
| Early stopping is the root problem | Demoted to symptom | Terminal suppression produces many rows but few new owners and can increase duplication or invalid output |
| Sequence length alone is the root problem | Unresolved but not leading | Same content and visual evidence with matched state but different history horizon would be needed |
| Random row order teaches set prediction | Rejected under the executed checkpoint comparison | A future truly set-level objective may still use many orders without treating any one as canonical |
| COCO missing labels teach conservative omission | Plausible and important | Exhaustive labels versus original labels plus controlled label thinning on identical images |
| Explicit slots or a ledger are necessary | Unsupported and held | Consider only after a valid set-level loss fails and an oracle carrier succeeds under matched supervision |
| True set-level supervision can distill sampled union support into greedy coverage | Highest-priority open treatment hypothesis | Run the 256-image grouped set-level screen described below |

## 9. True Set-Level Supervision Contract

### 9.1 Non-negotiable semantics

For an actual rollout prefix `P_t`:

\[
C_t=\{\text{verified owners already covered}\},
\]

\[
U_t=O(I)\setminus C_t.
\]

Every owner in `U_t` is a valid next owner. No geometry-sorted owner, sampled
owner, or easiest owner is the unique target.

For each candidate continuation `tau_k`, construct:

\[
S_k=S(\tau_k),
\]

the final distinct trusted owner set. Compare trajectories using a partial
order:

\[
\tau_a \succ \tau_b
\quad\text{only if}\quad
S_b\subset S_a,
\]

provided `tau_a` is not worse in confirmed unsupported entities, malformed
rows, severe duplication, or other frozen safety criteria.

Examples:

- `{A, B, X}` dominates `{A, B}`.
- `{A, X}` and `{A, B}` are incomparable.
- Two different orders of `{A, B, X}` are set-equivalent.
- Repeating `A` does not expand the set.
- An unknown unmatched prediction is neutral, not automatically harmful.

No nondominated valid trajectory may be used as a negative. "Nondominated"
here is always relative to the declared candidate group; it is not a claim
that sampling found every possible route. Under the first all-pairs objective,
an incomparable trajectory that participates in no strict dominance edge is
neutral rather than an implicit positive or negative.

A group with only one observed nondominated serialization is not sufficient
evidence for order-independent multiple-positive supervision. The admission
census must measure exact-token aliases, distinct positive row orders, and
distinct first-owner identities. The primary screen may admit a group only
when at least two valid positive aliases begin with different uncovered owners,
or when a separately declared order-alias augmentation has passed the same-set
and first-owner invariance tests. Otherwise exclude the group or recollect;
never silently fall back to one maximal trajectory.

An individual dominance edge is admissible only when every valid first-owner
identity used by its lower-set aliases also appears as the first owner of at
least one higher-set positive alias. Otherwise whole-trajectory preference
could suppress a valid next owner merely because its observed route stopped
early. Exclude that edge or construct and validate the missing order alias;
do not call the owner negative.

### 9.2 First-screen grouping boundary

The first implementation should begin at one shared root prompt state per
image. At that state the covered set is empty, and the candidate group consists
of the frozen bounded Source greedy continuation plus the sixteen sampled
continuations, all projected to the same complete-row budget. This is the
cleanest available whole-trajectory set comparison.

Later events may begin at a nonempty model-produced prefix only when every
candidate continuation has exactly the same image, prompt, prefix token
identifiers, and prefix hash. Never compare scores from different prefixes as
if they were alternatives at one decision state.

### 9.3 Required training record

Each grouped record should contain:

```text
image identity and exact image input
model checkpoint and prompt identity
actual rollout prefix or declared root state
covered physical-owner identifiers
remaining verified physical-owner identifiers
multiple candidate continuations or trajectories
row-to-owner matches for every candidate
final unique trusted-owner set for every candidate
entity/category adjudication status
geometry adjudication status
duplicate, malformed, confirmed-false, and unknown status
decode policy and sample identity
```

The record must not contain a single `selected_owner_id` or `target_row` that
determines the whole loss.

For the first screen, a candidate containing an unresolved entity row or
untrusted exact geometry inside the scored continuation is ineligible for the
primary pairwise loss. Preserve it for diagnostics, but do not reinterpret it
as an empty-owner or harmful trajectory. This whole-candidate exclusion is the
only simple way to guarantee that an unknown row receives neither direct loss
nor indirect whole-trajectory preference gradient. A later masked or
marginalized treatment would be a separately tested extension.

### 9.4 Primary objective

The first screen uses one objective: all valid strict set-dominance edges.
After deduplicating identical token hashes, group trajectories with the same
trusted owner set as outcome aliases. For every pair of outcome classes in
which class `A` strictly contains class `B` and is no worse on the frozen
safety axes, compare every exact-token alias from `A` with every alias from
`B`:

\[
\mathcal L_{\mathrm{pair}}
=
\sum_{\tau^+\succ\tau^-}
\operatorname{softplus}
\left(s_\theta(\tau^-)-s_\theta(\tau^+)\right).
\]

Use every valid strict-dominance edge, including transitive strict-inclusion
edges, not one selected pair. Average aliases within each semantic edge, then
average edges within a state, states within an image, and images within an
optimizer update. This prevents images with more samples, aliases, or edges
from receiving a larger training dose. Do not compare incomparable owner
exchanges. A listwise Pareto-front probability-mass loss is deferred because
it does not preserve the same partial-order weighting in chains such as
`A` strictly containing `B` strictly containing `C`.

The trajectory score must control for length. Plain summed log-probability
mechanically penalizes longer valid enumerations; plain mean token log-
probability changes the weighting toward token predictability rather than set
completion. Freeze one explicit score in the research unit, such as complete-
row-normalized likelihood or frozen-Source-relative trajectory improvement,
and retain raw per-row and per-token components for diagnosis.

The primary scientific property is the partial order on final owner sets. The
research unit must freeze the trajectory score and normalization after the
admission census; implementation may not change the set semantics.

### 9.5 Safety terms

A scalar diagnostic reward may be recorded:

\[
R(\tau)
=
|S(\tau)|
-\lambda_d N_{\mathrm{duplicate}}
-\lambda_i N_{\mathrm{invalid}}
-\lambda_f N_{\mathrm{verified\ false}}
-\lambda_s\mathbf 1[\mathrm{STOP\ while\ trusted\ owners\ remain}].
\]

Strict set dominance remains preferable to a scalar reward because a scalar
can hide the loss of one owner behind the gain of another.

Rules:

- confirmed duplicate physical owners are harmful;
- malformed or invalid rows are harmful;
- confirmed unsupported-entity hallucinations are harmful;
- premature stop is harmful only when trusted uncovered owners remain;
- real owners with imperfect geometry retain entity credit and a separate
  geometry error;
- unknown or potentially unlabeled predictions receive no negative gradient;
- another valid uncovered owner is never a negative merely because it differs
  from one sampled route.

The implementation must perturb logits only at an excluded unknown row and
verify that the primary loss and gradients are unchanged. If that test cannot
pass, the candidate is not neutral and the training path must stop.

### 9.6 Auxiliary terms that remain allowed

After set-level assignment, trusted auxiliary terms may support transcription:

- token-type gating to limit gradient sites;
- phrase syntax or row-closure loss;
- trusted coordinate supervision;
- a same-prefix valid-versus-confirmed-harmful branch margin; and
- Source behavior preservation when represented at the final-set level.

No auxiliary may make one owner or row the primary unique teacher.

### 9.7 Offline first, refresh later

The first screen should be offline and bounded:

```text
Source trajectories
  -> construct grouped owner-set records
  -> train one set-level treatment
  -> evaluate a new clean greedy rollout
```

This is not yet online reinforcement learning. If the objective improves
training groups but not clean rollout, generate trajectories from the treated
checkpoint and run one mixed old-plus-refreshed dataset round. That result
tests state-distribution mismatch. Do not begin with a complex online loop,
Group Relative Policy Optimization, or external reward model before the offline
set semantics are shown to be learnable.

## 10. Data Construction for the First Screen

### Candidate source

Reuse the existing 2,432-image, sixteen-sample trajectory panel where valid.
The frozen pool already has disjoint `train-candidate.jsonl`,
`development.jsonl`, and `heldout.jsonl` files under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
candidate-pool-v1/
```

The admission census and all gradient-bearing selection must use only the
2,048 identifiers in `train-candidate.jsonl`. Development and held-out image
identifiers must be excluded before route inspection, and the three image sets
must be proven disjoint from the immutable split receipt.

For the selected 256 images, pair sampled trajectories with the existing
frozen `Source@B16` production baseline at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
source-b16-vllm/production-v1/
```

Only the 2,004 Source-eligible training-candidate images may enter the join.
The 44 Source-ineligible training images are ineligible, not empty-owner
baselines, and must not be rerun under altered batching to change status. Do
not recollect the sampled panel before a data census shows that it lacks enough
strict set-dominance relations.

Select 256 training images with:

- reliable trusted physical-owner annotations;
- multiple objects, preferably dense scenes;
- meaningful variation among sampled trajectories;
- at least one owner found only in some trajectories; and
- enough comparable candidate sets to produce nonzero set-level gradient.

The twelve manually refined validation images remain evaluation-only.

### Mandatory census before implementation scale

For every candidate image, compute:

- number of valid sampled trajectories;
- trusted unique-owner set per trajectory;
- number of distinct owner sets;
- number of strict superset relations;
- number of nondominated sets;
- number of positive exact-token and row-order aliases per nondominated set;
- number of distinct first-owner identities among positive aliases;
- count of singleton-maximal groups that would collapse to one serialization;
- count of valid first-owner identities that appear only on a dominated side;
- number of incomparable owner-exchange pairs;
- confirmed duplicate, invalid, false, and unknown counts; and
- whether geometry is adequate for owner matching.

If strict superset relations or multi-owner positive aliases are rare, the
stored data cannot support the proposed objective. Improve exploration or use
a separately declared and tested order-alias construction. Do not fall back to
unique-row cross-entropy merely to make the dataset nonempty.

### Physical-owner adjudication

Use the smallest axis-aligned rectangle containing visible evidence confidently
attributable to one owner. Do not silently extrapolate fully occluded extent.
When disconnected visible parts are confidently the same owner, their union is
valid; when ownership is uncertain, mark unknown.

Record three separate axes:

1. physical entity/owner existence;
2. COCO-80 category; and
3. geometry or boundary quality.

For ambiguous geometry, retain coherent admissible boxes or boundary intervals
rather than treating one annotator's exact box as absolute truth. Crop-enlarged
review may use ordinary interpolation to make pixels easier to inspect, but
must not introduce synthetic detail.

## 11. Minimal Implementation Roadmap

### Step 1: freeze a census and admission unit

Create a small immutable, read-only census unit under:

```text
research/investigations/qwen3-vl-dense-enumeration/experiments/
<new-census-unit-id>/unit.md
```

It should freeze the input split and receipts, Source join, census outputs,
disjointness checks, and decision branches for insufficient strict edges,
singleton-maximal groups, insufficient first-owner diversity, or too many
unknown rows. It does not yet freeze a training score or authorize training.

### Step 2: run and close the trajectory-set census

Build the read-only analyzer and close the census unit. It must answer whether
the frozen training split contains a useful, genuinely multiple-positive set
signal before any training machinery is added.

### Step 3: freeze the training unit

Only after the census closes, create the immutable training unit. It should
freeze:

- owner-set semantics;
- cohort and split policy;
- candidate grouping;
- strict dominance and incomparability;
- length-control choice;
- alias deduplication, all-edge policy, and image-balanced weighting;
- safety and unknown handling;
- primary clean-greedy endpoints;
- recipe-only versus component-attribution claim boundary;
- implementation non-goals; and
- stop/scale decisions.

### Step 4: decide whether OpenSpec is needed

Use OpenSpec only if the pilot requires a stable reusable grouped-record schema,
shared loss interface, canonical training configuration, or artifact contract.
The hypothesis, cohort, thresholds, and scientific verdict remain in
`research/`. Do not create an OpenSpec merely to restate the research unit.

### Step 5: implement the minimum path

Likely reusable surfaces:

- `/data/CoordExp/.worktrees/research-probes/src/training/rollout_calibration.py`
- `/data/CoordExp/.worktrees/research-probes/src/losses/rollout_calibration.py`
- `/data/CoordExp/.worktrees/research-probes/scripts/research/collect_vllm_trajectory_panel.py`
- existing StateBank, packing, checkpoint, inference, and owner-ledger code from
  the July 21 through July 23 screens.

The existing grouped entity-transition preference function may help with a
local auxiliary. It is not the final set-level objective.

The current production objectives cannot be reinterpreted as this treatment:
the positive-path objective requires one positive complete row, and the
grouped entity-transition objective selects one winning owner or alias before
applying continuation cross-entropy. Add a separate opt-in trajectory-set
record and loss path so that configuration, logs, and tests can prove the old
single-row objectives are inactive.

### Step 6: required mechanical tests

Before a GPU training smoke, prove:

1. identical owner sets in different row orders receive equivalent set-level
   supervision;
2. any previously uncovered verified owner can receive positive credit;
3. repeating a covered owner does not increase set reward;
4. a strict owner-set superset dominates its subset when safety is not worse;
5. owner-exchange trajectories are incomparable, not positive versus negative;
6. unknown entities are neutral;
7. premature stop is harmful only when trusted owners remain;
8. length control does not automatically punish the longer valid superset;
9. the resolved configuration proves that set-level loss is active; and
10. no grouped record silently collapses to one positive candidate row;
11. chain, incomparable-orphan, duplicate-alias, and reordered-trajectory
    fixtures consume exactly the declared semantic edges and weights; and
12. a valid first owner that appears only in a dominated route makes that edge
    ineligible until a positive alias covers the same first owner; and
13. perturbing an excluded unknown row leaves primary loss and gradients
    unchanged.

### Step 7: real smoke

Use 8 to 16 images first:

- materialize grouped records;
- run one forward and backward pass;
- verify finite gradients and nonzero set loss;
- save and reload one checkpoint;
- run clean greedy inference;
- inspect gained, retained, and lost owners; and
- confirm that output health did not collapse.

Run the first real smoke early. Do not first build broad manifests, exhaustive
runtime guards, general plugin interfaces, or future-facing abstractions.

### Step 8: 256-image screen

Use the geometry-sorted pure-cross-entropy checkpoint as Source. Freeze the
vision tower and multimodal projector; train language-tower DoRA parameters.
Use low learning rate near the established `1e-5` scale, gradient clipping, and
the largest safe grouped batch or packing configuration after the real smoke.
Do not add Kullback-Leibler regularization in the first treatment unless a
demonstrated preservation failure requires it.

Use all eight graphics-processing units when available and aim for high useful
utilization. Save checkpoints around 30, 60, and 90 percent of the planned
updates plus the terminal checkpoint. Select a milestone on development data,
then read held-out once.

Primary comparison:

```text
frozen Source checkpoint
versus
256-image set-level treatment
```

This comparison measures the efficacy of the complete treatment recipe. It
does not, by itself, identify the set-level loss as the sole cause because the
trained arm also differs in optimizer exposure and candidate composition. If
the screen is promising, either keep the conclusion explicitly at recipe
level or run a matched-budget single-positive trajectory control from the same
candidate groups before claiming that set semantics caused the improvement.
The training unit must freeze that choice before the 256-image launch.

Historical single-owner treatments remain evidence; do not retrain every old
arm unless the matched-control choice requires one exact comparator.

## 12. Evaluation Contract

Evaluate new clean greedy rollouts, not only training loss or teacher-forced
likelihood.

### Primary owner-set outputs

```text
retained Source owners
gained owners
lost owners
net owner-set expansion
greedy unique-owner recall
unique verified owners per emitted row
gap between greedy coverage and sixteen-sample union coverage
```

Report at fixed complete-row budgets where useful, including 4, 8, 16, and 32
rows, while retaining natural closure and invalidity status.

### Safety and quality outputs

```text
physical-owner duplicates
malformed, invalid, dropped, or unparseable rows
confirmed unsupported entities
unknown unmatched predictions
category errors
full-box geometry quality
per-boundary coordinate error
neighbor-instance contamination
natural termination
rollout length and prediction count
```

Mean Average Precision and standard COCO metrics remain useful secondary
summaries. They are not sufficient: mAP can improve while unique-owner coverage
exchanges or declines.

### Interpretation table

| Observation | Interpretation and next action |
| --- | --- |
| Training objective and clean greedy owner set improve without owner loss | Promising; run one refreshed-trajectory round or expand to 1,024 images |
| Selected or sampled-only owners increase but Source owners fall | Owner exchange persists; inspect grouping, dominance, safety, and preservation semantics |
| Training groups improve but clean rollout does not | Offline state-distribution mismatch; run one mixed old-plus-treated-prefix refresh |
| Few strict dominance pairs exist | Data lacks constructive set comparisons; improve exploration, do not revert to unique-row CE |
| Set loss does not improve even on training groups | Implementation, scoring, or optimization failure before a model-capacity claim |
| Clean rollout gains only through more duplicates or longer output | False improvement; reject or narrow |
| Valid set supervision works locally but fails on held-out images | Insufficient transfer; do not scale unchanged |
| Set-aligned native-prefix training fails while an oracle carrier succeeds | Evidence for a missing or inaccessible task-state carrier; then design the smallest explicit carrier |

Promotion from 256 to 1,024 images does not require a rigid arbitrary metric
margin, but it does require a genuinely promising behavioral signature: clean
greedy set expansion, meaningful owner retention, and acceptable output health.
Promotion from 1,024 to full size requires held-out evidence and a successful
real inference smoke from the chosen checkpoint.

## 13. What Not to Do Next

- Do not continue scaling positive-only single-row or single-trajectory
  imitation.
- Do not make one geometry-sorted or sampled owner the unique next target.
- Do not append all missed objects in one fixed teacher-forced tail and call it
  set supervision. Randomizing that tail still optimizes a distribution over
  serialized sequences, not final set probability.
- Do not treat another real uncovered owner as a negative.
- Do not treat official unmatched predictions as hallucinations.
- Do not suppress the terminal token as the main treatment.
- Do not tune repetition penalty as a substitute for training valid-owner
  evidence.
- Do not launch beam search merely because it is more structured; current low-
  temperature trajectories already provide useful exploration. Revisit beam
  search only if the candidate-set census shows a specific exploration gap.
- Do not scale coordinate-only correction.
- Do not assume random row ordering is a set objective.
- Do not build object slots, a persistent ledger, a cursor renderer, or an
  external detector before the set-level loss test.
- Do not use attention maps as causal evidence.
- Do not infer an object owner from `x1` alone.
- Do not interpret more rows, lower STOP probability, or lower training loss as
  improved detection.
- Do not overbuild manifests, resume layers, guards, or generalized interfaces
  before one real smoke proves the required seam.

## 14. Reusable Research Workflow

The project follows an experiment-first gated loop:

```text
mechanism or intuition
  -> explicit falsifiable hypothesis
  -> strongest competing explanation
  -> smallest discriminating research unit
  -> independent scientific review
  -> minimum implementation
  -> synthetic test and one real smoke
  -> bounded GPU experiment
  -> metrics plus crop-assisted visual analysis
  -> observed / supported / ruled out / unresolved / not claimed
  -> belief and decision update
  -> next discriminator or training promotion
```

Architecture emerges from passed gates. It is not fixed before experiments.

### 14.1 Research-unit design

An exploratory unit needs:

1. one falsifiable question;
2. the strongest alternative and a separating control;
3. the smallest observation that changes the next decision;
4. exact checkpoint, cases, changed factor, invariants, and decode semantics;
5. reused infrastructure, non-goals, one real smoke, rough cost, and stop rule;
6. one immutable output root and compact receipt; and
7. a terminology declaration for every abbreviation and arm name.

Do not freeze a speculative code interface before the runtime seam is known.

### 14.2 Sample-size policy

- Mechanism or case study: deliberately select 4 to 8 images; use at most 16
  before deciding whether the phenomenon is worth broader estimation.
- After several tiny units on the same mechanism, estimate prevalence or test a
  training treatment rather than adding another special case.
- Training learnability screen: typically 256 images.
- If promising: 1,024 images, then full size.
- Training data may use 256, 512, 1,024, 2,048, or larger scopes when the
  hypothesis needs it; maximize useful graphics-processing-unit throughput.
- The twelve manually refined images are a safety panel, never gradient data.

Metrics on tiny case panels are descriptive. Visual inspection often carries
more scientific meaning when official labels are incomplete.

### 14.3 Minimal implementation policy

1. Find the narrow existing repository owner.
2. Build only the path required for the primary observation.
3. Run deterministic synthetic fixtures.
4. Run one real model/image smoke immediately.
5. Fix only conclusion-changing blockers.
6. Expand to the declared panel after the seam works.
7. Generalize only after a second real consumer proves the abstraction useful.

One-time experiment code may be ugly if it is correct and bounded. Reliability
of the scientific result matters more than framework elegance. Runtime guards
are justified only for demonstrated silent-failure modes that could change the
conclusion.

### 14.4 Review and audit policy

- Independent reviewers receive the same evidence scope and do not see one
  another's verdict before deciding.
- Separate implementation ownership from scientific judgment.
- Audit only what can change the result, claim boundary, or expensive launch.
- Every finding must lead to `fix`, `narrow`, `drop`, `probe`, or `needs user
  decision`.
- Do not wait for every low-severity improvement before making the experiment
  roll forward.
- Reuse one reviewer for the fixed-point recheck rather than spawning repeated
  general audits.
- Failed or invalid runs remain provenance but never enter the result table.

### 14.5 Subagent delegation policy

- Give each subagent a narrow English brief, exact evidence paths, read/write
  scope, and stop condition.
- Use `fork_turns="none"` for self-contained discovery, bounded
  implementation, and independent audit; use one to three recent turns only
  when the task genuinely depends on recent decisions.
- Use cheaper models for mechanical discovery and tested implementation;
  reserve the strongest reasoning for conclusion-critical science,
  architecture, and final adjudication.
- Use one implementation owner per surface. Parallelize independent discovery,
  artifact, implementation, and audit lanes rather than duplicating writers.
- The parent agent synthesizes disagreements and owns the final scientific
  decision.

### 14.6 Evidence and reporting policy

Every closed nontrivial unit separates:

```text
Observed
Supported
Ruled out
Unresolved
Not claimed
```

After closure:

- update only beliefs changed by the result;
- retain rejected explanations and why they failed;
- link metrics instead of copying every table into the compass;
- identify the strongest surviving alternative;
- name the next discriminator; and
- keep research decision, implementation authorization, architecture
  promotion, and stable OpenSpec contract as separate gates.

## 15. Numerical and Runtime Discipline

- Use full-model `float32` for delicate causal logits, residual patching,
  layerwise comparisons, and small coordinate-margin claims.
- Batch decoding can introduce small numerical variation. It is acceptable for
  high-throughput support estimation when sample/seed exact parity is not the
  research variable.
- For exact causal comparisons, require no-operation parity and semantically
  equivalent batch execution.
- Record prompt, checkpoint, image, precision, decode policy, maximum length,
  row budget, and parser/closure status.
- A request banner is not evidence that the runtime used the requested value.
  Verify persisted runtime configuration. The first repetition-penalty-1.10
  panel was invalid because the collector actually used 1.0.
- Use immutable run identifiers. A repaired run gets a new identifier.
- Preserve truncation and malformed outputs in denominators; do not silently
  drop them to improve metrics.

## 16. Annotation and Visual-Review Standard

Official COCO labels are incomplete, especially in dense scenes. Use two
independent judgment axes.

### Entity/category axis

1. labeled or human-confirmed true physical owner;
2. real but officially unlabeled true owner;
3. duplicate of an already represented owner;
4. semantic category error;
5. unsupported entity hallucination; or
6. uncertain.

### Geometry axis

1. acceptable visible physical extent;
2. shifted, oversized, undersized, or incomplete localization;
3. neighboring-instance contamination or a box between instances; or
4. uncertain.

A real correctly categorized owner with imperfect geometry retains entity
credit. It is not an entity hallucination.

Use crop-enlarged inspection for low-pixel, occluded, overlapping, or cross-
category cases. Ordinary resize interpolation is allowed to make existing
pixels easier to inspect. Do not use synthetic super-resolution as new
evidence. Inspect the full image before judging a crop so global context is
retained.

## 17. Scale, Stop, and Architecture Gates

### Stop, narrow, or redirect when

- the candidate data contains no meaningful strict set relations;
- mechanics or identity checks fail;
- the proposed loss does not actuate on training groups;
- wrong-object or generic controls reproduce the effect;
- clean rollout does not improve despite better local scores;
- gained owners are offset by lost owners;
- output grows mainly through duplicates, invalid rows, or hallucinations;
- improvement exists only on gradient images;
- a stronger baseline or changed decode policy explains the result; or
- further work would only add images, epochs, seeds, or knobs without new
  information.

### Scale when

- set-level supervision is mechanically active;
- clean greedy rollout gains trusted owners while retaining Source owners;
- duplicates, invalid rows, semantic stability, geometry, and closure remain
  acceptable;
- the effect survives development and held-out review;
- the result is not caused by a weaker Source baseline or decode-policy change;
  and
- an independent audit accepts the claim boundary.

### Consider an explicit carrier only when

1. sampled or controlled evidence proves the owners are available;
2. true set-level supervision is correctly implemented and learnable;
3. native-prefix training still cannot produce safe set expansion; and
4. an oracle coverage or task-state carrier improves the same objective under
   matched supervision.

Only then decide whether the smallest useful carrier is a visual coverage map,
compact task state, recurrent owner memory, feature-space writeback, or another
mechanism. Slots are not the default answer.

## 18. Key Evidence Reading Path

### Program routers

1. `/data/CoordExp/.worktrees/research-probes/handoff/2026-07-23-qwen3-vl-dense-enumeration-comprehensive-research-flow.md`
2. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/compass.md`
3. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/index.md`
4. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/overview.md`
5. `/data/CoordExp/.worktrees/research-probes/memories/current.md`

### Current treatment correction

1. `/data/CoordExp/.worktrees/research-probes/handoff/from-side-chat.md`
2. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/source-b16-vllm-receipt.md`
3. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md`
4. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/results.md`
5. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/results.md`
6. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/results.md`

### Mechanism chain

1. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/2026-07-13-to-2026-07-16-weekly-research-report.md`
2. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md`
3. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-15-prefix-state-phrase-geometry-factorial/results.md`
4. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-19-same-covered-set-prefix-order-equivalence/results.md`
5. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-19-sampled-history-target-reachability-and-complete-row-value/results.md`
6. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-individual-trajectory-versus-union-support-audit/results.md`

### Annotation and geometry

1. `/data/CoordExp/.worktrees/research-probes/handoff/2026-07-21-sampled-trajectories-to-greedy-set-coverage-and-geometry-ambiguity.md`
2. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/results.md`
3. `/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/results.md`

### Runtime and reusable code

1. `/data/CoordExp/.worktrees/research-probes/src/training/rollout_calibration.py`
2. `/data/CoordExp/.worktrees/research-probes/src/losses/rollout_calibration.py`
3. `/data/CoordExp/.worktrees/research-probes/scripts/research/collect_vllm_trajectory_panel.py`
4. `/data/CoordExp/.worktrees/research-probes/.codex/skills/coordexp-vllm-mechanistic-loop/SKILL.md`
5. `/data/CoordExp/.worktrees/research-probes/.codex/skills/coordexp-research-knowledge-workflow/references/research-graph-contract.md`

## 19. Immediate Fresh-Session Task

The next session should not reopen the architecture debate first. Its initial
job is:

1. read this handoff and `handoff/from-side-chat.md`;
2. inspect live worktree status and preserve all unrelated dirty changes;
3. draft and freeze a read-only admission-census research unit;
4. run that census only on Source-eligible members of the frozen 2,048-image
   training split, measuring strict dominance, positive order aliases,
   first-owner diversity, singleton-maximal groups, and unknown-row exclusion;
5. close the census, then freeze a separate 256-image training unit, trajectory
   score, weighting, data contract, and recipe-level claim boundary;
6. decide whether a small OpenSpec is required for the grouped training record
   and loss seam;
7. implement the smallest path and execute the thirteen mechanical tests;
8. run an 8-to-16-image real smoke; and
9. only after smoke acceptance, launch the 256-image eight-graphics-processing-
   unit treatment screen.

The first question is not “what final architecture should we build?” It is:

> Can true order-independent final-set supervision reshape Qwen3-VL's existing
> distributed prefix state so that one greedy trajectory gains owners without
> exchanging away owners it already knew how to find?

## 20. Open Questions to Preserve

- Does the stored sampled panel contain enough strict set-dominance examples,
  or must exploration be improved?
- What length-controlled trajectory score best preserves the final-set partial
  order without rewarding short easy sequences?
- After the root-state screen, do exact naturally shared nonempty prefixes add
  enough signal to justify a separate extension?
- How much of the owner exchange came from objective semantics versus offline
  prefix-distribution mismatch?
- Can final-set supervision use the native distributed prefix state, or will an
  explicit task-state carrier become necessary?
- How should Source retention be represented without making the Source route a
  unique canonical teacher?
- How should entity credit and geometry uncertainty be combined without
  allowing badly localized rows to dominate the set reward?
- Can the sampled-union gap shrink under greedy decoding while preserving the
  model's low unsupported-hallucination behavior?
- Does more complete dense-scene annotation materially alter stopping and
  coverage learning?
- If the 256-image screen succeeds, does one refreshed trajectory round improve
  transfer before scaling to 1,024 images?

## Final Research Position

The project has not discovered a final detector architecture, and it should not
pretend otherwise. It has established a more useful boundary:

```text
the model can see and transcribe many of the missing physical owners
  + sampled trajectories expose complementary reachable owner sets
  + prefix rows causally change later routing
  + local owner-directed credit clearly changes behavior
  - current sequence objectives reward one route or owner at a time
  - owner gains are often paid for by owner losses
  - no stable order-free coverage carrier is demonstrated
```

The next scientific move is therefore neither blind scaling nor an immediate
slot-based redesign. It is the first honest test of the objective we actually
care about: final unique physical-owner set expansion under one clean greedy
rollout.
