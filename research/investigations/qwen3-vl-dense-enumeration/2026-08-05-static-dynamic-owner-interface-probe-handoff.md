---
title: Static-Dynamic Owner Interface Probe Handoff
description: Continuation brief for the no-training probe unit that will decide how post-LLM image states and autoregressive history should jointly shape dense enumeration.
type: handoff
role: continuation-brief
authority: transport-only
status: ready-for-transfer
updated: 2026-08-05
---

# Static-Dynamic Owner Interface Probe Handoff

## 1. Authority, scope, and handoff contract

This document is a transport artifact, not a live research authority and not an
execution approval. It packages the current question, evidence, proposed probe
contract, and recovery information so a fresh agent can continue without
reconstructing the discussion.

The current authority order is:

1. the user's latest instruction and any subsequent explicit decisions;
2. the live investigation route in [compass.md](compass.md);
3. applicable records under `research/decisions/`;
4. the future formal `unit.md`, once explicitly authorized and created;
5. executed results and sealed external artifacts;
6. this handoff and advisory reviews.

If this handoff conflicts with a later user decision, live compass, or approved
unit, reconcile the conflict and update the owning artifact. Do not silently
treat this handoff as authority.

### Authorized in the turn that created this handoff

- Add this one standalone handoff document.
- Inspect existing artifacts read-only to make the handoff recoverable.

### Not authorized by this handoff

- Creating or executing the proposed formal probe unit.
- Modifying the investigation compass, index, memory, OpenSpec, scripts,
  configs, model code, data, or checkpoints.
- Running GPU inference, backpropagation, training, or a production launch.
- Adding, removing, or minimizing special tokens or changing the full wrapper.
- Staging, committing, pushing, cleaning, resetting, or otherwise reconciling
  the already-dirty worktree.

The next agent must obtain or inherit an explicit instruction to create the
unit or run it before crossing those boundaries.

### Session provenance

Use these Codex session IDs only to recover discussion context that is not yet
owned by a durable research artifact:

```text
current static/dynamic probe-design session:
019fd043-5abb-7eb1-997c-d11a8316281a

sorted-with-commit / owner-commit experiment session:
019fca99-d2b3-7812-b89f-ec45471b90d7
```

Session history is provenance, not live authority. Reconcile any recovered
claim against the current compass, approved unit, runtime identity, and sealed
artifacts before using it in a decision.

## 2. Current objective

Design one bounded, no-training probe unit that decides what the next
large-scale production training should change at two coupled surfaces:

1. **STATIC surface:** image-position representations after they have passed
   through the LLM tower, including the per-layer image K/V cached during
   prefill;
2. **DYNAMIC surface:** the current query plus autoregressive row history that
   selects, commits, suppresses or reallocates among owners, tracks coverage,
   and decides when to stop.

The north-star behavioral loop is:

```text
static owner inventory / addressable visual field
                    ↓
dynamic explore → select → commit → suppress or reallocate
                    ↑                         ↓
                    └──── repeat until no supported owner ────→ STOP
```

The unit must determine whether the next intervention should supervise a
scalar occupancy/density field, a foreground/class field, an owner-addressable
visual field, dynamic coverage behavior, or a coupled static-dynamic mechanism.
It must not assume the answer in advance.

## 3. User-level invariants and corrected focus

These constraints are fixed for this probe design unless the user explicitly
reopens them:

1. Preserve each checkpoint's native **full wrapper**. Do not revisit
   special-token compression or minimality in this unit.
2. The main question is not whether a special token can be decoded by a probe.
   It is what post-LLM image representations and their cached K/V contribute to
   free decoding, and how history consumes that substrate.
3. Treat static image state and dynamic query/coverage state as separate but
   coupled responsibilities.
4. Prefer full released rows and short self-rollout consequences over
   teacher-forced-only or segment-only success.
5. Do not impose a canonical object order as the mechanism. The model may
   choose an order, but it must gain unique owners without replacement and stop
   for the right reason.
6. Plain sorted and sorted-with-commit/A3 are both useful, but they are not a
   matched causal pair. Compare within-checkpoint intervention effects; do not
   attribute raw cross-checkpoint differences to `<|commit|>` or InfoNCE.

## 4. Why this work belongs in `research-probes`

Use this worktree:

```text
/data/CoordExp/.worktrees/research-probes
```

This is the correct owner because it already contains the investigation's
decision chain, exact-prefix HF probe infrastructure, owner matching, regional
image-key eligibility interventions, fixed-encoding residual replacement, the
admitted 13-image panel, and the recent sorted mechanism units. The proposed
work is a decision-bearing causal probe, not yet a training implementation.

The additive-density worktree/OpenSpec remains a **candidate downstream
treatment**, not the owner of this discriminator. In particular, a previously
fixed block-23 scalar density target must not become the default merely because
it is implementation-ready. This probe may support it, rescope it, or veto it.
Do not edit that OpenSpec as part of this handoff or future probe execution
unless separately authorized.

## 5. Checkout identity and dirty-worktree safety

Snapshot at handoff creation:

```text
worktree: /data/CoordExp/.worktrees/research-probes
branch:   research-probes
HEAD:     723c3ef92ec2e6c9d97b702165efb890193ccdb3
subject:  Add coordinate confidence replay visualizers
```

This checkout is heavily dirty. Existing modified and untracked files include
the investigation compass/index, memory, inference configs, current August
research units, scripts, tests, and rendering code. They belong to the user or
other active work. Therefore:

- run `git status --short` before every write phase;
- never use `git reset`, `git clean`, destructive checkout, or bulk staging;
- do not overwrite or normalize existing dirty files;
- place the future unit in a fresh, uniquely named directory;
- stage nothing unless the user later requests it, and then stage explicit
  paths only;
- revalidate branch, HEAD, and target paths because this snapshot can drift.

This handoff intentionally does not modify `compass.md`, `experiments/index.md`,
`memories/current.md`, configs, scripts, or tests.

## 6. Minimum reading path for a fresh agent

Read in this order before drafting or executing the formal unit:

1. [this handoff](2026-08-05-static-dynamic-owner-interface-probe-handoff.md);
2. [compass.md](compass.md) for the current live route;
3. [overview.md](overview.md) for investigation-level scope;
4. [require-target-specific-causal-consumption.md](../../decisions/require-target-specific-causal-consumption.md);
5. [separate-selection-transcription-commit-and-stop.md](../../decisions/separate-selection-transcription-commit-and-stop.md);
6. [sorted owner-accessibility census results](experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/results.md);
7. [sorted crossing-boundary results](experiments/2026-08-03-sorted-crossing-boundary-owner-release-realization/results.md);
8. [prospective 13-image panel admission unit](experiments/2026-08-04-sorted-prospective-13-image-panel-admission/unit.md);
9. [sealed A3 P0 results](/data/CoordExp/.worktrees/owner-commit-binding/outputs/probes/coordexp_swift/frozen_owner_set_probe/a3_step2445/finalize/sealed/results_final.md);
10. the exact runtime configs, scripts, and checkpoints listed below.

Do not broaden into the full historical corpus before this route has been read.
If the live compass has moved, reconcile that change first.

## 7. Evidence that motivated this route

### 7.1 Plain sorted mechanism evidence

Current sorted evidence supports a present-but-not-reliably-emitted phenotype:

- many false-negative owners have tested localization support;
- many have an open continuation gate at some tested boundary;
- crossing failures can be displaced toward another uncovered owner rather
  than simply lacking visual information;
- exact-prefix, native-KV and late residual interventions already provide
  useful actuator seams;
- none of this demonstrates an order-free coverage ledger.

The exact counts and admission rules must be taken from the current executed
units rather than copied from this handoff if they become claim-bearing.

### 7.2 A3 owner-commit evidence

The sealed A3 P0 reports:

- 221 trajectories over 13 images, with natural `<|im_end|>` termination;
- only 8 malformed rows among 3,521 valid object rows;
- image 2299 greedy coverage of `14/46`, increasing to `34/46` in the K16
  sampled-trajectory union;
- legacy-12 greedy coverage of `114/346`, increasing to `172/346` in the K16
  union;
- 2,046 matched commit observations;
- owner retrieval@1 of `75.27%`, and same-class retrieval@1 of `71.30%`;
- commit states more owner-local than `<|box_end|>` or the last coordinate
  controls.

These results establish that the generated `<|commit|>` state is a useful
owner-local readout and that alternative trajectories can access many greedy
misses. They do **not** establish that commit history causally suppresses the
covered owner, reallocates mass to an uncovered owner, or implements stopping.

The retrieval prototypes in that experiment are pre-LLM merger-output visual
features. Therefore the result does not answer whether post-LLM image-position
representations or cached image K/V are owner-addressable and behaviorally
consumed.

There is no decision-grade matched A2 checkpoint in the current evidence set.
The available A2 artifact is a preparation/smoke, not an attribution control.
Consequently A3 is a sensitized comparison checkpoint, not proof that InfoNCE
or the commit token caused its phenotype.

### 7.3 What random exploration does and does not show

K16 union recall shows that the checkpoint contains multiple reachable basins.
It is not the desired decoder:

- it incurs large unmatched and duplicate burdens;
- it does not provide a reliable accept/reject rule;
- it does not prove stable without-replacement behavior;
- a union over many trajectories cannot own the production claim.

Use existing sampled trajectories to select informative cases. Do not rerun a
large K sweep as the main probe.

## 8. First-principles responsibility model

For a fixed image and prompt, image positions are computed through every LLM
block during prefill and their per-layer K/V are cached. Later assistant tokens
can read those cached states, but cannot rewrite the earlier image positions.
This gives the two surfaces different natural duties:

| Surface | Should own | Should not be required to own alone |
|---|---|---|
| Static post-LLM image residual/K/V | existence of candidate owners; spatial and semantic payload; owner separability/addressability; a salience surface that queries can access | current covered set; row order; history-dependent suppression; final stopping decision |
| Dynamic current query and row history | exploration and selection; commitment; suppression or reallocation after coverage; path/set state; continuation and STOP | reconstructing visual owners that are absent from the static substrate |
| Static-dynamic interface | whether the current query can route to an uncovered owner and consume its description/geometry payload | a claim based only on linear readout, attention visualization, or teacher-forced likelihood |

Before the first assistant token, the static field should already contain an
addressable candidate inventory at the granularity the model is expected to
enumerate. The row-start query does not need to be a one-hot owner immediately.
It may remain distributed, then progressively collapse through description and
geometry, becoming sharp by row completion/commit. This reconciles a
detector-like visual substrate with autoregressive exploration.

Salience is the coupling surface: a large or easy owner may dominate query-key
competition even when other owners remain represented. A static objective is
useful only if it changes which owners the dynamic query can reliably consume,
not merely if it makes an auxiliary label decodable.

## 9. Competing causal explanations

The unit should adjudicate at least these explanations:

### H1: static owner information is absent or not separately addressable

Missed owners, especially same-class or overlapping instances, do not occupy a
usable post-LLM image carrier. Better dynamic decoding cannot recover what is
not present at the tested interface.

### H2: static owner information exists but loses salience/routing competition

The owner is present and can causally support a row, but the current query
under-allocates mass to it. Static field shaping or owner-selective image-KV
routing is the main treatment.

### H3: dynamic history fails to implement coverage

Image K/V are sufficiently addressable, but committed rows do not suppress the
covered owner, do not reallocate to remaining owners, or do not calibrate STOP.
The main treatment is multi-row rollout/coverage behavior.

### H4: the bottleneck is coupled

Static owner accessibility and dynamic suppression/reallocation interact. A
joint training arm is justified only if their causal combination gains and
retains owners better than either alone.

### H5: neither tested interface is causal

If qualified static and dynamic actuators both fail, the next route should
move toward search/on-policy/final-set objectives or a different interface,
not add more frozen-state labels by default.

The strongest alternative to this probe is a cheap scalar-density training
pilot. It remains reopenable because frozen-state affordance does not guarantee
trainability. It is not the current first move because one scalar pilot only
tests one assumed treatment, while earlier representational improvements have
not guaranteed causal consumption and can trade recall for duplication or
precision.

## 10. Proposed formal unit

After explicit authorization, create:

```text
research/investigations/qwen3-vl-dense-enumeration/experiments/
  2026-08-05-static-dynamic-owner-interface-crossover/
    unit.md
```

Proposed unit ID:

```text
2026-08-05-static-dynamic-owner-interface-crossover
```

Proposed external artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-05-static-dynamic-owner-interface-crossover/<run-id>/
```

Do not create this directory or formal unit merely because it is named here.
The future `unit.md` must freeze the exact cohort, runtime identity,
interventions, endpoints, thresholds, resource budget, and stop rules before
execution.

## 11. Runtime identities to freeze

### 11.1 Plain sorted checkpoint: primary substrate

Inference config currently available:

```text
/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/
qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined13_hf_fp32_rp1p0.yaml
```

Adapter:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/
checkpoints/step-4887/adapter
```

Special-token embeddings:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/
checkpoints/step-4887/special_token_embeddings
```

Re-resolve the exact base-model identity from the live config before execution;
do not infer it only from checkpoint naming.

### 11.2 A3 checkpoint: sensitized comparator

Base model:

```text
/data/Qwen3-VL/model_cache/models/Qwen/
Qwen3-VL-2B-Instruct-coordexp-owner-commit-natural-adjacent
```

Adapter:

```text
/data/CoordExp/.worktrees/owner-commit-binding/outputs/prod/coordexp_swift/
owner_commit_patch_binding/a3_patch_binding_4epoch/checkpoints/step-2445/adapter
```

Special-token embeddings:

```text
/data/CoordExp/.worktrees/owner-commit-binding/outputs/prod/coordexp_swift/
owner_commit_patch_binding/a3_patch_binding_4epoch/checkpoints/step-2445/
special_token_embeddings
```

The primary matched mechanism comparison should use common HF fp32/SDPA greedy
decode with `repetition_penalty=1.0` and `max_new_tokens=3084`, while preserving
each checkpoint's native full wrapper. The sealed A3 `rp=1.10` result remains a
separate sensitivity result, not the primary matched decode. If the A3 rp1.0
config is missing, derive it only after the unit contract authorizes that write.

### 11.3 Shared admitted panel

Input:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-04-sorted-prospective-13-image-panel-admission/
evaluation-inputs/human-refined-13.coord.jsonl
```

Admission receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-04-sorted-prospective-13-image-panel-admission/receipt.json
```

Always report legacy-12 and image 2299 separately before any pooled summary.

## 12. H0: substrate and cohort admission

H0 should reuse existing artifacts and fill only missing matched cells. Its
purpose is to prevent the causal stages from being confounded by checkpoint,
decode, panel, wrapper, or owner-identity drift.

### 12.1 Freeze both checkpoint families

- Plain sorted step-4887 is the primary native reference.
- A3 step-2445 is a sensitized contrast.
- Do not compare raw latent coordinates, raw logits, or headline recall across
  them as if they differed only by commit binding.
- Compare intervention-minus-own-baseline deltas inside each checkpoint.
- If an A3-only effect becomes the premise of a production decision, a matched
  A2/equal-schema control becomes conditionally mandatory before attribution.

### 12.2 Freeze exact prefixes and wrappers

- Record token IDs, wrapper grammar, prompt identity, image-token spans,
  position IDs/MRoPE metadata, generated row boundaries, and hashes of the
  exact serialized prefixes.
- Natural self-generated prefixes own conclusions.
- Forced clean rows or forced owner descriptions are diagnostic controls only.
- Preserve the full wrapper for plain sorted and the full wrapper plus
  `<|commit|>` for A3.

### 12.3 Select a small fixed causal cohort

Target roughly 24–32 owner events from 6–8 images if the existing artifact
coverage supports it. A smaller cohort is acceptable for an exploratory first
pass, but the unit must state the limit rather than silently generalize.

Stratify across:

- native true-positive owners for actuator qualification;
- supported owners missed by plain sorted greedy decode;
- A3 greedy misses recovered in an existing sampled trajectory;
- owners never recovered in the existing K16 set as negative candidates;
- same-class, spatially separated owners;
- same-class overlapping or image-token-colliding owners;
- cross-class sanity controls.

Potential dense-scene cases already surfaced include image 14038 (books), 4134
(persons), 7511 (tiny persons/token collision), and 6040 (truck/car). These are
leads, not frozen owner IDs. Re-derive every owner ID, class, box, token support,
and native status from the current ledgers before admission. Do not copy stale
indices from an advisory memo.

For overlapping owners, split the image-token support into A-exclusive,
B-exclusive, and shared-core regions. A shared cell cannot establish
owner-specific causality by itself.

## 13. P1: static post-LLM image-field probe

### 13.1 Descriptive census

Capture image-position residual states at:

- merger output / pre-LLM reference;
- decoder block-0 input after scatter;
- every decoder block output, zero-based blocks 0 through 27;
- final norm.

This all-layer capture is observational. It can locate where foreground,
density, class, geometry, or owner identity becomes readable, but it cannot
decide the training target by itself.

Candidate readouts:

- foreground and `log1p` fractional occupancy/density;
- density-conditioned multi-label class information;
- owner prototypes using fractional GT-box-to-merger-cell overlap;
- same-image and same-class owner retrieval;
- hardest-negative owner margin;
- coordinate-only, area/density-only, shuffled-owner, and cross-image geometry
  controls.

Prevent leakage: shared cells must not appear simultaneously in both a query
and its owner prototype when evaluating overlapping owners.

### 13.2 Causal layers

Run causal residual interventions only at:

- block 23: primary evidence-backed late seam;
- block 13: preregistered earlier comparator;
- block 27: mechanical no-effect control, because a returned post-final-block
  image residual has no later decoder block that can consume it.

Do not launch a broad layer/head sweep. If block 27 changes generation, treat
the instrumentation or seam definition as invalid.

### 13.3 Causal arms

Reuse or minimally extend existing exact-prefix tools for:

1. native baseline;
2. byte-identical self/no-op intervention;
3. row-query-only target-region image-key eligibility/spotlight;
4. removal of keys belonging to an already covered owner;
5. same-class competitor-region spotlight;
6. equal-area random or background-region control;
7. block-13 and block-23 norm-matched background replacement of an
   owner-exclusive image residual;
8. A/B owner-exclusive residual swap while keeping spatial token positions and
   MRoPE metadata unchanged;
9. shared-core knockout, interpreted only as density/region evidence rather
   than owner-identity evidence.

Use native true positives to qualify the actuator: removing or replacing a
trusted target's support must reduce that target's realization. A readout can
be strong while the chosen intervention surface is behaviorally inert.

### 13.4 Primary endpoints

The primary endpoint is a **source-specific complete released row**:

- correct owner description/class evidence;
- all required coordinate tokens;
- valid wrapper closure;
- `<|commit|>` included for A3;
- strict match to the intended physical owner, not merely the same category or
  overlapping region.

Also report:

- description-only and geometry-only likelihood/margins;
- target versus best uncovered-owner margin;
- target versus covered-owner margin;
- row-entry versus `<|im_end|>` margin;
- non-target valid-owner damage;
- malformed, unmatched, and duplicate outcomes.

Teacher-forced likelihood, linear retrieval, and attention allocation are
secondary. They cannot pass P1 without transfer to a complete released row.

### 13.5 P1 trust gate

- Exact self/no-op generation must be identical.
- Selected-token numeric drift must remain within the already validated
  harness tolerance; `1e-4` is the current conservative candidate and must be
  confirmed in the formal unit.
- Native-TP removal must move the target in the expected direction, or the
  actuator is unqualified.
- An owner-addressability conclusion must reproduce on at least two images and
  include at least one same-class overlap case.
- Do not freeze an effect-size threshold from an advisory note without the
  formal unit owner's approval. Directional consistency and strict owner
  transfer matter more than a post-hoc scalar threshold.

## 14. P2: dynamic history/coverage probe

P2 asks whether completed-row history is suppressive, reallocating,
reinforcing, idle, or distributed. Keep the token sequence and current boundary
fixed while altering the cached/hidden contribution of the completed row.

### 14.1 A3 arms

At an exact natural boundary immediately after a committed row:

1. native commit/history intact;
2. final commit state muted or replaced with a norm-matched mean;
3. same-image, same-class A→B commit donor at the same destination position;
4. whole latest completed-row state muted;
5. equal-length earlier-row state muted as a distributed-history control;
6. whole-row donor/replacement as an actuator positive control.

The intervention must preserve token IDs, row length, destination positions,
and wrapper. If a pre-RoPE K/V replacement is later implemented, reposition it
correctly rather than swapping a cached positional phase blindly.

### 14.2 Plain sorted analogue

Plain sorted has no commit token. Use the last `<|box_end|>` and the latest
complete-row history as the analogous boundary carrier, with last-coordinate,
equal-length earlier-row, and whole-row controls.

This is not a claim that `<|box_end|>` and `<|commit|>` are semantically
equivalent. It is the least-confounded native comparison available without
retraining or altering the wrapper.

### 14.3 Prefer existing intervention seams first

Use block-23 residual capture/replacement and current exact-prefix history
masking/eligibility mechanisms before developing exact K-only/V-only swap
infrastructure. Exact K/V decomposition is warranted only if an existing
causal intervention is positive and whether address lives in K versus payload
in V would change the training objective.

### 14.4 Dynamic endpoints and classification

Measure, before and after releasing one complete row:

- probability/margin and strict realization of just-covered owner A;
- probability/margin and strict realization of trusted uncovered owner B and
  all other verified uncovered owners;
- row-start versus `<|im_end|>` margin;
- repeat, duplicate, unmatched, and malformed outcomes.

Classify the native state as:

| Class | Required causal pattern |
|---|---|
| Suppressive/reallocating | Compared with mute/swap, it lowers the committed owner and raises or realizes at least one still-uncovered owner. |
| Reinforcing | It raises the committed owner or repeat probability. |
| Idle | Neither commit-only nor whole-last-row intervention yields an owner-selective effect. |
| Distributed | Commit-only is inert but the whole latest row is owner-selectively causal. |

No class passes on a teacher-forced margin alone. The effect must transfer to a
complete released row on a natural prefix.

## 15. P3: static-dynamic 2x2 crossover

At a natural boundary where owner A is covered and trusted owner B remains
uncovered, cross two factors:

| Cell | Static image field | Dynamic completed-row state |
|---|---|---|
| 1 | native | intact |
| 2 | B region boosted, or covered-A keys removed | intact |
| 3 | native | commit/latest-row neutralized |
| 4 | same static intervention as cell 2 | same dynamic neutralization as cell 3 |

The exact static arm must be selected from a P1-qualified actuator. The exact
dynamic arm must be selected from a P2-qualified actuator. Do not interpret a
2x2 whose components failed their trust gates.

Release one row first for a local effect, then three rows or natural stop for
stability. On admitted cases, add the diagnostic order control:

```text
F_AB: identical natural boundary plus two clean complete covered rows A then B
F_BA: the same covered-owner set in order B then A
```

Forced AB/BA prefixes diagnose set-versus-path sensitivity. They do not replace
natural-prefix evidence.

### 15.1 Primary outcome vector

Report per event and aggregated within each checkpoint:

- whether target B is newly gained;
- which originally reachable suffix owners are retained or lost;
- newly covered unique owners per released row;
- repeat hazard for covered A at rows `t+1`, `t+2`, and `t+3`;
- duplicate, unmatched, malformed, and valid-row burden;
- premature STOP and over-continuation after supported owners are exhausted;
- final gained-minus-lost unique owners.

Stable without-replacement behavior requires more than one-step rescue:

1. newly committed owners remain suppressed at later boundaries;
2. mass reallocates to verified remaining owners, not merely to generic
   continuation;
3. AB and BA lead to similar remaining-owner sets even if the next row differs;
4. STOP rises only as supported remaining owners are exhausted;
5. the three-row gained-minus-lost owner direction is positive.

A positive static-by-dynamic interaction justifies a small matched combined
training arm. A null or negative interaction requires staging or retiring the
treatments separately.

## 16. P4: gradient-path audit before any training pilot

Only candidate objectives selected by P1–P3 should enter this stage. On exact
native prefixes, perform a first-order/gradient audit that checks whether the
candidate objective:

- raises the intended uncovered owner;
- lowers a trusted duplicate/covered owner when appropriate;
- preserves other valid uncovered owners;
- preserves source owners, grammar, calibrated STOP, and invalid-output mass;
- reaches the intended post-LLM image/query interface instead of changing only
  the LM head or a convenient readout slot.

P4 is still a diagnostic, not authorization to optimize the model. If it
passes, the next request may specify a small matched 2x2 training pilot:

```text
baseline | static only | dynamic only | combined
```

No large production run should start before the selected causal route and its
gradient path both pass.

## 17. Outcome-to-training decision table

| Probe outcome | Next training design |
|---|---|
| Static addressability fails, especially for never-recovered or same-class overlap owners | Build a higher-dimensional post-LLM owner-addressable field; scalar density is insufficient. |
| Cross-class effects pass but same-class owner crossover fails | Train an instance/geometry-aware field rather than class-only supervision. |
| Only foreground/density is causal, changing row-entry/STOP or candidate mass but not A-vs-B identity | Use scalar occupancy/density as an inventory or remaining-mass signal; do not claim owner binding. |
| Owner carrier is causal but salience buries weak owners; target spotlight or covered-key removal rescues them | Train owner-selective query-key routing or salience allocation; equal-per-owner density is a candidate auxiliary, not yet the mechanism. |
| Static access is good but completed-row history is idle | Prioritize dynamic rollout/coverage state; do not add more static labels by default. |
| Commit swap transfers owner-specific suppression/reallocation | Treat contextualized history as a causal carrier; train its consumption against contextualized image K/V and test multi-row stability. |
| Suppression works for one row but not three | Use on-policy multi-row or final-set gained/retained/lost credit; another local commit loss alone is insufficient. |
| History reinforces the covered owner | Add explicit covered-owner rejection and remaining-owner reallocation; do not merely reward continuation. |
| Static and dynamic arms show positive interaction and positive three-row net utility | Authorize a small matched combined pilot. |
| Interaction is null or negative | Stage static and dynamic pilots separately or retire the weaker route. |
| Local metrics improve but final gained-minus-lost is non-positive, or invalid/duplicate/STOP safety fails | Reject production promotion. |
| Neither qualified causal handle moves complete rows | Move to objective/search/on-policy alternatives or a different interface; do not add frozen-state shaping by default. |

If only A3 shows the effect and the production claim depends on it, create and
evaluate a matched A2/equal-schema arm before attributing the effect to InfoNCE
or owner-commit binding:

- `A2 ≈ A3` suggests token/schema/recipe learning rather than InfoNCE;
- `A3 > A2` with `A2 ≈ plain` makes InfoNCE a viable causal candidate;
- no A2 is required merely to run the within-checkpoint frozen causal screen.

## 18. Stop rules

Stop the unit or route immediately under the corresponding condition:

1. **Instrumentation invalid:** exact-prefix mismatch, non-identical self/no-op
   generation, excessive no-op drift, or a block-27 post-state intervention
   changes behavior.
2. **Actuator invalid:** removing a native TP's trusted support does not reduce
   its realization while the claimed actuator is otherwise treated as causal.
3. **Static route null:** source-specific complete-row effects are absent at
   both block 13 and block 23. Do not answer with a layer/head/temperature/panel
   sweep.
4. **Generic continuation only:** an intervention only increases output length,
   row-start probability, unmatched rows, or late STOP without gaining the
   intended physical owner.
5. **Local-only dynamic gain:** one-row rescue disappears by three rows or
   merely displaces previously reachable suffix owners.
6. **Set utility failure:** three-row gained-minus-lost unique owners is null or
   negative, or duplicate/invalid burden erases the gain.
7. **Forced-prefix only:** the effect appears under a clean forced prefix but
   not under a natural self-prefix. Record the diagnosis; do not promote it.
8. **Readout-only:** representation retrieval, attention maps, teacher-forced
   likelihood, or K16 union improves without free-row causal transfer.

Do not let any single local proxy own a production decision.

## 19. Reuse map and tooling gaps

Prefer existing mechanisms before adding infrastructure:

- residual-state capture/replacement:
  [run_fixed_encoding_downstream_residual_state_portability.py](../../../scripts/research/run_fixed_encoding_downstream_residual_state_portability.py);
- object-centered regional image-key eligibility:
  [run_fixed_encoding_object_centered_spatial_eligibility_crossover.py](../../../scripts/research/run_fixed_encoding_object_centered_spatial_eligibility_crossover.py);
- shared intervention helpers:
  [intervention.py](../../../src/analysis/visual_support_counterfactual/intervention.py).

Resolve these links and confirm their current CLI/runtime contracts before
reuse; existing dirty edits may have changed them.

Known gap: there is no established general, exact K-only/V-only swap seam that
preserves positional semantics across arbitrary image/history donors. Do not
build it first. Existing residual replacement and query-key eligibility should
decide whether deeper K/V decomposition is decision-bearing. If it becomes
necessary, operate before positional rotation or correctly reapply destination
MRoPE; blindly swapping cached post-RoPE K introduces a position confound.

## 20. Expected artifacts once execution is authorized

The future formal unit should define and produce, at minimum:

```text
unit.md

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-05-static-dynamic-owner-interface-crossover/<run-id>/
    runtime_identity.json
    cohort.json
    exact_prefix_manifest.jsonl
    intervention_manifest.json
    per_event_results.jsonl
    summary.json
    results.md
```

Names may be adjusted by the unit owner, but the artifacts must preserve:

- checkpoint/config/base/adapter/embedding hashes or stable identities;
- exact prompt and wrapper identity;
- image/owner/prefix/intervention identity;
- native versus forced-prefix provenance;
- legacy-12 versus image2299 strata;
- intervention-minus-own-baseline results;
- trust-gate outcomes and stop-rule disposition;
- enough provenance to reproduce every claim-bearing row.

The experiment directory may link to the external `results.md` after execution.
Do not put large tensors or rollout artifacts into Git.

## 21. Advisory Fable review

A read-only `claude-fable-5/xhigh` review agreed with the shared harness:

- P1 static field causality;
- P2 dynamic history causality;
- P3 crossover and short free-rollout stability;
- static K/V owns existence, separability, and payload;
- dynamic state owns coverage, order/path management, and STOP;
- salience is the coupling surface.

Its strongest counterargument was that frozen affordance is not the same as
trainability, so a cheap density pilot might reach a useful observation faster.
That objection is retained as the alternative route in Section 9. The review
still favored this discriminator because a scalar pilot only tests one assumed
treatment and prior readable representations did not guarantee behavioral
consumption.

This review is advisory evidence, not authority and not an execution approval.

## 22. Deferred and superseded routes

Do not accidentally reopen these during this unit:

- **Special-token minimization:** deferred. Keep the native full wrapper.
- **A new query token at row start:** not part of this probe.
- **A standalone decode-time controller:** not the first experiment; frozen
  causal responsibility must be mapped first.
- **Multi-sample union as the decoder:** diagnostic evidence only.
- **Leave-one-out/segment-only teacher forcing:** cannot own self-rollout
  alignment without natural-prefix and multi-row transfer.
- **Fixed block-23 scalar density as a foregone conclusion:** demoted to one
  outcome-contingent training route.
- **Broad layer/head/temperature search:** prohibited by the stop rules.
- **Exact K-only/V-only engineering:** deferred until existing actuators make
  the decomposition decision-bearing.
- **Matched A2 production run:** conditional on an A3-only causal effect.
- **val200 or production-scale expansion:** outside this unit.

Reopen one only when a qualified result makes it the shortest discriminator or
the user explicitly changes the route.

## 23. Fresh-agent continuation checklist

Before any write or run:

1. `cd /data/CoordExp/.worktrees/research-probes`.
2. Recheck `git status --short`, branch, HEAD, and any active processes.
3. Read the minimum path in Section 6 and reconcile live compass drift.
4. Verify all checkpoint, config, panel, and script paths in Sections 11 and 19.
5. Confirm the user's current authorization boundary.
6. If authorized only to draft: create the proposed `unit.md`, freeze the
   cohort/contrast/endpoints/stop rules, validate links, and stop.
7. If authorized to execute: inspect live GPU occupancy and avoid interference;
   run the smallest trust-gate cells before the full cohort.
8. Reuse sealed/native artifacts; fill only missing matched cells.
9. Stop at the first invalid actuator or instrumentation gate.
10. Report the outcome through the decision table, including negative results
    and the strongest surviving alternative.

## 24. Completion condition

The proposed unit is complete only when it can answer, with natural-prefix
free-row evidence:

1. whether post-LLM image states contain a causally consumable owner substrate;
2. whether completed-row history suppresses/reallocates owner mass or is merely
   a readable bookmark;
3. whether their interaction produces stable, positive unique-owner utility
   over multiple rows with calibrated STOP;
4. which specific training objective family follows, or why none should be
   promoted.

Anything less is a diagnostic partial result and should be recorded as such.
