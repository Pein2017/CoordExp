## Context

The current Stage-2 training/runtime spine is:

- training entrypoint and config materialization in `src/sft.py` and `src/config/loader.py`,
- typed Stage-2 config in `src/config/schema.py`,
- Stage-2 AB dispatch through `custom.trainer_variant: stage2_two_channel`,
- Channel-B rollout prep and target construction in `src/trainers/stage2_two_channel.py`,
- rollout parsing/matching helpers under `src/trainers/rollout_matching/`,
- objective execution through `src/trainers/teacher_forcing/`,
- YAML-first experiment control under `configs/stage2_two_channel/`.

Today Channel-B is already clean-prefix and duplicate-aware, but it is still fundamentally a **single-rollout** correction scheme.
That means:

- the anchor trajectory and the correction evidence are the same object,
- generic unmatched clean extras stay neutral,
- duplicate-certified continuations are the only local negatives,
- and recoverable GT objects discovered only under alternate decoding are invisible to the training target unless they are already ordinary FN under the same rollout.

The new v3 direction is to keep the v2 clean-prefix training shape while letting a second rollout answer a sharper question:

> for each object hypothesis near the anchor trajectory, is it GT-backed, stable-enough-to-shield, or dead?

## Goals / Non-Goals

**Goals**

- Keep v3 as a minimal perturbation of v2 clean-prefix Channel-B.
- Use exactly two rollout views in v1:
  - anchor = greedy
  - explorer = stochastic, default `T=0.7`
- Preserve the existing Channel-B Hungarian + gating contract for what counts as GT-backed.
- Build the final positive target by **editing the anchor clean sequence**, not by rebuilding a union trajectory.
- Keep one merged teacher-forced forward and avoid introducing a second training branch in v1.
- Make recoverability-aware GT weighting the main new positive signal.
- Keep shielded unlabeled-consistent objects neutral.
- Restrict local negative supervision to anchor-side dead continuations.
- Keep the change config-first and auditable through metrics, dumps, docs, and OpenSpec deltas.

**Non-Goals**

- No recovered-prefix distillation in v1.
- No explorer-prefix teacher forcing in v1.
- No promotion of explorer-only non-GT-backed objects to positives or shielded context in v1.
- No semantic-nearness classifier or likelihood-based posterior as the primary v1 triage rule.
- No whole-sequence RL / GRPO / reward-model training.
- No DETR-style head, objectness classifier, or separate detector branch.
- No blanket FP penalty over unmatched objects.

## Decisions

### 1) Channel-B keeps the v2 skeleton, but now runs it twice before target construction

The canonical v1 Stage-2 Channel-B flow becomes:

`anchor rollout + explorer rollout`
`-> per-run bounded salvage + strict record acceptance`
`-> per-run bbox-valid filtering`
`-> per-run sequential dedup`
`-> per-run Hungarian on accepted clean objects`
`-> cross-rollout pairing / triage`
`-> anchor-edited clean target`
`-> one merged teacher-forced forward`
`-> weighted FN positives + dead-anchor first-divergence UL`

This is intentionally not a new training architecture.
It is the current clean-prefix Channel-B contract with a stronger evidence-gathering stage before the final target is built.

### 2) K=2 does not require a general clustering engine in v1

The research note talks about cross-rollout clusters, but the first implementation can stay simpler.

For `K=2`, the canonical runtime object is:

- a high-IoU anchor/explorer pair, or
- an unpaired anchor singleton, or
- an unpaired explorer singleton.

So the first implementation may represent “clusters” as pair-or-singleton triage records rather than a general K-way clustering data structure.

The association itself must still be normative:

- compute candidate anchor/explorer pairs by IoU,
- keep only pairs with `IoU >= consistent_iou_threshold`,
- choose a **one-to-one bipartite max-IoU matching**,
- if multiple assignments achieve the same maximum total IoU, choose the one whose sorted pair list `[(anchor_index, explorer_index), ...]` is lexicographically smallest.

This keeps the trainer changes smaller and makes debugging easier:

- each accepted anchor object can inspect at most one paired explorer object,
- explorer-only objects are explicit singleton cases,
- triage reduces to pair classification plus singleton fallback.

### 3) The final target is anchor-edited, never rebuilt from a union order

The positive target source of truth remains the anchor clean sequence.

Normative target-construction rule:

- start from `accepted_objects_clean` from the anchor rollout,
- preserve anchor order for all retained objects,
- delete anchor objects classified as `dead_anchor`,
- keep anchor objects classified as `anchor_gt_backed`,
- keep anchor objects classified as `shielded_anchor` as neutral context,
- append GT FN objects, marking `recovered_fn` objects with higher per-object desc+geo+coord weight.

Rejected alternative:

- rebuilding a canonical union over anchor + explorer.

Reason for rejection:

- it silently changes ordering, prefix semantics, masking, and downstream auditability all at once,
- and it promotes the explorer trajectory from miner to teacher.

### 4) GT-backed inherits the existing Channel-B matching contract, but training actions stay side-specific

The v1 triage contract does not invent a stricter new notion of “true positive.”

GT-backed evidence means:

- matched under the existing Channel-B accepted-clean Hungarian + gate contract.

Recovered GT means:

- missed in the anchor accepted-clean match,
- hit in the explorer accepted-clean match.

This keeps v3 directly comparable to v2 and avoids hidden metric drift.

Training actions projected from that evidence are:

- `anchor_gt_backed`
  - the anchor member matched GT and remains in the anchor prefix
- `recovered_fn`
  - the anchor member missed GT but the explorer member matched; the GT is injected through the FN tail with higher per-object weight
- `shielded_anchor`
  - stable non-GT-backed anchor object retained as neutral context
- `dead_anchor`
  - anchor object removed from the positive prefix and eligible for local UL
- `dead_explorer`
  - explorer-only or unretained explorer evidence used for diagnostics and triage, but not as a direct positive or separate suppression branch

### 5) Shielded means both-rollout stable, geometry-first, and anchor-resident

The canonical v1 shield rule is intentionally high precision:

- not GT-backed,
- present in both rollouts after per-run cleaning,
- high geometric agreement across the two rollouts,
- not an obvious redundant re-description of an already-kept anchor GT-backed object,
- and only retained if it already exists in the anchor clean sequence.

The broader posterior language in the research note remains valid as theory, but it is not the first runtime contract.

Practical implication:

- semantics may act as a weak veto later if needed,
- but the first acceptance rule is geometry-first,
- and explorer-only objects are not shielded by default.

### 6) Explorer-only non-GT-backed objects are dead by default in v1

If an object hypothesis:

- is not GT-backed,
- does not appear as a stable anchor/explorer pair,
- and exists only in the explorer rollout,

then v1 treats it as `DEAD` for training purposes.

Reason:

- single hot-rollout evidence is the noisiest part of the system,
- and v1 explicitly avoids turning the explorer into a second teacher path.

This keeps unlabeled-positive modeling conservative in the first implementation.

### 7) The positive/negative losses all come from one merged forward

Conceptually, v3 is:

`L(clean_anchor) + L(explore-derived corrections)`

Operationally, v1 realizes that as one teacher-forced pass over one edited target.

The forward carries:

- clean-prefix structure supervision,
- GT-backed / FN tail positive supervision,
- higher per-object desc+geo+coord weights for recovered FN objects,
- anchor-side dead continuation UL targets.

Rejected alternative:

- separate anchor and explorer teacher-forced passes.

Reason for rejection:

- more implementation risk,
- harder to audit,
- and not necessary to test the main scientific question first.

### 8) `duplicate_ul` stays the module slot, but its semantic source broadens

To minimize churn in the objective pipeline:

- keep `duplicate_ul` as the canonical B-only module name,
- keep `loss/B_rollout_text/duplicate_ul` as the logging key,
- keep first-divergence suppression as the token-level mechanism.

What changes is the **source of the targets**:

- v2: same-desc duplicate-certified continuations only
- v3 v1: any anchor-side continuation triaged as `DEAD`

This lets the trainer reuse existing objective-pipeline plumbing while still changing the actual training semantics.

### 9) New typed config should group the v3-specific knobs

To keep the change explicit and auditable, add a grouped typed config block under `stage2_ab.channel_b`:

- `stage2_ab.channel_b.v3_k2.explorer_temperature`
- `stage2_ab.channel_b.v3_k2.explorer_top_p`
- `stage2_ab.channel_b.v3_k2.explorer_top_k`
- `stage2_ab.channel_b.v3_k2.consistent_iou_threshold`
- `stage2_ab.channel_b.v3_k2.recovered_fn_weight`

Existing Channel-B knobs stay where they already live:

- `duplicate_iou_threshold`
- `producer_wait_timeout_s`
- `ddp_phase_timeout_s`

Anchor decode remains fixed to greedy / deterministic in v1, so no separate anchor sub-config is required yet.

### 10) Rollout infrastructure must support per-call decode overrides across HF and vLLM

The v3 contract requires two rollout policies inside the same Channel-B step:

- anchor: greedy / deterministic
- explorer: stochastic (default `T=0.7`)

So the rollout path must grow a per-call decode override seam that both rollout backends honor:

- HF rollout helpers must accept call-local decode params instead of only reading one global `rollout_matching.decoding` block,
- vLLM colocate rollout helpers must accept the same overrides rather than forcing greedy-only operation,
- vLLM server rollout helpers must accept the same overrides rather than forcing greedy-only operation,
- if a configured backend/runtime combination cannot honor the dual-policy contract, trainer initialization must fail fast.

### 11) Metrics must expose triage outcomes separately from duplicate diagnostics

Legacy duplicate metrics remain valuable, but v3 needs explicit triage accounting.

Add count-like metrics:

- `stage2_ab/channel_b/triage/N_anchor_gt_backed`
- `stage2_ab/channel_b/triage/N_shielded_anchor`
- `stage2_ab/channel_b/triage/N_dead_anchor`
- `stage2_ab/channel_b/triage/N_dead_explorer`
- `stage2_ab/channel_b/triage/N_recovered_gt`
- `stage2_ab/channel_b/triage/recovered_gt_num`
- `stage2_ab/channel_b/triage/recovered_gt_den`
- `stage2_ab/channel_b/triage/dead_anchor_num`
- `stage2_ab/channel_b/triage/dead_anchor_den`

Keep existing duplicate counters/gauges as supporting diagnostics rather than removing them.

### 12) DDP/runtime symmetry remains mandatory

The new dual-rollout path must preserve rank-symmetric control flow on B steps:

- all ranks request both anchor and explorer rollouts for the same raw-sample set,
- all ranks build the same number of per-sample rollout views before packing,
- no rank-local “skip the explorer” or “fallback to anchor only” path is allowed once distributed control flow is committed.

This is a correctness and deadlock-avoidance constraint, not an optimization detail.

## Data Flow

### Per-sample runtime data products

For each raw training sample on a Channel-B step:

1. `rollout_anchor_raw`
2. `rollout_explorer_raw`
3. `parsed_bbox_objects_raw_anchor`
4. `parsed_bbox_objects_raw_explorer`
5. `accepted_objects_clean_anchor`
6. `accepted_objects_clean_explorer`
7. `match_anchor`
8. `match_explorer`
9. `triage_records`
10. `anchor_objects_kept`
11. `anchor_objects_dead`
12. `fn_objects`
13. `fn_objects_recovered`
14. `y_v3_target`
15. `duplicate_ul_targets` (now semantically dead-anchor continuation targets)

### Triage outcome rules

For each pair-or-singleton triage record:

- if the **anchor** side is matched under the existing contract:
  - `anchor_gt_backed`
- else if the anchor side misses and the **explorer** side matches:
  - `recovered_fn`
- else if both sides exist, neither is GT-backed, geometry agreement exceeds `consistent_iou_threshold`, and the pair does not conflict with an already-kept anchor GT-backed object:
  - `shielded_anchor`
- else if the anchor side exists:
  - `dead_anchor`
- else:
  - `dead_explorer`

Then convert triage into the final target:

- keep anchor `anchor_gt_backed`
- keep anchor `shielded_anchor`
- remove anchor `dead_anchor`
- append all FN GT objects
- apply higher per-object desc+geo+coord weight to `fn_objects_recovered`

## Risks / Trade-offs

- **Throughput risk:** B steps now decode twice.
  - Mitigation: reuse existing rollout backend/chunking, keep `K=2` fixed in v1, and add one shared decode-override seam instead of forking backend codepaths semantically.
- **False suppression risk in crowded scenes:** geometry-only pairing can still mislabel hard same-class scenes.
  - Mitigation: keep the shield rule high precision and apply suppression only anchor-side.
- **Weighted-loss plumbing risk:** recovered-FN desc+geo+coord weighting now requires new per-object metadata to flow through `token_ce`, `bbox_geo`, and `coord_reg`.
  - Mitigation: make the metadata contract explicit and add unit tests at each module seam.
- **Semantic drift risk:** broadening `duplicate_ul` semantics without renaming could confuse operators.
  - Mitigation: update docs/specs/metrics wording explicitly and keep the behavior change visible in monitor dumps.
- **Config drift risk:** v3-specific knobs could sprawl across `rollout_matching` and `stage2_ab`.
  - Mitigation: keep v3-specific knobs grouped under `stage2_ab.channel_b.v3_k2`.
- **Implementation-surface risk:** target construction, masks, metrics, and monitor dumps all change together.
  - Mitigation: land the work in small slices with explicit tests around target editing, weighting, and UL target provenance.
