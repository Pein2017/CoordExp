## Context

Today the active `stage2_two_channel` duplicate path is:

`rollout tokens -> bounded parse/salvage -> bbox-valid filtering -> sequential dedup -> Hungarian match -> triage -> clean-prefix edit -> duplicate first-divergence target build -> one teacher-forced forward -> duplicate unlikelihood`

That architecture is too narrow for the observed pathology.

The current Stage-2 path decides duplicates too early using a sequential
same-description plus high-IoU heuristic, while the offline evaluator relies on
temporary post-op duplicate stripping to estimate how much metric loss comes
from catastrophic re-enumeration. The system therefore lacks one canonical,
shared duplicate-control contract.

The empirical case for a redesign is already strong:

- duplicate-collapse appears even under greedy decode,
- the failure looks more like a narrow local attractor than a universal global
  preference for duplicates,
- and the offline duplicate-like relation in
  [small_object_duplication_study.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/analysis/small_object_duplication_study.py)
  is materially stronger than the current sequential heuristic.

The chosen outcome for this change is explicit:

- prioritize reducing catastrophic duplicate-collapse,
- accept some recall loss,
- keep malformed-output handling separate,
- and make the duplicate-control policy canonical across training and offline
  evaluation.

## Goals / Non-Goals

**Goals**

- Replace legacy sequential duplicate suppression with one canonical
  duplicate-control policy for `stage2_two_channel`.
- Apply duplicate-control before GT matching, but only after anchor and
  explorer views are both available.
- Keep `loss_duplicate_burst_unlikelihood` as the canonical B-only suppression
  consumer and preserve the one-forward Channel-B contract.
- Remove non-exempt duplicate non-survivors from the positive prefix and turn
  them into UL targets.
- Apply the same duplicate-control policy in offline evaluation post-op so raw
  and guarded metrics are both first-class outputs.
- Normalize duplicate-control metrics, artifact names, and typed knobs rather
  than preserving sequential-duplicate legacy naming.

**Non-Goals**

- Redesign malformed parse / invalid-rollout handling. That remains owned by
  parse/salvage and strict-drop policy.
- Add a second teacher-forced forward or replace
  `loss_duplicate_burst_unlikelihood` with a new objective module.
- Preserve backward compatibility for the legacy sequential duplicate path,
  legacy metric aliases, or legacy config names.
- Expand the first landing to `stage2_rollout_aligned`.
- Depend on post-hoc confidence scoring during training.

## Core Thesis

The best bet is not “invent a bigger anti-dup loss.” The best bet is:

1. promote duplicate detection into a shared repo-owned policy layer,
2. suppress duplicate-collapse before GT matching,
3. keep the current UL consumer stable,
4. and use the same policy in offline evaluation as a guard.

This matches the empirical evidence better than a loss-first redesign:

- the model already shows partial latent signal against duplicates in many clean
  prefix states,
- the major failure is local collapse in specific autoregressive states,
- and a post-op duplicate guard already improves benchmark precision on the
  hardest scenes.

## Decisions

### 1. Introduce a shared duplicate-control core

This change will promote the duplicate-like relation from analysis code into a
repo-owned runtime helper, expected to live under
[src/common/](/data/home/xiaoyan/AIteam/data/CoordExp/src/common).

The shared core owns:

- duplicate feature extraction from bbox rollout objects,
- duplicate-like pair testing,
- connected-component building,
- conservative crowd-safety exemption checks,
- deterministic survivor selection,
- normalized duplicate-control metrics,
- and a policy decision record that can be consumed by both training and
  offline evaluation.

Why:

- training and offline post-op should not maintain separate duplicate logic,
- analysis code already contains the strongest evidence-backed relation,
- and shared ownership reduces drift between “what training suppresses” and
  “what evaluation guards.”

### 2. Duplicate-control runs before GT matching, after anchor plus explorer evidence is assembled

The chosen control point is:

`parse anchor + explorers -> build duplicate-control evidence -> suppress/exempt -> GT matching`

This honors the request to suppress the core symptom early while still allowing
explorer support to serve as a conservative exemption signal.

Important implication:

- duplicate-control decisions cannot rely on GT-backed status, because GT
  matching has not happened yet.
- GT-backed information may still be logged later for diagnostics, but it is
  not available to the suppression policy itself.

### 3. Anchor objects are the suppression surface; explorer objects provide exemption evidence

The positive prefix remains anchor-rooted.

The shared policy therefore treats:

- anchor rollout objects as the objects that may survive or be suppressed,
- explorer rollout objects as evidence used to decide whether a same-description
  local cluster looks like real crowding or duplicate collapse.

Explorer-only objects do not become positive prefix additions in this redesign.
They are used only to support conservative exemptions.

### 4. The canonical duplicate-like relation follows the proven analysis policy

The default relation is taken from
[small_object_duplication_study.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/analysis/small_object_duplication_study.py):

- same normalized description, and
- either IoU above threshold, or center-distance within a radius scaled by
  object size.

The runtime refactor ports that relation into repo-owned shared code; it does
not import analysis scripts at runtime.

### 5. The typed config surface stays minimal

This change intentionally exposes only the minimum typed surface needed to make
duplicate-control reproducible.

Proposed Stage-2 training surface:

- `stage2_ab.channel_b.duplicate_control.iou_threshold`
- `stage2_ab.channel_b.duplicate_control.center_radius_scale`

Proposed offline evaluation surface:

- `eval.duplicate_control.enabled`

Everything else remains fixed policy and must be captured in code, docs, and
resolved artifacts rather than becoming a large authored heuristic tree.

### 6. Conservative crowd-safety exemptions are mandatory

The chosen exemption posture is conservative.

A cluster should be exempt only when duplicate-control has strong reason to
believe it is not a local collapse pattern, for example:

- the cluster is spatially spread rather than tight,
- or explorer views provide consistent support for multiple same-description
  instances.

This matches the stated objective: suppress catastrophic duplicate-collapse
first, and only preserve ambiguous crowded scenes when the evidence is strong
enough.

### 7. Deterministic survivor selection uses only pre-match evidence

Because control happens before GT matching, survivor choice must use only
evidence available at that seam.

The first landing will therefore use this deterministic lexicographic order:

1. higher explorer support count/rate,
2. not border-saturated,
3. earlier anchor emission order.

This is an intentional adaptation of the earlier general recommendation. GT
backing was part of the broader heuristic, but it is unavailable at the chosen
pre-match seam and is therefore excluded from the canonical survivor rule.

### 8. Non-exempt non-survivors always disappear and become UL

This change does not keep a neutral duplicate bucket.

For any suppressible duplicate cluster:

- exactly one survivor remains on the positive prefix,
- every non-survivor disappears from the positive prefix,
- and every non-survivor contributes duplicate UL metadata.

Crowd-safety exemptions are the only path that keeps multiple same-description
objects out of suppression. If a cluster is exempt, those objects are not
considered non-survivors under the suppression policy.

### 9. The UL consumer shape stays unchanged

The runtime producer changes, not the UL consumer contract.

The canonical `loss_duplicate_burst_unlikelihood` module continues to consume:

- `(boundary, rel_pos, token_id)` targets,
- from one teacher-forced forward,
- on Channel-B only.

UL multiplicity also stays intentionally collapsed by `(boundary, token_id)` in
the first landing. This preserves the current payload shape and deterministic
ordering behavior.

### 10. Offline evaluation reports both raw and guarded outputs

Offline evaluation becomes a first-class client of the shared duplicate-control
policy.

The evaluator should:

- score raw predictions exactly as emitted,
- apply the canonical duplicate guard offline,
- score guarded predictions separately,
- and persist both result families.

The recommended artifact contract is:

- raw artifact remains the canonical input JSONL,
- guarded artifact is emitted as `gt_vs_pred_guarded.jsonl`,
- raw metrics remain in `metrics.json`,
- guarded metrics are emitted in `metrics_guarded.json`,
- and a dedicated duplicate-control report records how many predictions were
  suppressed and why.

This keeps the raw research/debug view visible while making guarded deploy/safety
metrics explicit.

## Proposed Architecture

### Shared Runtime Layer

Expected new core module:

- [src/common/duplicate_control.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/common/duplicate_control.py)

Expected responsibilities:

- `DuplicateControlObject`: normalized runtime object view
- `DuplicateControlCluster`: connected component over duplicate-like edges
- `DuplicateControlDecision`: per-object keep/suppress decision record
- `build_duplicate_clusters(...)`
- `apply_duplicate_policy(...)`
- `compute_duplicate_metrics(...)`

Expected upstream helpers reused:

- normalized description helpers from
  [semantic_desc.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/common/semantic_desc.py)
- canonical object schema helpers from
  [schemas.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/common/schemas.py)

### Training Flow

The new Stage-2 flow becomes:

`rollout tokens`
`-> bounded parse/salvage`
`-> bbox-valid filtering`
`-> build anchor/explorer duplicate-control evidence`
`-> duplicate-control policy on anchor objects`
`-> GT matching on surviving anchor objects`
`-> clean-prefix edit`
`-> UL target projection from suppressed non-survivors`
`-> one teacher-forced forward`

Key ownership changes:

- [rollout_views.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/rollout_views.py)
  stops making final sequential dedup decisions.
- [target_builder.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py)
  consumes explicit duplicate-control decisions instead of
  `duplicate_bursts_by_boundary` from a sequential heuristic.
- [stage2_two_channel.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel.py)
  logs normalized duplicate-control counters and gauges instead of legacy
  sequential-burst accounting.

### Offline Evaluation Flow

The evaluator path becomes:

`raw gt_vs_pred.jsonl`
`-> detection-evaluator raw score`
`-> shared duplicate-control offline guard`
`-> gt_vs_pred_guarded.jsonl`
`-> detection-evaluator guarded score`

The same policy relation is reused, but the offline path has no explorer views.
It therefore operates with the evidence available in artifact records only and
uses a strictly documented “offline conservative” subset of the shared policy.

Malformed-output handling remains separate:

- parse / invalid geometry behavior stays where it already lives,
- duplicate-control only operates on valid bbox objects.

## Metrics And Artifact Contract

This redesign intentionally normalizes duplicate-control naming.

### Training Metrics

Recommended gauge family:

- `dup/raw/max_desc_count`
- `dup/raw/saturation_rate`
- `dup/raw/duplicate_like_max_cluster_size`
- `dup/raw/desc_entropy`

Recommended additive counter family:

- `dup/raw/near_iou90_pairs_same_desc_count`
- `dup/raw/near_iou90_pairs_any_desc_count`
- `stage2_ab/channel_b/dup/N_clusters_total`
- `stage2_ab/channel_b/dup/N_clusters_exempt`
- `stage2_ab/channel_b/dup/N_clusters_suppressed`
- `stage2_ab/channel_b/dup/N_objects_suppressed`
- `stage2_ab/channel_b/dup/N_ul_boundaries`
- `stage2_ab/channel_b/dup/N_duplicate_burst_unlikelihood_skipped_no_divergence`

The key point is that “raw pathology” gauges and “policy action” counters are
separate. The old sequential duplicate-burst naming does not survive unless the
metric still literally means the same thing.

### Offline Evaluation Artifacts

Recommended outputs:

- `gt_vs_pred.jsonl`
- `gt_vs_pred_guarded.jsonl`
- `gt_vs_pred_scored.jsonl` for score-aware COCO/AP evaluation
- `gt_vs_pred_scored_guarded.jsonl` for guarded score-aware COCO/AP evaluation
- `metrics.json`
- `metrics_guarded.json`
- `duplicate_guard_report.json`
- deterministic guarded companions such as:
  - `per_image_guarded.json`
  - `per_class_guarded.csv`
  - `matches_guarded.jsonl`
  - `matches@{iou_thr}_guarded.jsonl` when threshold-specific match artifacts
    are emitted

### Reporting Rule

- Raw metrics remain the primary research/debug headline.
- Guarded metrics become the explicit post-op safety/deploy report.
- Both are required outputs for this change.

## Risks And Trade-offs

- [Risk] Conservative suppression may still remove some real crowded objects.
  Mitigation: keep exemptions conservative but explicit, and verify against
  dense-scene fixtures plus guarded-vs-raw eval deltas.
- [Risk] The analysis-derived relation may still be too weak for some collapse
  modes. Mitigation: compare new raw duplicate gauges and suppression counters
  against the current baseline on the same smoke runs.
- [Risk] Normalizing metric and config naming increases refactor cost.
  Mitigation: do the naming reset in the same patch, rather than carrying a
  hybrid legacy/new contract.
- [Risk] Offline guarded metrics could be misread as model-only metrics.
  Mitigation: keep raw and guarded outputs separate and document the difference
  in evaluator artifacts.

## Verification Strategy

Required deterministic fixture coverage:

- non-sequential same-description chain clusters,
- tight local duplicate-collapse clusters,
- spatially spread same-description crowded scenes,
- explorer-supported exemption cases,
- collapsed UL multiplicity for repeated same-boundary token divergence.

Required runtime checks:

- Stage-2 smoke on `stage2_two_channel`,
- duplicate-control metrics reduce correctly under gradient accumulation,
- raw and guarded evaluator artifacts are both produced,
- raw and guarded eval metrics differ in the expected direction on hard cases.

## Handoff Notes

Important evidence and context to preserve:

- offline duplicate-strip experiment:
  - [dup_postop_eval.py](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval.py)
  - [comparison.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/comparison.json)
  - [dedup_summary.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/dedup_summary.json)
- duplication diagnostics:
  - [stage2_near_duplication_2026-03-05.md](/data/home/xiaoyan/AIteam/data/CoordExp/progress/diagnostics/stage2_near_duplication_2026-03-05.md)
  - [small_object_duplication_offline_findings_2026-03-26.md](/data/home/xiaoyan/AIteam/data/CoordExp/progress/diagnostics/small_object_duplication_offline_findings_2026-03-26.md)

Assumption carried into this design:

- because duplicate-control runs before GT matching, the canonical survivor rule
  cannot use GT-backed status and therefore uses pre-match evidence only.
