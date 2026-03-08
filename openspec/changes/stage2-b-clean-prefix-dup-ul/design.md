## Context

The current Stage-2 training/runtime spine is:

- training entrypoint and config materialization in `src/sft.py` and `src/config/loader.py`,
- Stage-2 AB dispatch through `custom.trainer_variant: stage2_two_channel`,
- Stage-2 Channel-B prep and loss assembly in `src/trainers/stage2_two_channel.py`,
- strict rollout parsing and Hungarian utilities under `src/trainers/rollout_matching/`,
- objective execution through the teacher-forcing pipeline and registry surfaces under `src/trainers/teacher_forcing/`,
- typed Stage-2 config under `src/config/schema.py`,
- YAML-first experiment profiles under `configs/stage2_two_channel/`.

The existing Channel-B implementation is built around the older rollout-prefix contract:

1. decode a raw rollout,
2. strict-parse it into predicted objects,
3. run matching on parsed predictions,
4. build a rollout-context teacher-forced target from the raw parsed prefix plus FN append,
5. apply rollout-context masking and the configured objective modules.

This is close to the original FP-neutral philosophy, but it is not enough for the near-duplication failure mode because later correct objects inherit duplicate-heavy positive prefixes.

The frozen v2 contract for this change is:

- Channel-B v2 formally supersedes the older immutable-rollout-prefix rule.
- Dedup happens after strict parse plus bbox-valid filtering, and before Hungarian.
- Dedup is bbox-only in v1.
- Dedup uses the shared `normalize_desc` plus a configurable IoU threshold.
- The canonical Channel-B v2 path is:
  - `raw rollout -> strict parse -> bbox-valid filtering -> sequential dedup -> clean accepted sequence + duplicate bursts -> Hungarian on clean accepted -> clean-prefix CE + duplicate UL`
- Duplicate UL is a new explicit objective atom, not a `token_ce` extension.
- Duplicate UL is aggregated as one unit-weight UL term per unique divergence token per clean boundary.
- The target token is the first true LCP-divergence token relative to the clean continuation, not a hard-coded first desc token.
- Recommended v2 configs stay A-hot / B-cold.

## Goals / Non-Goals

**Goals**

- Make Channel-B matching and positive supervision operate on a clean accepted object sequence.
- Keep generic unmatched accepted objects neutral by default.
- Add targeted negative signal only for duplicate-certified continuations.
- Preserve existing Stage-2 teacher-forcing machinery and positive geo/coord loss flow where possible.
- Keep the first implementation minimal, readable, and config-first.
- Add diagnostics that make duplicate collapse auditable in training and eval.

**Non-Goals**

- No RL, GRPO, mAP reward shaping, or repulsive set priors.
- No separate objectness classifier or DETR-style detection head.
- No generic penalty for unmatched clean objects.
- No whole-span negative CE for duplicate bursts.
- No large refactor of Channel-A or Stage-1 behavior.

## Decisions

### 1) Clean-prefix reconstruction is the core contract change

Channel-B v2 is allowed and intended to rebuild a deduplicated clean assistant target. This is the key semantic change, because the failure mode is not only about which predicted objects enter Hungarian; it is also about which prefix later positives are trained under.

Normative consequence:

- Later correct objects MUST be teacher-forced on the clean accepted prefix, not on the raw duplicate-contaminated prefix.
- Generic accepted unmatched extras MAY remain in the clean accepted context.
- Duplicate objects MUST be removed from the positive teacher-forced sequence and represented separately as duplicate bursts attached to clean boundaries.

### 2) Sequential dedup happens before Hungarian because that is the chosen v2 contract

Sequential dedup is not being introduced as the only mathematically valid ordering. It is the chosen v2 contract because it enforces clean-prefix supervision semantics.

Normative sequence:

1. strict-parse the raw rollout,
2. drop invalid/non-bbox records using the existing bbox-valid filtering rules,
3. traverse bbox objects in rollout order,
4. compare each candidate only against previously accepted bbox objects,
5. mark it duplicate when:
   - `normalize_desc(candidate.desc) == normalize_desc(accepted.desc)`, and
   - `IoU(candidate.bbox, accepted.bbox) >= tau_dup`,
6. otherwise accept it into the clean sequence.

v1 defaults:

- bbox-only dedup,
- exact equality on shared normalized descriptions,
- `tau_dup = 0.90` configurable.

### 3) Clean accepted sequence and duplicate bursts are separate data products

Channel-B prep needs two outputs from the raw parsed rollout:

- `accepted_objects_clean`: the deduplicated accepted bbox sequence in rollout order,
- duplicate bursts attached to clean boundaries.

Boundary attachment contract:

- A duplicate burst is the sequence of duplicate bbox objects observed between two consecutive clean accepted objects, or between the last clean accepted object and EOS.
- Duplicate bursts are indexed by the clean boundary after accepted object `k`, with support for the pre-first boundary if needed by future work.
- Duplicate metadata must retain enough information to reconstruct the duplicate continuation tokens from that clean boundary.

This separation lets the positive side use the clean sequence while the negative side uses duplicate-certified continuations from the same clean boundary.

### 4) Hungarian, FN detection, and matched geometry use the clean accepted sequence

Hungarian matching for Channel-B MUST operate on the clean accepted bbox sequence, not the raw duplicate-heavy parsed objects.

Normative consequences:

- duplicates do not enter the matched candidate set,
- GT matching and FN detection are computed against the clean accepted sequence,
- matched geometry/coord losses continue to use the existing positive path semantics,
- generic unmatched clean accepted objects remain FP-neutral by default.

This preserves the current “do not punish ambiguous extras” philosophy while removing duplicate-certified objects from the positive candidate set.

### 5) Duplicate UL is boundary-local and LCP-defined

For each clean boundary:

- define the clean continuation from that boundary using the clean teacher-forced target,
- define each duplicate continuation using the duplicate object tokens from that same boundary,
- compute the longest common prefix between the clean continuation and the duplicate continuation,
- identify the first true divergence token in the duplicate continuation after that LCP,
- apply UL only to that divergence token.

v1 aggregation:

- collect the divergence tokens produced by duplicates at the same clean boundary,
- deduplicate them,
- apply one UL term per unique divergence token with unit weight,
- if no safe divergence token exists, skip UL for that continuation and count it in diagnostics rather than failing training.

This is the minimal safe version for same-class-next-object cases because it does not blindly suppress the first desc token.

### 6) Duplicate UL is a new explicit objective atom

The current Stage-2 objective registry/pipeline only knows `token_ce`, `bbox_geo`, and `coord_reg`. The duplicate penalty should be introduced as a new explicit objective atom/module surface rather than folded into `token_ce`.

Why:

- `token_ce` currently owns rollout-context positive masking semantics.
- duplicate UL is boundary-local negative supervision with its own metrics and skip behavior.
- keeping it explicit preserves auditability in pipeline identity, trainer metrics, and config.

Expected integration surfaces:

- Stage-2 Channel-B trainer prep in `src/trainers/stage2_two_channel.py`,
- objective registry / pipeline plumbing under `src/trainers/teacher_forcing/`,
- typed config in `src/config/schema.py`,
- trainer wiring in `src/sft.py`.

### 7) Channel-A should remain stable

This change is deliberately Channel-B-scoped. Channel-A should not change except where small shared pipeline/config plumbing updates are required to host the new Channel-B objective atom or metrics.

Recommended config posture:

- keep new v2 experiments A-hot / B-cold by default,
- reduce permissive rollout budgets in the recommended v2 profile,
- keep old configs loadable where possible, but add a clearly named v2 profile for the new contract.

### 8) Metrics must prove duplicate collapse without hiding regressions

The change should add duplicate-specific diagnostics without weakening existing parse/match/localization monitors.

Minimum diagnostics:

- `dup/max_desc_count`
- `dup/near_iou90_pairs_same_desc`
- `dup/near_iou90_pairs_any_desc`
- `dup/saturation_rate`
- counts for raw parsed objects, clean accepted objects, duplicate objects, duplicate bursts,
- counts for boundaries with UL applied and duplicate continuations skipped due to no safe divergence token.

Success should be auditable as:

- duplicate metrics drop,
- prediction count per sample stops blowing up,
- matched localization remains stable,
- parse truncation does not worsen materially.

## Risks / Trade-offs

- **Clean-prefix reconstruction changes an older invariant.**
  - Mitigation: scope the contract change explicitly to Channel-B v2, document it in the change artifacts, and keep Channel-A untouched.
- **Duplicate detection could over-collapse valid same-class instances if the similarity rule is too loose.**
  - Mitigation: keep v1 rule strict (`normalize_desc` equality + high IoU) and bbox-only.
- **Duplicate UL needs careful token-boundary handling.**
  - Mitigation: define it via first true LCP divergence, dedupe targets per boundary, and skip safely when no divergence exists.
- **Pipeline plumbing could sprawl if the new atom is wired too generically too early.**
  - Mitigation: keep the first implementation local and minimal, with explicit well-named helpers in Channel-B prep and a small explicit registry addition.
