## Context

The current Stage-2 training/runtime spine is:

- training entrypoint and config materialization in `src/sft.py` and `src/config/loader.py`,
- Stage-2 AB dispatch through `custom.trainer_variant: stage2_two_channel`,
- Stage-2 Channel-B prep and loss assembly in `src/trainers/stage2_two_channel.py`,
- rollout parsing and Hungarian utilities under `src/trainers/rollout_matching/`,
- objective execution through the teacher-forcing pipeline under `src/trainers/teacher_forcing/`,
- typed Stage-2 config under `src/config/schema.py`,
- YAML-first experiment profiles under `configs/stage2_two_channel/`.

Today Channel-B still follows the older raw-prefix contract in practice:

1. decode a raw rollout,
2. recover parsed predicted objects from the rollout text,
3. bbox-filter those parsed objects,
4. run matching on that raw filtered list,
5. teacher-force the raw rollout prefix plus FN append.

That is not sufficient for the near-duplication failure mode because later correct objects inherit duplicate-heavy positive prefixes.

This change intentionally does **not** preserve the old raw-prefix Channel-B path. The goal is to replace it with one canonical clean-prefix contract.

## Goals / Non-Goals

**Goals**

- Make Channel-B matching and positive supervision operate on a clean accepted object sequence.
- Keep generic unmatched clean accepted objects neutral by default.
- Add targeted negative signal only for duplicate-certified continuations.
- Preserve existing Stage-2 positive geo/coord loss flow where possible.
- Keep the first implementation config-first, explicit, and easy to audit.
- Make the docs under `docs/` part of the implementation done-definition for this contract change.

**Non-Goals**

- No RL, GRPO, reward shaping, or repulsive set priors.
- No extra objectness classifier or DETR-style detection head.
- No generic penalty for unmatched clean objects.
- No whole-span negative CE for duplicate bursts.
- No backward-compatibility mode, config alias, or legacy contract toggle for the old raw-prefix Channel-B path.

## Decisions

### 1) Stage-2 Channel-B now has one canonical contract

This change defines a single canonical Stage-2 Channel-B contract:

`raw rollout -> bounded container salvage + strict record acceptance -> bbox-valid filtering -> sequential dedup -> accepted_objects_clean + duplicate_bursts -> Hungarian on clean -> clean-prefix CE + duplicate_ul`

Normative consequences:

- The old raw-prefix / immutable-prefix Channel-B contract is removed rather than preserved beside the new path.
- Later correct objects MUST be teacher-forced on the clean accepted prefix, not the raw duplicate-contaminated prefix.
- Duplicate objects MUST be removed from the positive teacher-forced sequence and represented separately as duplicate bursts attached to clean boundaries.
- Generic unmatched clean accepted extras MAY remain in the clean prefix as neutral context.

### 2) The config hierarchy stays simple and strict

`stage2_ab` remains the canonical typed namespace for Stage-2 two-channel training.

Normative config structure for this change:

- `stage2_ab.pipeline` remains required and explicit.
- The canonical Stage-2 AB objective ordering for this contract is:
  1. `token_ce`
  2. `duplicate_ul`
  3. `bbox_geo`
  4. `coord_reg`
- `duplicate_ul` MUST be declared with `channels: [B]`.
- `duplicate_ul` module `weight` is the only v1 scaling surface for duplicate UL.
- `duplicate_ul.config` is an empty mapping in v1. There is no secondary inner weight, enable knob, or alternate duplicate-target mode.
- `stage2_ab.channel_b` is restricted to:
  - `duplicate_iou_threshold`
  - `producer_wait_timeout_s`
  - `ddp_phase_timeout_s`

Removed by design:

- no `sequential_dedup_enabled` toggle,
- no `duplicate_ul_weight` flat knob,
- no raw-prefix compatibility mode,
- no `drop_invalid_struct_ce_multiplier` legacy path or alias.

The intended result is that pipeline structure determines the loss surface, while `stage2_ab.channel_b` only contains Channel-B runtime/data-contract knobs that are still semantically necessary.

### 3) “Strict parse” is defined as bounded container salvage plus strict record acceptance

The older wording “strict parse” is too ambiguous for the current code path, so this change makes the parsing contract explicit.

Normative parsing contract:

- The parser MAY use bounded salvage to locate the top-level `{"objects": [...]}` container and the last closed record boundary inside the rollout text.
- Record acceptance is strict:
  - only fully closed records inside the recovered container are eligible,
  - only bbox records are eligible in v1,
  - bbox arity and coord-token alignment must be valid,
  - invalid, ambiguous, non-bbox, or schema-violating records are dropped deterministically.
- No repaired record becomes a positive training object.
- Invalid container-level rollouts fall back to the canonical empty prefix `{"objects": [` so FN append can still proceed deterministically.

This preserves robust rollout consumption while avoiding any ambiguity about repaired records entering matching or supervision.

### 4) Channel-B produces explicit clean-prefix data products

Channel-B prep MUST materialize the following per-sample intermediates in order:

1. `parsed_bbox_objects_raw`
   - rollout-order list after bounded container salvage, strict record acceptance, and bbox-valid filtering.
2. `accepted_objects_clean`
   - deduplicated bbox objects in rollout order.
3. `duplicate_bursts_by_boundary`
   - duplicate objects grouped by clean boundary.
4. `match_on_clean`
   - Hungarian result between `accepted_objects_clean` and GT.
5. `clean_teacher_forced_prefix`
   - canonical assistant serialization of `accepted_objects_clean`.
6. `clean_teacher_forced_target`
   - `clean_teacher_forced_prefix + FN append + closure/EOS`.
7. `duplicate_ul_targets`
   - per-boundary duplicate-ul supervision derived from clean vs duplicate continuations.

The spec is intentionally explicit here because this is the highest-risk semantic change in the refactor.

### 5) Sequential dedup is bbox-only and happens before Hungarian

Sequential dedup is the chosen canonical ordering because it is what enforces clean-prefix teacher forcing.

Normative sequence:

1. traverse `parsed_bbox_objects_raw` in rollout order,
2. compare each candidate only against previously accepted clean bbox objects,
3. mark the candidate duplicate iff:
   - `normalize_desc(candidate.desc) == normalize_desc(accepted.desc)`, and
   - `IoU(candidate.bbox, accepted.bbox) >= duplicate_iou_threshold`,
4. otherwise accept the candidate into `accepted_objects_clean`.

v1 defaults:

- bbox-only dedup,
- shared `normalize_desc`,
- exact equality on normalized desc,
- `duplicate_iou_threshold = 0.90`.

### 6) Clean boundaries are explicit and include the pre-first case

Clean boundaries are indexed by insertion position in the clean sequence:

- boundary `0`: before the first clean accepted object,
- boundary `k` for `0 < k < N`: between `accepted_objects_clean[k-1]` and `accepted_objects_clean[k]`,
- boundary `N`: after the last clean accepted object and before closure/EOS,
  where `N = len(accepted_objects_clean)`.

Normative consequences:

- If `accepted_objects_clean` is empty, there is still exactly one valid boundary: `0`.
- Duplicates observed before the first accepted clean object attach to boundary `0`.
- Duplicates observed after the last accepted clean object attach to boundary `N`.

This removes the earlier ambiguity around pre-first duplicates and empty-clean-prefix cases.

### 7) Matching, FN detection, and positive supervision run on the clean accepted sequence

Hungarian matching for Channel-B MUST operate on `accepted_objects_clean`, not on the raw duplicate-heavy parsed list.

Normative consequences:

- duplicates do not enter the matched candidate set,
- GT matching and FN detection are computed against the clean accepted sequence,
- matched geometry/coord losses continue to use the existing positive path semantics,
- generic unmatched clean accepted extras remain present in the clean prefix but stay neutral.

Operational neutrality contract for unmatched clean extras:

- they MAY remain in the clean prefix as context,
- they MUST NOT populate `prefix_struct_pos`,
- they MUST NOT populate `prefix_coord_pos` or `prefix_coord_target_bins`,
- they MUST NOT populate `bbox_groups_prefix`,
- they MUST NOT create duplicate-ul positive targets.

That is the precise way the “neutral extras” philosophy is preserved under the current masking architecture.

### 8) The clean teacher-forced target is canonically reserialized

The positive teacher-forced target is no longer sourced from raw rollout prefix token ids.

Normative target construction:

- `clean_teacher_forced_prefix` is derived by canonical assistant serialization of `accepted_objects_clean` under the configured `custom.object_field_order`.
- `clean_teacher_forced_target` is built as:
  - canonical clean prefix,
  - canonical FN append inside the same `objects[]` container,
  - normal closure / EOS supervision.

Raw rollout token spans remain useful for parsing/diagnostics, but they are not the source of truth for the positive teacher-forced prefix under the new contract.

### 9) Duplicate UL is boundary-local, LCP-defined, and canonically serialized

`duplicate_ul` is a new explicit objective module. It is not folded into `token_ce`.

For each clean boundary `b`:

- `clean_continuation(b)` is the canonical token sequence from `clean_teacher_forced_target` starting at boundary `b`.
- For each duplicate attached to boundary `b`, `duplicate_continuation(b, dup)` is:
  - canonical serialization of that duplicate object at boundary `b`,
  - followed by the same canonical clean suffix that would follow boundary `b`.
- The UL target token is the first true divergence token of `duplicate_continuation(b, dup)` relative to `clean_continuation(b)`.
- If two duplicates at the same boundary yield the same divergence token id, they collapse to one UL term for that boundary.
- If no safe divergence token exists, the continuation is skipped and counted in diagnostics.

This is an intentional semantic refinement relative to earlier draft math that summed one UL term per duplicate object. The canonical v1 contract collapses same-boundary duplicates to one UL term per unique divergence token so burst length does not amplify the objective when multiple duplicates represent the same bad continuation choice.

Important consequence:

- The target is not “always the first desc token.”
- Same-class-next-object cases stay safe because the LCP can consume shared prefix tokens before divergence is chosen.

### 10) Metrics must expose duplicate collapse and use aggregation-safe names

The change adds one new objective atom:

- `loss/B_rollout_text/duplicate_ul`

The change also adds duplicate-collapse diagnostics and counters.

Normative gauges:

- `dup/max_desc_count`
- `dup/saturation_rate`

Normative count-like metrics:

- `dup/near_iou90_pairs_same_desc_count`
- `dup/near_iou90_pairs_any_desc_count`
- `stage2_ab/channel_b/dup/N_raw_bbox_valid`
- `stage2_ab/channel_b/dup/N_clean_accepted`
- `stage2_ab/channel_b/dup/N_duplicates`
- `stage2_ab/channel_b/dup/N_duplicate_bursts`
- `stage2_ab/channel_b/dup/N_ul_boundaries`
- `stage2_ab/channel_b/dup/N_ul_skipped_no_divergence`

The count-like names intentionally use `/N_` or `_count` so optimizer-step metric aggregation sums them rather than averaging them.

### 11) Docs sync is part of the contract

This change affects repo source-of-truth docs, not just code.

Minimum docs expected to change once implementation lands:

- `docs/training/STAGE2_RUNBOOK.md`
- `docs/training/METRICS.md`

Conditionally update when their documented surfaces change during implementation:

- `docs/ARTIFACTS.md`
- `docs/eval/README.md`
- `docs/README.md`

## Risks / Trade-offs

- **Canonically reserializing the clean prefix is a deliberate break from the old token-immutable rollout-prefix rule.**
  - Mitigation: make it the only supported Channel-B contract and define token provenance explicitly.
- **Sequential dedup could over-collapse valid same-class instances if the similarity rule is too loose.**
  - Mitigation: keep v1 rule strict (`normalize_desc` equality + high IoU) and bbox-only.
- **Duplicate UL needs precise boundary/token semantics.**
  - Mitigation: boundary indexing, canonical continuation construction, and LCP divergence are all defined above.
- **New metrics can aggregate incorrectly if named like gauges.**
  - Mitigation: use `/N_` and `_count` naming for additive counts, and keep gauges/rates clearly separate.
