## Overview

The new objective is a shared teacher-forcing framework for autoregressive
grounding/detection. It trains next-token predictions under teacher-forced
roll-in prefixes while respecting the latent order of object sets.

The contract intentionally unifies hard SFT, pure valid-set marginal, and
coverage-regularized valid-set marginal under one target IR and module system.
It does not preserve `ET_RMP_CE` or `recursive_detection_ce` as active training
terminology.

## Goals

- Use one reusable target IR for Stage-1 and Stage-2 teacher-forcing.
- Separate token-type stability from within-type or within-role valid-token
  likelihood.
- Remove false-negative gradients at ambiguous object-entry and coordinate
  branch points.
- Support repeated same-description objects whose first branch may be a
  coordinate token.
- Support shared description-prefix cases such as COCO `car` / `carrot` and
  future open-vocabulary prefix conflicts.
- Keep hard path commitment after a sampled branch becomes singleton.
- Keep standard hard SFT available as a clean comparator profile.
- Preserve historical checkpoint scoring without allowing old training APIs to
  keep running.

## Non-Goals

- No full recursive autoregressive subtree DP in the first implementation.
  The objective is sampled-path next-token valid-set marginal over teacher-forced
  roll-in prefixes.
- No Gaussian/Gibbs/IoU/CIoU coordinate soft CE as core coordinate ambiguity
  handling.
- No object subsampling for overlength samples in v1.
- No long-lived compatibility wrapper for old recursive-detection training.
- No required decode-time grammar mask for primary evaluation.

## Architecture

Canonical objective execution should extend the existing semantic runner under
`src/training/objectives/` and `src/training/supervision/`. Shared
teacher-forcing contracts, vocab roles, validation helpers, probability math,
and metrics should live under `src/training/teacher_forcing/`, with the concrete
objective module registered from `src/training/objectives/teacher_forcing.py`.
Detection-specific compact-full target construction should live under
`src/detection/teacher_forcing/`. Trainer code should integrate with the shared
runner, strip sidecars before model forward, validate logits/positions, and log
metrics, but should not own objective math or create a parallel objective stack.

The core sidecar key is `teacher_forcing_target_ir`. The core IR is
`TeacherForcingTargetIR`, containing `SupervisionAtom` records. Each atom is one
causal next-token supervision event after roll-in selected the teacher prefix.
Implementations should define one reusable `TEACHER_FORCING_TARGET_IR_KEY` and
wire it through sidecar filtering, collation, model-input bundling, trainer
bridges, and objective runners. The sidecar must be stripped before upstream
model forward while legitimate attention and flash-attention kwargs remain
forwarded.

Important atom fields:

- `batch_index`
- `logit_position`
- `target_position`
- `allowed_token_roles`
- `selected_token_role`
- `valid_token_ids`
- `selected_token_id`
- `latent_valid_token_ids`
- `coverage_target_weights`
- `loss_tags`
- optional `coord_role`
- diagnostic branch/provenance metadata

The v1 causal invariant is:

```text
target_position = logit_position + 1
logits[batch_index, logit_position] predicts input_ids[batch_index, target_position]
```

Loss modules must not shift labels internally.

The target token position is the canonical sequence position. `logit_position`
is retained in the IR as an explicit redundant validation field; objective
execution should still route row lookup through the existing label-row mapping
abstraction so there is one owner for target-position to logit-row alignment.

The runner must validate full unsliced rank-3 logits before loss computation:

```text
logits.ndim == 3
logits.shape[:2] == input_ids.shape[:2]
```

`logits_to_keep`, packed, or padding-free paths are unsupported in v1 unless a
later implementation explicitly defines an exact atom-position mapping. When an
attention mask is present, both the causal logit position and target token
position must be live. Rank-2 logits remain invalid even for a single-sample
batch because `SupervisionAtom.batch_index` is explicit.

The objective marginal scope is:

```text
sampled_path_next_token
```

This means valid sets are computed only for the current teacher-forced roll-in
prefix. V1 does not compute or claim full recursive subtree marginal likelihood
over alternate hidden-state prefixes.

## Token Roles

Roles are:

- `SCHEMA`
- `TEXT`
- `COORD`
- `STOP`

Most atoms have a singleton `allowed_token_roles` set. Mixed-role atoms are
supported for shared trie-prefix cases. Stage-1 marker-delimited `compact_full`
v1 may emit only singleton role sets plus `{TEXT, SCHEMA}` for
description-boundary ambiguity.

Example:

```text
car    -> <|box_start|>
carrot -> rot
```

After `<|object_ref_start|>car`, the valid set is:

```text
{<|box_start|>, rot}
```

The atom uses:

```text
allowed_token_roles = {TEXT, SCHEMA}
```

## Objective Modules

The active core modules are:

- `token_type_mass`
- `conditional_valid_set_likelihood`
- `within_valid_coverage`
- `continuation_margin` as optional/default-off training module with required
  diagnostics

Training uses full-vocabulary logits and full-vocabulary softmax. For each atom:

```text
P_allowed = sum p(v) for v in union(role_vocab(role) for role in allowed_token_roles)
L_type = -log P_allowed
```

Inner modules use probabilities conditional on the allowed role union:

```text
p_bar(v) = p(v) / P_allowed
L_valid = -log sum p_bar(v) for v in valid_token_ids
```

Coverage, if enabled, normalizes inside the valid set:

```text
p_valid(v) = p_bar(v) / sum_{u in valid_token_ids} p_bar(u)
L_coverage = CE(coverage_target, p_valid)
```

Coverage weights are raw, nonnegative residual-object branch masses, aggregated
by next token id. They are normalized by the coverage module.

## Profiles

Profiles are semantic bundles, not hyperparameter encodings:

- `hard_sft`
- `pure_valid_set_marginal`
- `hybrid_valid_set_marginal`

`coverage_strength` lives only under
`objective.modules.within_valid_coverage.coverage_strength`. Production and
ablation configs using the hybrid profile must set it explicitly.

Old names such as `ET_RMP_CE`, `et_rmp_like`, `support_balance`, `alpha1`,
`prefix_rollin_et_rmp_ce`, and `typed_trie_alpha0p1_random_rollin` are not
active profile names.

The old ET-RMP comparator semantics are represented structurally as
`hybrid_valid_set_marginal` with
`objective.modules.within_valid_coverage.coverage_strength: 1.0`.

## Compact-Full Serialization

The template id remains `compact_full`. The new training serialization policy is
`marker_delimited`:

```text
<|object_ref_start|>desc<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|object_ref_start|>...
```

No row-separator newline is part of new training. Production configs must
explicitly set:

```yaml
detection_template:
  id: compact_full
  serialization_policy: marker_delimited
```

This must be implemented as a policy-aware renderer/parser migration rather
than a global mutation of historical `compact_full`. Legacy newline-delimited
rendering/parsing is allowed only behind explicit historical inference/eval
policy such as `legacy_compatible`; resolved configs and artifacts should record
the serialization policy separately from the template id.

Description tokenization is context-aware. Canonical description token ids are
extracted from tokenizing:

```text
<|object_ref_start|>{desc}<|box_start|>
```

Marker boundaries and text-span to token-span mapping must be exact and unique.
Mapping failures are hard errors.

## Roll-In

The authored roll-in policy lives at
`objective.target_ir.rollin_policy.name`. The default roll-in policy is
`random_permutation` with base seed `17`.
Training roll-in varies deterministically by epoch and sample:

```text
hash64(17, epoch, sample_stable_id, rollin_policy_name, rollin_policy_version)
```

Routine teacher-forced validation uses fixed eval roll-in, such as seed `17` and
`eval_rollin_epoch: 0`. Permutation sensitivity analysis may evaluate several
fixed roll-in epochs/seeds.

## Stage-2 Packing

Stage-2 v1 has an explicit packing decision: do not run the new teacher-forcing
objective on packed or padding-free Stage-2 forwards until exact atom-position
mapping exists. The implementation roadmap may choose one of two paths:

```text
v1 fail-fast:
    reject objective.id=teacher_forcing with Stage-2 training.packing=true

future mapped path:
    define segment-local to packed-position mapping using packed segment
    metadata and flash-attention kwargs
```

The first implementation should prefer the fail-fast path unless the roadmap
explicitly budgets the packed-position mapping and tests it.

## Length and Empty-Object Handling

V1 rejects overlength samples before target IR construction. It does not
subsample objects to fit max length.

For Stage-1 COCO v1, samples with missing detection lists and explicit empty
object lists are dropped with counters. STOP-only empty-object samples are
reserved for future intentionally constructed negative-image datasets.

## Inference and Parser Compatibility

Primary new-model evaluation should use raw/minimally constrained decoding.
Compact grammar decoding is optional and explicitly labeled.

Parser mode is separate from grammar policy:

- `marker_delimited_strict`: new default for new teacher-forcing outputs.
- `legacy_compatible`: historical old-checkpoint scoring only.

Strict parsing is all-or-nothing for primary AP/AR. Diagnostic salvage may exist
but must not feed headline metrics unless explicitly labeled.

Strict compact-full parse errors use the stable taxonomy:

```text
empty_output
legacy_separator_in_new_format
missing_object_ref_start
missing_box_start
empty_description
forbidden_description_token
wrong_coord_arity
invalid_coord_token
trailing_garbage
invalid_geometry
```

## Migration

Active training configs using `recursive_detection_ce`, `prefix_rollin_et_rmp_ce`,
ET-RMP support/balance aliases, old recursive sidecars, and old trainer mixins
must fail with migration guidance. Historical inference/eval configs for
already-trained checkpoints may remain in a legacy namespace.

## Open Questions Deferred To Implementation Plan

- Exact file deletion order and import migration sequence.
- Exact unit-test grouping and smoke command matrix.
- Whether fixed eval/probe target IR caching can be implemented immediately or
  must be disabled until cache keys are extended.
