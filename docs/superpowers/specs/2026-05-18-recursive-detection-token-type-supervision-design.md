# Recursive Detection Orthogonal Token-Type Supervision Design

Status: draft superpowers design; implementation in progress on the feature worktree.

Date: 2026-05-18

Owner: CoordExp training/objective

Target worktree: `codex/instance-trie-gaussian-softce`

Primary files:

```text
src/detection/token_types.py
src/detection/objective.py
src/detection/loss.py
src/detection/runtime.py
src/detection/dataset.py
src/config/schema.py
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1.yaml
tests/test_compact_type_gate.py
tests/test_recursive_detection_ce_target_builder.py
tests/test_recursive_detection_ce_loss_adapter.py
tests/test_latest_training_config_contract.py
```

## Purpose

The current instance-trie Gaussian SoftCE branch intentionally separates
coordinate smoothing from recursive entry-choice supervision, but the actual
loss roles are still too entangled. Coordinate SoftCE currently uses
full-vocabulary normalization, so it also acts as a coordinate-token gate.
Trie multi-positive targets can also appear on coordinate and structural
boundary tokens when serialized objects share prefixes. This makes it hard to
interpret failures: the same loss row may be doing structure validity,
description ambiguity, coordinate exclusivity, and coordinate smoothing at once.

This design makes the supervision roles orthogonal:

```text
structure supervision       -> hard CE on schema / boundary tokens
description ambiguity       -> hard CE plus trie support/balance CE on free-text desc tokens
token-type supervision      -> bidirectional valid-type mass gate on schema, desc, coord, eos
coordinate smoothness       -> SoftCE over the 1000 coordinate-token vocabulary only
```

The goal is not to invent a stronger objective. The goal is to make each signal
mean one thing, so later ablations can tell whether failures come from
structure, description identity, coordinate exclusivity, coordinate smoothing,
or boundary/stop calibration.

## Non-Goals

- No inference, parser, duplicate-guard, or evaluator changes.
- No OpenSpec change until this objective becomes a stable default or benchmark
  contract.
- No relaxation of strict compact parsing.
- No rollout/RL objective.
- No new production dependency.
- No broad rewrite of recursive detection target construction.
- No attempt to salvage the historical IoU/CIoU-Gibbs negative-result configs.

## Current State

The latest compact recursive detection stack already has the pieces we need:

- `TokenRole` spans identify `CONTROL`, `DESC`, `BBOX_START`, `COORD`,
  `SEPARATOR`, and `TERMINAL` positions.
- `src/detection/token_types.py` builds compact token groups:
  `struct`, `coord`, `eos`, `excluded_control`, and `desc`.
- `objective.type_gate` schema already exists for `prefix_rollin_et_rmp_ce`.
- `TokenTarget` already carries `type_gate_token_ids` and
  `type_gate_weight`.
- `src/detection/loss.py` already adds an allowed-mass type gate to every
  position that carries those fields.

The gap is wiring and role separation:

- type gates are currently attached only for prefix-rollin examples;
- `random_permutation_et_rmp_ce` rejects objectized `objective.type_gate`;
- trie multi-positive branch targets can be attached to coordinate and
  structural boundary positions;
- coordinate SoftCE for `instance_trie_gaussian` uses full-vocabulary SoftCE,
  which duplicates the coordinate gate role.

## Target Supervision Contract

### 1. Structural And Boundary Tokens

Structural/schema positions use hard CE only for the teacher token plus the
global token-type gate for the structural token group.

This includes compact-full tokens such as:

```text
<|object_ref_start|>
<|box_start|>
separator/newline tokens
terminal <|im_end|>
```

Important tradeoff: if one object description is a prefix of another, for
example `car <|box_start|>` versus `cart ...`, the `<|box_start|>` position is
still structural and remains hard CE for the teacher path. We do not treat the
description-continuation token `t` as a structural-position positive. This is
intentional because trie CE is now scoped to free-text description positions.

### 2. Description Tokens

Description text positions use:

```text
hard CE on the teacher desc token
+ trie support/balance CE over valid desc child tokens when the desc trie branches
+ desc token-type gate
```

The hard CE term preserves single-path commitment. The trie term keeps
remaining-object order ambiguity visible inside free text. The type gate
suppresses coordinate tokens, schema tokens, EOS, and excluded special tokens
from entering description positions.

Non-branching description positions use hard CE plus the desc type gate.

### 3. Coordinate Tokens

Coordinate positions use:

```text
coord token-type gate
+ coordinate SoftCE over exactly the 1000 <|coord_*|> token rows when enabled
```

The coordinate type gate is responsible for "this position must be a coordinate
token." Coordinate SoftCE is responsible only for shaping probability inside the
coordinate vocabulary.

For `instance_trie_gaussian`, coordinate SoftCE should therefore use a
coordinate-vocabulary softmax:

```text
p_coord(k | prefix) =
  exp(logit(<|coord_k|>)) / sum_{j=0}^{999} exp(logit(<|coord_j|>))

L_coord_soft_ce =
  - sum_k q(k) log p_coord(k | prefix)
```

The target distribution `q(k)` remains the active-branch instance Gaussian
mixture already defined by `coord_soft_targets.py`.

### 4. Token-Type Gate

The type gate is a full-vocabulary mass gate:

```text
L_type = -log sum_{token in allowed_type(position)} p_full(token | prefix)
```

It is "bidirectional" in the practical CE sense: increasing valid subset mass
necessarily suppresses mass on invalid groups under the same full-vocabulary
softmax.

Allowed subsets are:

| Position kind | Allowed subset |
|---|---|
| schema/structure/control | `struct` |
| description text | `desc` |
| coordinate token | `coord` |
| terminal stop | `eos` |

The `excluded_control` group is never a positive allowed subset. It exists so
description positions cannot absorb arbitrary special/control tokens.

## Data Flow

Target preparation should attach type-gate metadata before loss computation:

```text
prepare_detection_training_example(...)
-> build_recursive_detection_targets(...)
-> _apply_compact_type_gate(...)
-> RecursiveDetectionTargets(token_targets=...)
-> compute_recursive_detection_ce_batch_loss(...)
```

For prefix-rollin, the existing type-gate attachment path should be generalized
rather than duplicated.

For random-permutation ET-RMP, runtime dataset construction should pass
`objective.type_gate` into full-sequence example preparation when enabled.

## Configuration Surface

Allow `objective.type_gate` for:

```text
random_permutation_et_rmp_ce
prefix_rollin_et_rmp_ce
```

Keep rejecting `objective.type_gate` for:

```text
sorted_sft
random_order_sft
trie_disabled_full_suffix_ce
```

Recommended first config for instance-trie Gaussian SoftCE:

```yaml
objective:
  type_gate:
    enabled: true
    mode: allowed_type_mass
    weights:
      struct: 1.0
      desc: 1.0
      coord: 1.0
      eos: 0.5
```

The weights are deliberately conservative and symmetric across struct/desc/coord
for the first smoke. They should be treated as ablation knobs, not stable
defaults.

## Metrics

Reuse the existing aggregate metrics where possible:

```text
recursive_detection_ce/type_gate_loss
recursive_detection_ce/type_gate_allowed_mass
recursive_detection_ce/type_gate_allowed_tokens
recursive_detection_ce/type_gate_weight
```

Add role-scoped summaries only if the aggregate metrics cannot answer whether
schema, desc, coord, and eos gates are behaving differently.

Coordinate SoftCE diagnostics should continue to report:

```text
recursive_detection_ce/coord_soft_ce/target_entropy
recursive_detection_ce/coord_soft_ce/target_peak_prob
recursive_detection_ce/coord_soft_ce/effective_support_size
recursive_detection_ce/coord_soft_ce/effective_candidate_count
```

Add one diagnostic to disambiguate coordinate normalization:

```text
recursive_detection_ce/coord_soft_ce/softmax_scope = coord_vocab
```

Because metric payloads are numeric today, this may be emitted as:

```text
recursive_detection_ce/coord_soft_ce/is_coord_vocab_softmax = 1.0
```

## Testing Strategy

Use TDD. The first failing tests should prove the intended separation:

1. A shared-prefix `car` versus `cart` sample keeps `<|box_start|>` hard CE and
   does not make it a trie multi-positive target.
2. Description branch positions add hard CE plus trie CE.
3. Type gates attach to full-sequence `random_permutation_et_rmp_ce` examples,
   not only prefix-rollin examples.
4. Coordinate SoftCE is invariant to non-coordinate logits when the coordinate
   target distribution is unchanged.
5. The type gate still responds to non-coordinate leakage at coordinate
   positions.
6. Config parsing accepts `objective.type_gate` for
   `random_permutation_et_rmp_ce` and rejects it for SFT variants.

## Risks

- Removing trie positives from structural boundary positions may reduce
  order-ambiguity coverage for prefix-description cases. This is the intended
  price of role separation and should be measured.
- Coordinate-vocab SoftCE plus a separate coordinate gate changes the gradient
  scale relative to full-vocab SoftCE. The first smoke must compare loss terms
  and coordinate exact-token metrics before production launch.
- Type-gate group construction depends on tokenizer vocabulary completeness.
  Tests should cover lightweight tokenizers and real coord-token ranges.

## Acceptance Criteria

- Structural boundary tokens are hard CE targets in recursive detection target
  metadata.
- Free-text description trie branches compute hard CE plus trie support/balance
  CE.
- Type gates are available and attached for random-permutation ET-RMP configs.
- Coordinate SoftCE for `instance_trie_gaussian` uses only the 1000 coordinate
  logits for its smooth target normalization.
- Non-coordinate leakage at coord positions is penalized only by the token-type
  gate, not by coordinate SoftCE.
- Focused tests pass under `conda run -n ms python -m pytest`.
