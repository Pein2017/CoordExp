# Compact-Full Prefix Roll-in Multi-Positive Unification Design

Date: 2026-05-06
Updated: 2026-05-07

Status: implementation authorized and in progress. The 2026-05-07 hardening
addendum below is part of the implementation contract.

Owner surface: latest compact detection stack under `src/detection/*`.

Primary variant: `prefix_rollin_et_rmp_ce`.

## 2026-05-07 Hardening Addendum

The user approved the post-audit P1/P2 fixes on 2026-05-07. These constraints
are binding for the current implementation:

- `prefix_rollin_et_rmp_ce` remains `compact_full` only; no compatibility path
  is required for legacy set-continuation configs or old `custom.*` objective
  knobs.
- Latest recursive CE sidecars must be validated against the collated batch
  before model forward. For every target, `input_ids[b, position]`,
  `labels[b, position]`, `attention_mask[b, position]`, and
  `attention_mask[b, position - 1]` must agree with the sidecar teacher token
  and causal predictor row.
- Recursive CE owns the token loss. `labels` must not be passed into the
  Qwen/HF forward path for this objective; labels may remain available locally
  for metrics.
- EOS trust weight scales only the main EOS CE term. It must not scale the
  independent type-gate loss, because type-gate is the schema/malformed-output
  guardrail.
- Target positions for full-sequence logits must satisfy
  `0 < position < sequence_length`. A target at the final sequence index is an
  off-by-one causal-label error.
- The global Qwen HF generation contract is:
  `eos_token_id = id("<|im_end|>")` and
  `pad_token_id = id("<|endoftext|>")`.
  This applies beyond the `prefix_rollin_et_rmp_ce` training variant. Training
  EOS targets still use only `<|im_end|>`.
- HF processor calls used by inference/rollout paths must set `do_resize=false`
  whenever the processor supports it, preserving the dataset-time geometry
  contract.
- vLLM local/server inference must stop on `"<|im_end|>"` only. Do not add
  `<|endoftext|>` or tokenizer-default EOS as a second stop token for
  compact-full decode. Local vLLM should pass multimodal processor
  `do_resize=false` when supported.
- `training.effective_batch_size` is the source of truth. When it is present,
  `training.gradient_accumulation_steps` is derived and must not be authored in
  YAML.
- Runtime artifacts must record both requested `effective_batch_size` and
  realized `actual_global_effective_batch_size`, plus the Qwen generation
  eos/pad/stop/geometry contract.
- Packing/padding-free future mode means `per_device_train_batch_size=1`; the
  effective batch then counts packed long-sequence units under the global max
  length. Non-packed padded mode may use `per_device_train_batch_size>1`, but
  the optimizer-step sample budget still comes from `effective_batch_size`.
- Compact-full trainable token rows are 1002 rows: 1000 coord rows plus
  `<|object_ref_start|>` and `<|box_start|>`. The persisted module name may
  remain `coord_offset_adapter`, but docs and manifests should describe the
  actual token-row scope.
- Raw coord-token GT boxes must be validated as non-inverted `xyxy`. Image
  paths must resolve inside `data.image_root`; missing or escaped image paths
  are hard failures.
- Zero-object / EOS-only examples are supported as a safety guard, even though
  the current training data is expected to contain detectable objects.
- Production EOS calibration is currently enforced at config shape level
  (`source: calibrated_formula_ref` plus a versioned artifact reference). The
  stronger content-level gate (`production_approved`, validation scope, probe
  refs, and formula class) remains an artifact/registry validation requirement
  unless and until a concrete validator is implemented.

## Purpose

Unify the current multi-positive detection fine-tuning idea into one strict, compact-only training surface for Qwen3-VL / Qwen-VL style decoder-only VLMs.

The target method remains ordinary autoregressive next-token training and ordinary autoregressive decode. It does not add a detection head, slot planner, RL objective, hidden-state cosine objective, or alternative generation mechanism.

The implementation direction is:

```text
compact_full rendered sequence
  + ground-truth prefix roll-in over emitted object subsets
  + entry-trie local multi-positive CE over remaining objects
  + balance-first support/balance loss
  + token-type schema gate
  + calibrated EOS supervision interface for incomplete annotations
```

The design intentionally removes old legacy surfaces from the new contract. Historical behavior can remain recoverable from git history, progress notes, and old artifacts, but the current implementation should delete active config/runtime/test exposure for superseded Stage-1 set-continuation code rather than preserving aliases, moved historical config directories, or backward-compatible runtime paths.

## Locked Decisions

| Area | Decision |
|---|---|
| Template scope | Support only `compact_full` for this new variant. |
| Serialized row shape | Use the existing compact row tokens: `<|object_ref_start|>`, free-text desc, `<|box_start|>`, four `<|coord_*|>` tokens. |
| Extra delimiter scope | No extra object-list delimiter is in scope; only the chat-template `<|im_end|>` assistant stop marker is supervised as EOS when eligible. |
| EOS token | `<|im_end|>` is the semantic EOS / assistant stop token. It must be resolved from the active Qwen/Qwen3-VL tokenizer and used consistently for template construction and supervision. |
| Roll-in depth | Sample `K` uniformly from every integer in `[0, N]`. |
| Prefix loss | Roll-in prefix tokens are context only and must have `labels=-100`. |
| Completion loss | Supervise the remaining suffix after the prefix. |
| K=N behavior | No remaining object suffix exists; only EOS `<|im_end|>` supervision is eligible, with calibrated EOS trust weight. |
| Local target | Use entry-trie multi-positive targets over remaining objects at every object-entry token where multiple legal continuations exist. |
| Object weighting | Use object-multiplicity-uniform weighting, not big/easy-object weighting. |
| Loss shape | Use local next-token CE with sparse targets; do not use segment-level ranking as the main objective. |
| Support/balance default | Prefer balance-first multi-positive learning: `support_weight=1.0`, `balance_weight=2.0` as the first ablation default. |
| Type discipline | Add an independent token-type gate for struct, coord, desc/text, and EOS positions. |
| EOS trust weight | Use `empirical_unlabeled_poisson_v0` as the first ablation EOS-trust prior: `unlabel_count=max(0,-0.35+0.43*GT_count)`, `eos_trust_weight=exp(-penalty_per_missing * unlabel_count / temperature)` with `penalty_per_missing=1.0` and `temperature=1.0` initially. This makes the log-penalty on EOS trust linear in the expected unlabeled count. Production claims still require an artifact-backed `calibrated_formula_ref`; the empirical prior is never a production source. |
| Decode | Keep the normal autoregressive decode path as the main path. Any grammar guard is diagnostic or upper-bound only. |
| Compatibility | No backward compatibility for legacy configs or old `custom.*` objective knobs in this new surface. |
| Cleanup order | Remove active legacy set-continuation exposure before adding the new `prefix_rollin_et_rmp_ce` schema/runtime/config. |

## Non-Goals

- Do not support `stage1_json_pretty`, `compact_no_desc`, `compact_no_bbox`, or other templates in this variant.
- Do not support old `custom.stage1_set_continuation` config names.
- Do not preserve legacy `branch_support_weight` / `branch_balance_weight` aliases.
- Do not introduce object slots, a detection decoder head, Hungarian decoding, or a separate planner.
- Do not change Qwen3-VL chat-template semantics or upstream HF model files.
- Do not make EOS trust weight a fixed dense-scene threshold heuristic.
- Do not construct training templates or training targets with `<|endoftext|>`, `<|end_of_text|>`, or any text-level tokenizer terminator other than `<|im_end|>`.
- Do not reserve probability mass for unknown unlabeled objects inside the trie in the first implementation.
- Do not make raw hidden-state cosine a primary alignment objective.

## Current Baseline To Consolidate

The repo currently has two related but non-identical surfaces:

| Surface | Existing role | New-contract treatment |
|---|---|---|
| `src/detection/*` latest compact recursive detection | Current latest compact stack with full-sequence random permutation and recursive sidecars. | Becomes the owner surface for the new strict variant. |
| `src/trainers/stage1_set_continuation/*` | Older physical prefix roll-in / ET-RMP-CE family. | Superseded by the new compact-only implementation; no compatibility promise. |

The new implementation should consolidate behavior under latest detection modules instead of teaching the old trainer more modes.

Important existing owners to preserve conceptually:

| Concept | Owner direction |
|---|---|
| Strict compact render/tokenization | `src/detection/template.py`, `src/detection/tokenization.py`, `src/detection/dataset.py` |
| Recursive target construction | `src/detection/objective.py` |
| Sparse recursive CE loss | `src/detection/loss.py` |
| Latest schema | `src/config/schema.py` |
| Runtime routing | `src/detection/runtime.py` as the latest-detection runtime owner; `src/sft.py` delegates and `src/training_runtime/plan.py` is touched only for legacy route deactivation. |
| Compact inference/parser compatibility | Existing inference/eval compact surfaces, without making them the objective owner. |

## Method Definition

For one image, let the labeled object set be:

```text
Y = {y_1, ..., y_N}
```

Each object serializes under `compact_full` into an entry token span:

```text
z_i = Tok(<|object_ref_start|> desc_i <|box_start|> coord_i1 coord_i2 coord_i3 coord_i4)
```

No extra object-list delimiter is part of this variant. The rendered assistant payload contains compact object rows only; it does not manually include `<|im_end|>`. The fully wrapped Qwen chat text contains exactly one assistant stop marker supplied by `apply_chat_template(..., add_generation_prompt=False)`.

### Tokenizer And Chat-Template Stop Contract

The compact roll-in variant uses the model's Qwen/Qwen3-VL chat-template stop marker, not a generic text tokenizer terminator.

Required training contract:

| Token text | Training-template role |
|---|---|
| `<|im_end|>` | The only semantic EOS / assistant stop token for `compact_full` training examples. |
| `<|endoftext|>` | Not used in training template construction or objective targets. May be recognized by parser/diagnostic cleanup only. |
| `<|end_of_text|>` | Not used in training template construction or objective targets. May be recognized by legacy cleanup only if the active tokenizer resolves it. |

Implementation rules:

- Resolve `im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")` and fail fast if it does not map to a valid single token id.
- Treat tokenizer unknown-token fallback as invalid: `<|im_end|>` must be present in the tokenizer vocabulary or special-token map, must not resolve to `unk_token_id`, and must encode as exactly `[im_end_token_id]` with `add_special_tokens=False`.
- Probe `apply_chat_template(..., add_generation_prompt=False)` with a minimal closed assistant turn and fail fast unless exactly one `<|im_end|>` id appears after the assistant payload span.
- If `tokenizer.eos_token` is present and equals `<|im_end|>`, it may be treated as the same id.
- If `tokenizer.eos_token` is present but is not `<|im_end|>`, do not silently switch the compact training target to that tokenizer EOS. The compact objective should still use `<|im_end|>` when it resolves to a valid token id, and should expose the mismatch as a diagnostic rather than as a target-construction rule.
- Do not append `<|endoftext|>` or `<|end_of_text|>` to rendered assistant training text.
- Do not include `<|endoftext|>` or `<|end_of_text|>` in EOS positive targets, type-gate EOS groups, or EOS diagnostics for this training objective.
- Parser and inference compatibility code may still strip or detect extra generated text-level terminators after `<|im_end|>`, but that compatibility behavior must remain outside the training target owner.
- `RenderedAssistantSequence.text` and assistant message content must exclude `<|im_end|>`; the Qwen chat template supplies the assistant stop marker when the assistant turn is closed with `add_generation_prompt=False`.
- EOS targets align to the tokenized assistant stop marker span, not to a manually appended payload token.
- A tokenizer whose `eos_token` is `<|endoftext|>` may be accepted only if `<|im_end|>` resolves to a valid single token id; the text-level EOS is diagnostic metadata and never a training target.

### Roll-in State Sampling

Sample a random permutation `pi` of all `N` labeled objects and a depth:

```text
K ~ Uniform({0, 1, ..., N})
```

Define:

```text
S = {pi_1, ..., pi_K}
R = Y \ S
```

Render the prefix entries in the sampled order:

```text
h = entry(pi_1) ... entry(pi_K)
```

Render the supervised suffix from remaining objects using the same sampled full permutation tail:

```text
teacher_suffix = pi[K:]
c_payload = entry(pi_{K+1}) ... entry(pi_N)
chat_template_stop = <|im_end|>
```

For v1, do not independently resample a second teacher continuation order from `R`. Independent suffix-order sampling is a future ablation and must not be used by the default `prefix_rollin_et_rmp_ce` contract.

The model input contains `h + c` inside the assistant response. Loss masking is:

| Span | Label behavior |
|---|---|
| system/user/image/chat prompt | `-100` |
| assistant header / generation prompt tokens | `-100` unless already outside labels by current tokenizer path |
| roll-in prefix `h` | `-100` |
| supervised suffix object entries | active labels and sparse multi-positive sidecars |
| chat-template `<|im_end|>` | active only with calibrated EOS weight when `R` is empty at that state |
| padding | `-100` |

This makes prefix sampling a state-distribution change, not a new decoder or architecture.

### Local Entry-Trie Target

At a token position inside the next object entry, let `u` be the partial object-entry token prefix already generated after the current roll-in state.

The active remaining candidates are:

```text
C(u) = {i in R : u is a prefix of z_i}
```

For a possible next token `a`:

```text
C(u + a) = {i in C(u) : z_i[|u|] = a}
```

With object-multiplicity-uniform weights:

```text
q_trie(a | S, u) = |C(u + a)| / |C(u)|
```

The local loss is sparse soft CE:

```text
L_soft = - sum_a q_trie(a | S, u) log p_theta(a | image, h, u)
```

When only one legal token exists, this degenerates to ordinary hard CE.

Hard-label safety invariant:

```text
At every target position that carries positive candidates and participates in sparse/soft CE,
the teacher hard label token must be present in the positive token set.
```

If that invariant fails, the batch should fail fast because tokenization, context, span alignment, or trie construction is wrong. Ordinary hard CE positions must either be represented as singleton positives `{teacher_token_id}` or be explicitly marked as soft-CE-exempt.

Normative trie sidecar contract:

- `positive_token_ids` is a sequence of unique vocabulary token ids.
- `q_by_token[a] = |{i in C(u): z_i[|u|] == a}| / |C(u)|`.
- Duplicate object instances contribute to the numerator of their shared token id, never by repeating that token id in `positive_token_ids`.
- `positive_size` counts unique next-token identities, while `candidate_object_count` tracks object multiplicity.
- Fail fast if `len(set(positive_token_ids)) != len(positive_token_ids)`.
- If all remaining candidates share the same next token, the position is singleton/hard-equivalent with `q=1` for that token even when multiple object instances remain.

### Balance-First Support/Balance Loss

For positive child tokens `V`, model probabilities `p`, and target distribution `q`:

```text
P_valid = sum_{a in V} p(a)
L_support = -log P_valid
L_balance = -sum_{a in V} q(a) log(p(a) / P_valid)
L = support_weight * L_support + balance_weight * L_balance
```

Facts:

| Weights | Meaning |
|---|---|
| `support_weight=1.0`, `balance_weight=1.0` | Equivalent to sparse soft CE over positive children. |
| `support_weight>1.0`, `balance_weight=1.0` | Extra pressure to put mass on the valid set. |
| `support_weight=1.0`, `balance_weight>1.0` | Extra pressure to avoid collapse inside the valid set. |

The first new ablation default should be:

```yaml
support_weight: 1.0
balance_weight: 2.0
```

Rationale: the core experiment is mode coverage over legal next objects. The model should not learn to route all gradient through one easy object or a canonical ordering. A nonzero support term remains necessary so the conditional balance term cannot be satisfied while the model assigns low absolute probability to all valid children.

Non-normative objective warning:

- `L_support = -log P_valid` is only the valid-set support subterm.
- `L_support` alone is a valid-mass/logsum objective and is not the primary mode-covering Prefix-Closed Multi-Target SFT objective.
- The normative target is sparse soft CE, or the balance-first support/balance decomposition with a meaningful positive balance term.
- The primary `prefix_rollin_et_rmp_ce` surface defaults to `support_weight=1.0`, `balance_weight=2.0`.
- Support-dominant variants must be explicitly labeled as support-dominant ablations and must not back primary E1/E4 claims.

### Token-Type Gate

The entry-trie target decides which legal token identities should be likely. The type gate separately teaches schema discipline.

Token groups for `compact_full`:

| Group | Tokens |
|---|---|
| `struct` | `<|object_ref_start|>`, `<|box_start|>` |
| `coord` | `<|coord_0|>` through `<|coord_999|>` |
| `desc` | free-text description tokens excluding compact `struct`, `coord`, `eos`, text-level terminators, padding, chat/image role sentinels, and tokenizer control/special tokens |
| `eos` | `<|im_end|>` |

At each supervised position, construct an allowed type set. The gate loss is:

```text
L_type = -log sum_{a in AllowedType(position)} p_theta(a | context)
```

Important branch rule:

```text
At a multi-positive trie position, AllowedType(position) must be the union of all positive child token types, not just the teacher token type.
```

This avoids punishing a legal branch only because the sampled teacher path picked another child type.

Compact token-type group contract:

- `struct`, `coord`, `eos`, and `excluded_control` are pairwise disjoint.
- `desc = vocab - struct - coord - eos - excluded_control`.
- `excluded_control` includes pad tokens, ignore-index-only special tokens, chat role/header sentinels, image placeholders, tokenizer-added control tokens, and text-level terminators such as `<|endoftext|>` / `<|end_of_text|>`.
- Every positive teacher token must belong to exactly one trainable type group.
- Fail fast if a compact structural token appears in `desc`, if `<|im_end|>` appears outside `eos`, or if a known chat/image/pad/control token appears in `desc`.

Total loss composition at each active supervised target position:

```text
L_main =
  support_weight * L_support + balance_weight * L_balance
  for trie multi-positive positions;
  hard_ce for singleton/control positions;
  eos_trust_weight * hard_ce(<|im_end|>) for eligible empty-remaining EOS positions.

If type_gate.enabled:
  L_total_position = L_main + type_gate.weights[position_type] * L_type
else:
  L_total_position = L_main
```

At multi-positive positions, `position_type` is the union of all positive child token types. The implementation must assert that every positive token is contained in the expanded allowed type set.

The type gate is training-time schema pressure. It improves parseability but does not mathematically guarantee zero malformed output under unconstrained sampling. A decode-time grammar mask may be used as a diagnostic upper bound, but it is not part of the main objective path.

### EOS Supervision

For this variant, there is no extra object-list delimiter. `<|im_end|>` is the only semantic stop token.

EOS policy:

| State | EOS behavior |
|---|---|
| Remaining labeled objects are nonempty | `<|im_end|>` is not a positive continuation. |
| Remaining labeled objects are empty | `<|im_end|>` receives a represented EOS target with CE weight `eos_trust_weight`; if `eos_trust_weight=0`, the target remains present with zero loss weight. |

`eos_trust_weight` controls the loss weight/trust of the represented positive `<|im_end|>` CE term. It does not decide whether the EOS target object exists, and it is not a hard stop label with fixed weight 1.

For v0, use the user-provided rough missing-label count formula as an empirical EOS prior:

```text
unlabel_count(N) = max(0, -0.35 + 0.43 * N)
eos_trust_weight_raw(N) = exp(-penalty_per_missing * unlabel_count(N) / temperature)
eos_trust_weight(N) = clamp(eos_trust_weight_raw(N), min_weight, max_weight)
```

where `N = GT_count` and the first ablation defaults are `penalty_per_missing = 1.0`, `temperature = 1.0`, `min_weight = 0.0`, and `max_weight = 1.0`.

This is a log-linear EOS trust design:

```text
-log(eos_trust_weight_raw) = penalty_per_missing * unlabel_count / temperature
```

So the user-provided linear `unlabel_Count` formula becomes a linear penalty in log-weight space. Directly making `eos_trust_weight` itself linear would either need an arbitrary clipping threshold or would become negative for dense images. The log-linear mapping keeps `eos_trust_weight` in `[0,1]`, makes small GT-count EOS highly trusted, and smoothly lowers EOS pressure as expected missing labels rise.

Example values for `temperature=1.0`:

| GT_count | unlabel_count | eos_trust_weight |
|---:|---:|---:|
| 0 | 0.00 | 1.000 |
| 1 | 0.08 | 0.923 |
| 2 | 0.51 | 0.600 |
| 3 | 0.94 | 0.391 |
| 5 | 1.80 | 0.165 |
| 10 | 3.95 | 0.019 |
| 20 | 8.25 | 0.00026 |

EOS loss contract:

```text
if remaining_labeled_objects != empty:
    <|im_end|> is not a positive token
    normal local CE/support-balance over legal next-object tokens competes against EOS

if remaining_labeled_objects == empty:
    L_eos = eos_trust_weight(GT_count) * CE(<|im_end|>)
    where CE(<|im_end|>) = -log p_theta(<|im_end|> | image, prefix)
```

EOS supervision applies whenever the supervised suffix has consumed all remaining labeled objects, including both initial `K=N` and `K<N` examples after the final remaining object is generated. If `eos_trust_weight=0`, the implementation should still represent the EOS target with zero weight so diagnostics remain observable and the loss path has a defined zero-weight contract.

`empirical_unlabeled_poisson_v0` is allowed only as an ablation/smoke EOS prior. A later production-calibrated formula must be stored as a versioned artifact with the exact full-val probe, non-collapse FP definition, formula parameters, and provenance, and production configs must reference it through `calibrated_formula_ref`.

## Config Shape

The new variant should use direct objectized latest-schema config sections, not `custom.*` piles.

Conceptual YAML:

```yaml
experiment:
  surface: ablation
  ablation_id: E1
  claim_scope: none

detection_template:
  id: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy

objective:
  id: recursive_detection_ce
  variant: prefix_rollin_et_rmp_ce
  state_weighting: uniform_permutation
  normalization: semantic_image_bucket_balanced

  rollin:
    enabled: true
    source: ground_truth
    prefix_loss: masked
    suffix_order: same_sampled_permutation
    k_distribution:
      type: uniform_inclusive
      min_k: 0
      max_k: object_count

  target:
    type: entry_trie_support_balance
    trie_scope: object_entry
    q_weighting: object_multiplicity_uniform
    singleton: hard_ce
    control_tokens: hard_ce
    support_weight: 1.0
    balance_weight: 2.0

  type_gate:
    enabled: true
    mode: allowed_type_mass
    weights:
      struct: 2.0
      coord: 1.0
      desc: 0.2
      eos: 0.5

  eos:
    eos_token: <|im_end|>
    tokenizer_contract:
      require_token: <|im_end|>
      reject_training_terminators:
        - <|endoftext|>
        - <|end_of_text|>
    policy: missing_label_prior_weighted_ce
    supervision:
      when_remaining_nonempty: not_positive
      when_remaining_empty: weighted_ce
    eos_trust_weight:
      source: empirical_unlabeled_poisson_v0
      expected_unlabeled_count:
        intercept: -0.35
        slope: 0.43
        floor: 0.0
      trust_mapping:
        type: log_linear_missing_count_penalty
        penalty_per_missing: 1.0
        temperature: 1.0
        min_weight: 0.0
        max_weight: 1.0
      allowed_surfaces:
        - ablation
        - smoke
      production_allowed: false
      allowed_ablation_sources:
        - empirical_unlabeled_poisson_v0
        - constant_ablation
        - disabled_ablation
```

The `struct`, `coord`, `desc`, and `eos` type groups are derived runtime groups,
not YAML-authored pass-through knobs. The YAML only authors the mode and per-type
weights. Runtime derives membership from the active tokenizer and compact
template:

- `struct`: compact structural rows such as `<|object_ref_start|>`,
  `<|box_start|>`, and row separators.
- `coord`: `<|coord_0|>` through `<|coord_999|>`.
- `eos`: `<|im_end|>` only.
- `desc`: non-control text tokens after excluding struct, coord, eos,
  tokenizer special control tokens, padding, image sentinels, and text-level
  terminators such as `<|endoftext|>` / `<|end_of_text|>`.

Strict schema requirements:

- Reject this variant unless `detection_template.id == compact_full`.
- Require `objective.state_weighting == uniform_permutation` and
  `objective.normalization == semantic_image_bucket_balanced` for this variant;
  runtime must not silently substitute these values while YAML says something
  else.
- Reject old `custom.stage1_set_continuation` knobs for this variant.
- Require `objective.rollin.suffix_order == same_sampled_permutation` for v1; independent suffix resampling is a separate future ablation.
- Allow `objective.target.support_weight` and `objective.target.balance_weight` as the canonical new paths, but keep obsolete-key scanning path-aware so it rejects `objective.support_weight`, `objective.balance_weight`, `objective.trie_support_weight`, `objective.trie_balance_weight`, `branch_support_weight`, `branch_balance_weight`, and every `custom.*` legacy path.
- Require `experiment.surface` for this variant, with enum values `smoke`, `ablation`, and `production`.
- Reject production configs unless `eos_trust_weight.source == calibrated_formula_ref` and the referenced calibration artifact is explicitly approved.
- Reject legacy/unknown EOS sources, including `deferred_user_formula`.
- Allow `empirical_unlabeled_poisson_v0`, `constant_ablation`, and `disabled_ablation` only in configs marked as `ablation` or `smoke`.
- Do not implement a boolean production escape hatch for empirical EOS priors. Production eligibility is determined by `experiment.surface` plus an approved `calibrated_formula_ref` artifact.
- Reject this variant at runtime if the active tokenizer cannot resolve `<|im_end|>` to one token id.
- Reject this variant at runtime if training template construction attempts to use `<|endoftext|>` or `<|end_of_text|>` as EOS.
- Keep packing/cache disabled until sidecar offset rewriting is separately designed.

## Implementation Owners

| Concept | Canonical owner |
|---|---|
| New config schema | `src/config/schema.py` |
| Roll-in sampled state | new `src/detection/rollin.py` or a focused section in `src/detection/objective.py` if kept small |
| Rendered compact sequence and spans | `src/detection/template.py`, `src/detection/tokenization.py` |
| Prefix/suffix label masking and sidecar alignment | `src/detection/dataset.py` |
| Entry-trie target construction | `src/detection/objective.py` |
| Support/balance and type-gate loss math | `src/detection/loss.py` |
| Runtime routing and failfast policy | `src/detection/runtime.py` owns latest-detection mode/runtime config; `src/sft.py` delegates; `src/training_runtime/plan.py` changes only for old `custom.trainer_variant` deactivation. |
| Trainer composition | `src/bootstrap/trainer_setup.py` if recursive CE mixin gating changes. |
| Metrics and diagnostics | existing trainer metric plumbing plus detection diagnostic helpers |
| Legacy removal | current active routing/docs/config indexes/tests that advertise old set-continuation as active, while preserving historical provenance only when clearly non-runnable for the new surface |

## Additional Binding Implementation Contracts

The following hardening contracts are part of the design and are not optional polish.

### Single Roll-in State

Each training example samples one `RollinState` exactly once. The same state must flow through:

```text
roll-in sampling
  -> compact payload rendering
  -> Qwen chat-template wrapping
  -> Swift/model encoding
  -> encoded label rewrite
  -> recursive target construction
  -> loss diagnostics
```

The target builder must not re-render or re-sample the roll-in state for `prefix_rollin_et_rmp_ce`.

### Instance Identity

`RollinState` stores `object_instance_id` values, not descriptions, category names, compact row text, or bbox tokens. A companion lookup maps each instance id to the normalized object, source index, rendered entry, and tokenized entry span.

The companion lookup should have one owner record so normalized/rendered/encoded state cannot drift across parallel dicts:

```text
EncodedCompactObjectEntry(
  instance_id,
  source_index,
  normalized_object,
  rendered_entry_span,
  encoded_entry_span,
  encoded_entry_token_ids
)
```

`PreparedPrefixRollinExample` should hold `entry_by_instance_id: Mapping[ObjectInstanceId, EncodedCompactObjectEntry]`. Convenience mappings such as `encoded_entry_span_by_instance_id` are derived views, not separate authoritative state.

Duplicate serialized compact entries are valid and contribute multiplicity to trie probabilities. Duplicate `object_instance_id` values are invalid.

### Same-Context Tokenization

Candidate object-entry token ids for trie construction must be sliced from the same full closed Qwen chat-template `input_ids` used for labels and sidecars. The target builder must not re-render entries, standalone-tokenize entry text, or tokenize description/coordinate fragments in isolation.

The required builder input is an instance-id keyed mapping such as:

```text
encoded_entry_span_by_instance_id: object_instance_id -> token span
encoded_entry_token_ids_by_instance_id: object_instance_id -> input_ids span slice
semantic_eos_span: token span supplied by the closed chat template
im_end_token_id: resolved <|im_end|> token id
```

This keeps BPE, leading-space, newline, separator, and added-token effects identical between the teacher hard label and alternate positive candidates.

The target builder should not accept a tokenizer or rendered entry text for candidate construction. Tests should use a spy/failing tokenizer to prove candidate ids are consumed from the encoded span slices rather than standalone entry tokenization.

### Encoded Label Rewrite

The dataset owner encodes the full closed Qwen chat text for input/vision alignment first. It then projects prompt, roll-in prefix, supervised suffix, and chat-template `<|im_end|>` spans onto encoded positions and rewrites `labels`:

```text
prompt/user/image/assistant header: -100
roll-in prefix: -100
supervised suffix: active
eligible <|im_end|>: active with eos_trust_weight weight
padding: -100
```

Alignment checks must run against the rewritten labels, not raw Swift full-assistant labels.

### Padding And Sidecars

The v1 runtime requires right padding for this sidecar objective. If left padding is detected and no explicit sidecar offset rewrite exists, fail fast. After collation, every target must satisfy:

```text
input_ids[batch_index, target.position] == target.teacher_token_id
labels[batch_index, target.position] == target.teacher_token_id
loss uses logits[batch_index, target.position - 1]
```

The v1 dataset/collator path must also fail fast on truncation that removes or partially cuts any supervised suffix object entry or the semantic EOS span. Silent truncation of prefix, suffix, encoded entry spans, or `<|im_end|>` is not allowed for this variant.

### Sidecar Boundary

All new non-tensor metadata should live under the existing `recursive_detection_targets` top-level sidecar unless a new top-level sidecar is explicitly registered through dataset, collator, trainer extras, and model-input stripping boundaries.

### EOS Target Scope

EOS supervision applies whenever a state has no remaining labeled objects after the supervised suffix is consumed. This includes both initial `K=N` examples and `K<N` examples after the last remaining object entry. The EOS target should still be represented when `eos_trust_weight=0` so the loss path and diagnostics have a defined zero-weight contract.

### Uniform K Efficiency

Uniform inclusive `K` has:

```text
P(K=N) = 1 / (N + 1)
E[K] = N / 2
E[remaining] = N / 2
```

The expected cost is not EOS-only over-sampling for large `N`; it is lower supervised-token-per-FLOP because roughly half of object-entry tokens are masked roll-in context. Diagnostics must report supervised suffix token ratio.

## Diagnostics Required

Minimum diagnostics for the new variant:

| Metric | Purpose |
|---|---|
| `rollin/k_histogram` | Prove `K` covers `[0, N]`. |
| `rollin/prefix_token_count_mean` | Check physical prefix length and masking. |
| `rollin/suffix_object_count_mean` | Check remaining suffix distribution. |
| `rollin/supervised_suffix_token_ratio` | Measure supervised-token efficiency under masked roll-in. |
| `trie/hard_label_in_positive_rate` | Detect tokenization or span bugs. |
| `trie/q_sum_error_max` | Detect invalid target normalization. |
| `trie/positive_size_mean` | Show whether multi-positive is actually active. |
| `trie/positive_token_ids_unique_violation_count` | Detect invalid duplicate positive token-id sidecars. |
| `trie/multipositive_position_count` | Denominator for multi-positive diagnostics. |
| `trie/singleton_position_count` | Denominator for singleton/hard positions. |
| `trie/target_entropy_mean` | Measure branch flattening target. |
| `trie/valid_mass_mean` | Measure probability mass over legal children. |
| `trie/pred_valid_entropy_mean` | Detect collapse within valid children. |
| `trie/conditional_kl_q_to_pred_mean` | Directly measure KL between target `q` and predicted valid-conditional distribution. |
| `trie/conditional_top1_matches_q_argmax_rate` | Detect deterministic branch preference when `q` is not one-hot. |
| `trie/max_positive_prob_mean` | Detect single-branch domination. |
| `type_gate/allowed_mass_mean` | Measure schema-type compliance. |
| `type_gate/type_violation_mass_mean` | Track malformed-token pressure. |
| `type_gate/allowed_vocab_fraction_mean` | Detect overly broad type-gate allowed sets. |
| `type_gate/desc_struct_union_rate` | Track mixed desc/struct branch cases where type gate is broad. |
| `type_gate/positive_not_in_allowed_type_count` | Detect type-gate/trie target incompatibility. |
| `type_gate/special_control_in_desc_count` | Detect malformed control tokens in desc allowed mass. |
| `type_gate/position_count` | Denominator for type-gate diagnostics. |
| `eos/eos_trust_weight_mean` | Track the applied EOS CE trust weight. |
| `eos/eos_target_count` | Denominator for EOS target diagnostics. |
| `eos/eos_zero_weight_target_count` | Prove zero-weight EOS targets are represented rather than dropped. |
| `eos/eos_positive_when_nonempty_violation_count` | Detect accidental early-EOS positives. |
| `eos/eos_trust_weight_raw_by_gt_count_bucket` | Compare unclamped empirical prior before clamp. |
| `eos/eos_trust_weight_applied_by_gt_count_bucket` | Track actual applied weight after clamp/source policy. |
| `eos/nonempty_state_count` | Count states where EOS must not be positive. |
| `eos/eos_prob_by_gt_count_bucket` | Support later EOS calibration. |
| `eos/eos_logit_by_prefix_depth` | Measure natural length-driven EOS rise. |
| `eos/continue_vs_eos_margin` | Compare valid-next-object start mass with EOS. |
| `objective/contributing_sample_count` | Detect zero-weight/empty-objective edge cases. |
| `decode/early_eos_rate` | Check premature stopping under remaining GT. |
| `decode/duplicate_rate` | Check repeated object behavior. |
| `decode/recall_by_gt_count_bucket` | Measure dense-scene recall. |

## Unit Test Requirements

Minimum synthetic tests:

| Test | Required assertion |
|---|---|
| Two objects, `K=0` | First object-start/desc branch positives include both legal objects. |
| Shared prefix | After shared desc prefix, coord branch splits correctly and narrows after one coord. |
| Same-context tokenization | Candidate ids are sliced from full closed chat-template encoded entry spans, not standalone tokenization. |
| Emitted object excluded | If prefix emitted object A, A is not positive in remaining trie. |
| Unique positive token ids | Shared next-token identities are aggregated into one token with multiplicity-derived `q`; duplicate positive token ids fail fast. |
| `K` coverage | Repeated sampling covers every integer in `[0, N]`. |
| `K=N` integration | Full dataset/chat-template example masks all object tokens and trains only weighted `<|im_end|>` on the assistant stop span. |
| Prefix masking | Prompt and roll-in prefix labels are `-100`; suffix labels are active. |
| Causal shift | `TokenTarget.position` consumes `logits[position - 1]`. |
| Hard label in positives | Every sparse positive-candidate CE target includes the teacher token, unless explicitly soft-CE-exempt. |
| Soft CE identity | `support=1, balance=1` equals sparse soft CE. |
| Balance-first non-collapse | Raising balance weight penalizes peaked valid-conditional distributions more strongly. |
| Type gate union | Multi-positive positions allow the union of positive child token types. |
| Type gate total loss | `L_total_position = L_main + type_weight * L_type`; disabled gate preserves sparse CE identity. |
| Desc exclusions | `desc` excludes chat/image/pad/control/text-level terminator tokens. |
| Tokenizer stop contract | Active tokenizer resolves `<|im_end|>` and training targets never use `<|endoftext|>` or `<|end_of_text|>`. |
| Assistant stop span | Assistant payload excludes `<|im_end|>` and the chat-template stop marker appears exactly once after the assistant payload span. |
| EOS nonempty | `<|im_end|>` is not positive while remaining labeled objects are nonempty. |
| EOS empty | `<|im_end|>` supervision uses `eos_trust_weight`, not hard weight 1, after every suffix completion that leaves no remaining labels. |
| EOS empirical prior | `eos_trust_weight` from `empirical_unlabeled_poisson_v0` matches expected values for representative GT counts. |
| EOS clamp/source semantics | Empirical prior applies `min_weight`/`max_weight`; `disabled_ablation` returns exact zero. |
| EOS zero weight | `K=N, eos_trust_weight=0` has a finite zero-weight EOS target path, not an empty loss crash. |
| Duplicate serialized entries | Exact duplicate compact rows are valid when `object_instance_id` differs and contribute multiplicity. |
| Encoded label rewrite | Swift full-assistant labels are rewritten so roll-in prefix positions become `-100`. |
| Padding guard | Right padding preserves target positions; left padding fails fast unless offset rewrite exists. |
| DDP metric reducer | Global metrics use all-reduced numerators/denominators or max reducers, not rank-local means. |
| Production formula guard | Production config rejects missing calibrated EOS formula. |
| Ablation constant guard | Ablation config can explicitly use a constant EOS trust weight. |

## Ablation Matrix

Minimum comparisons should isolate one axis at a time:

| ID | Order policy | Roll-in policy | Target policy | Support | Balance | Type gate | EOS policy | Purpose |
|---|---|---|---|---:|---:|---|---|---|
| A0 | canonical | none | one-hot | n/a | n/a | off | baseline/current | Baseline ordering bias. |
| B0 | random permutation | none | one-hot | n/a | n/a | off | same as A0 | Isolate random order. |
| C0 | random permutation | `K uniform [0,N]` | one-hot | n/a | n/a | off | same as B0 | Isolate prefix state coverage. |
| D0 | random permutation | `K uniform [0,N]` | entry trie | 1.0 | 1.0 | off | same as C0 | Sparse soft CE identity. |
| E0 | random permutation | `K uniform [0,N]` | entry trie | 1.0 | 2.0 | off | same as D0 | Balance-first anti-collapse. |
| E1 | random permutation | `K uniform [0,N]` | entry trie | 1.0 | 2.0 | on | empirical EOS v0 | Default first runnable ablation. |
| E2 | random permutation | `K uniform [0,N]` | entry trie | 1.0 | 2.0 | on | disabled/zero | EOS-force removal upper bound. |
| E3 | random permutation | `K uniform [0,N]` | entry trie | 1.0 | 2.0 | on | constant grid | EOS sensitivity grid if needed. |
| E4 | random permutation | `K uniform [0,N]` | entry trie | 1.0 | 2.0 | on | production calibrated artifact | Production candidate after calibration artifact is frozen. |

The first runnable config is `E1` with `empirical_unlabeled_poisson_v0`. It is the default ablation surface, not final production evidence.

Compute normalization for the first pass: keep model/data/optimizer settings and optimizer-step count fixed, and report `supervised_suffix_token_count`, `assistant_token_count`, and wall-clock/GPU-time diagnostics so token-efficiency differences are visible.

Every ablation row must also carry an executable registry record:

- `ablation_id`
- `status`: `planned`, `config_materialized`, `run_started`, `run_complete`, or `archived`
- `config_path` and `base_config_path`
- `objective.variant`
- `order_policy`, `rollin_policy`, and `suffix_order_policy`
- `target_policy`, `support_weight`, `balance_weight`, `type_gate_policy`, `eos_policy`, and `eos_trust_weight.source`
- `train_scope`, `eval_scope`, `dataset_jsonl`, `image_root`, `checkpoint_init`, `seed`, and `replicate_id`
- `launch_shape`: world size, GPU count/model, per-device batch, gradient accumulation, effective batch, max length, and max pixels
- `run_manifest_refs`: pointers to `resolved_config.json`,
  `effective_runtime.json`, `experiment_manifest.json`, `run_metadata.json`,
  `runtime_env.json`, train/eval data provenance, and nullable
  `pipeline_manifest.json`
- `pipeline_manifest_status`: `present`, `not_applicable`, or
  `missing_unexpected`; Stage-1 latest compact detection should normally use
  `not_applicable` rather than fabricating an empty pipeline manifest
- `compute_normalization`: optimizer-step budget, seen-image count, supervised suffix token count, assistant token count, masked prefix token count, forward count, backward count, wall-clock time, and GPU hours
- `required_metric_keys`, `required_artifact_families`, and `artifact_root`
- `interpretation_status`: `hypothesis`, `running`, `result`, `interpretation`, or `stable_contract`

Paper-facing comparisons may use fixed optimizer steps as the first-pass operational rule, but they must also report `seen_image_count` and `supervised_suffix_token_count`. If `supervised_suffix_token_count` differs by more than 5% from the chosen baseline, objective-superiority claims require a same-supervised-token sensitivity run. If `seen_image_count` differs by more than 5%, data-exposure claims require a same-seen-image sensitivity run. If GPU hours differ by more than 10%, efficiency claims must report both fixed-step and fixed-compute views.

Use canonical scope labels only:

- `tiny`
- `val200`
- `limit=200`
- `first-200`
- `full-val`
- `proxy`
- `test-dev`

Every metric claim must also include `coordinate_surface`, `bbox_format`, `eval_surface` (`raw`, `scored`, `guarded`, or `scored_guarded`), `checkpoint_path`, `config_path`, `artifact_root`, `metrics_json`, `parser_mode`, and `decode_settings`.

## Research Artifact Contracts

Training scalar diagnostics are not enough for research claims. The following artifact families should be defined before production-scale claims:

These artifact families must be registered in `docs/ARTIFACTS.md` before they are used as evidence for paper-facing claims.

| Artifact | Required purpose |
|---|---|
| `prefix_forced_eval/summary.json` | Forced-prefix scoring over GT/random/self prefixes with NLL, valid mass, EOS margin, and continuation recovery. |
| `prefix_forced_eval/per_case.jsonl` | One row per forced-prefix case with `record_idx`, `gt_count`, `prefix_k`, `prefix_mode`, `remaining_gt_count`, `nll_teacher_suffix`, `eos_logit`, `continue_logsumexp`, `continue_minus_eos_margin`, and `valid_mass`. |
| `permutation_nll/summary.json` | Same image/object set under multiple GT permutations; report NLL mean/std/CV by GT-count bucket. |
| `prefix_jitter/summary.json` | V2 diagnostic for clean prefix vs shuffled prefix vs self-prefix vs jittered prefix; v1 keeps jitter out of training. |
| `prefix_rollin_decode_diagnostics/summary.json` | Post-eval summary separating raw, scored, guarded, and scored-guarded parse/duplicate/recall/early-EOS metrics, with non-materialized views explicitly marked. |
| `eos_calibration_probe/summary.json` | Full-val pre-calibration probe for the EOS formula with exact checkpoint, config, dataset, eval artifacts, and non-collapse FP definition. |
| `eos_calibration_formula.json` | Versioned formula artifact for any production `eos_trust_weight` policy. |

`noncollapse_fp_rate` must be defined inside the calibration probe/formula artifact, including raw-vs-guarded source, IoU/semantic thresholds, score policy, duplicate/collapse exclusion rule, denominator, checkpoint, config, dataset, artifact roots, command, git commit, and row hash.

Common research artifact header fields:

- `schema_version`
- `artifact_family`
- `created_at_utc`
- `git_commit`
- `command`
- `config_path`
- `resolved_config_json`
- `checkpoint_path`
- `dataset_jsonl`
- `image_root`
- `dataset_split`
- `dataset_fingerprint_or_row_hash`
- `sample_scope`
- `limit`
- `coordinate_surface`
- `bbox_format`
- `eval_surface`
- `artifact_root`
- `source_run_artifacts`

`prefix_forced_eval/per_case.jsonl` rows must include at least:

- `record_idx`
- `image`
- `gt_count`
- `prefix_k`
- `prefix_mode`
- `prefix_seed`
- `prefix_object_instance_ids`
- `remaining_object_instance_ids`
- `teacher_suffix_object_instance_ids`
- `remaining_gt_count`
- `nll_teacher_suffix_total`
- `nll_teacher_suffix_per_token`
- `nll_first_remaining_entry`
- `supervised_suffix_token_count`
- `assistant_token_count`
- `masked_prefix_token_count`
- `hard_label_in_positive_rate`
- `positive_size_mean`
- `target_entropy_mean`
- `valid_mass_mean`
- `valid_mass_min`
- `type_allowed_mass_mean`
- `eos_logit`
- `continue_logsumexp`
- `continue_minus_eos_margin`
- `eos_trust_weight`
- `row_hash`

`eos_calibration_formula.json` must be impossible to confuse with empirical v0. It must include `formula_class`, `calibration_status`, `production_approved`, `calibration_probe_ref`, `fit_scope`, `noncollapse_fp_definition_id`, `valid_gt_count_range`, `clipping_rules`, `interpolation_rules`, and `extrapolation_rules`. `production_approved=true` is valid only when `formula_class=calibrated_production`.

## Acceptance Criteria

The design is implemented only when all of the following are true:

- `prefix_rollin_et_rmp_ce` is the only new active multi-positive roll-in variant.
- It rejects non-`compact_full` templates.
- It samples `K` uniformly over `[0, N]`.
- It masks the roll-in prefix and trains the suffix only.
- It builds entry-trie targets from remaining objects only.
- It slices candidate token ids from the same closed chat-template encoding used for labels and sidecars.
- The target builder consumes encoded entry slices and does not standalone-tokenize candidate entries.
- EOS targets align to the closed chat-template `semantic_eos_span`, not a manual payload token or tokenizer-level text EOS.
- It aggregates duplicate object multiplicity into unique positive token ids and `q`, never repeated token ids.
- Exact duplicate serialized rows remain distinct by `object_instance_id` and are consumed one instance at a time.
- It fails if hard labels are missing from positives.
- It proves the integrated batch loss uses sparse local next-token multi-positive CE/support-balance, not teacher-only hard CE fallback.
- It exposes support/balance weights with balance-first ablation default.
- It adds compact token-type gate diagnostics and loss terms.
- It treats `<|im_end|>` as the sole EOS token.
- It fails fast if the tokenizer/template path cannot use `<|im_end|>` as the compact training stop marker.
- It never constructs compact training templates or objective targets with `<|endoftext|>` or `<|end_of_text|>`.
- It implements `empirical_unlabeled_poisson_v0` as an ablation EOS prior from the user-provided formula.
- It preserves a production-calibrated EOS formula interface backed by a versioned artifact and rejects empirical EOS sources on production.
- It requires `experiment.surface` to gate ablation/smoke vs production EOS policies.
- It rejects production runs without an explicit calibrated EOS formula.
- It fails fast on left padding, packing/cache reuse, or truncation that cuts supervised entry spans or the semantic EOS span.
- It removes old legacy config/runtime compatibility for this surface.
- It preserves normal autoregressive generation as the main decode path.
