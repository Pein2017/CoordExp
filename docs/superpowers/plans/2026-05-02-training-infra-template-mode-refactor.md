# Training Infrastructure Greenfield Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor CoordExp Stage-1 detection training into a greenfield latest-schema stack where raw detection data, templates, tokenization, objective targets, loss normalization, packing, and evaluation are explicit, typed, validated, and minimal.

**Architecture:** Use one full-sequence teacher-forced autoregressive pipeline:

```text
raw JSONL row
-> normalized detection sample
-> seeded object ordering
-> template-rendered assistant sequence with authoritative spans
-> chat-template tokenization and span alignment
-> token-level target builder
-> SFT / random-order SFT / Random-Permutation ET-RMP-CE loss
```

The new codebase does not carry old set-continuation, candidate-scoring, suffix-row, explicit prefix-sampling, or archived-config compatibility modules. Historical behavior remains available only through archived docs, older branches, or pinned commits.

**Tech Stack:** Python dataclasses and typing, current Qwen3-VL tokenizer/chat-template path, `conda run -n ms python -m pytest`, repo-local superpowers workflow.

---

## Execution Boundary

This superpowers document is an implementation plan, not an implementation. Do not modify production code until the user explicitly starts execution from this plan.

Primary design spec:

```text
docs/superpowers/specs/2026-05-02-training-infra-template-mode-refactor-design.md
```

Source-of-truth raw data:

```text
public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
```

Target branch:

```text
codex/compact-detection-sequence
```

## Non-Negotiable Invariants

- Preserve geometry and image alignment end-to-end.
- Keep canonical `stage1_json_pretty` as the current JSON ablation/control template, not as an old-schema compatibility layer.
- Keep `compact_full` as the compact coord-token template using `<|object_ref_start|>` and `<|box_start|>`.
- Keep template logic independent from objective logic.
- Keep training-objective construction independent from raw data parsing.
- Fail fast on unsupported template, geometry, coordinate surface, parser mode, packing mode, or obsolete config key.
- Do not implement old candidate-balanced set-continuation, candidate-energy/logZ scoring, chunk-level MP, candidate-branch CE, explicit runtime prefix sampling, suffix-row generation, PEM, or margin ranking in the new code path.
- Do not describe ET-RMP-CE as "MP on the first token of each object"; implement it as trie-based sparse multi-positive CE at every true multi-child object-entry trie node.
- Standardize terminology on `trie`: use `object-entry trie`, `trie node`, `multi-child trie node`, `trie child token`, and `valid child set`.
- Decide hard CE versus multi-positive CE from the trie at each token position: `len(valid_next_tokens) == 1` is hard CE, and `len(valid_next_tokens) > 1` is support/balance CE.
- Token names such as `<|object_ref_start|>` and `<|box_start|>` are role metadata, not static loss-policy switches.
- Keep the active remaining-object multiset as private objective-builder state; do not expose it as a separate public training mode or module.
- Treat semantic image-balanced normalization as a distinct objective profile, not as part of objective-preserving equivalence.
- Scope exact-equivalence claims to `stage1_json_pretty`, packing disabled, same random-order family, legacy row-mean denominator, and explicit compiled per-position weights.
- Do not add CLI flags when typed config sections can express the behavior.

## Phase 0: Baseline And Docs Hygiene

- [ ] Record current branch, dirty status, and key current implementation surfaces before code changes.

```bash
git status --short --branch
rg -n "detection_sequence_format|stage1_set_continuation|trainer_variant|custom|prefix_conditioning|branch_support_weight|candidate_balanced" src configs tests docs
```

- [ ] Snapshot the source-of-truth raw schema counts from `val.coord.jsonl`.
- [ ] Capture one current canonical Stage-1 JSON rendered output as the `stage1_json_pretty` parity fixture.
- [ ] Capture frozen local-target parity fixtures for canonical `stage1_json_pretty` ET-RMP-CE toy samples before removing legacy row-path code; these fixtures are test data, not runtime compatibility modules.
- [ ] Verify archived/superseded banners on older superpowers notes retained as history for retired Stage-1 set-continuation work.
- [ ] Keep the active 2026-05-02 spec/plan as the execution source for the greenfield refactor.

Acceptance:

- The old implementation surface is documented, but not carried forward as a target architecture.
- The canonical JSON parity fixture is available before rewriting rendering.
- Local target parity fixtures exist for hard-CE versus trie-CE decisions, valid child sets, and multiplicity weights under canonical JSON.
- Archived docs are visually marked as historical provenance, not active implementation plans.

## Phase 1: Detection Data Contract

Target module:

```text
src/detection/data.py
```

Tests:

```text
tests/test_detection_raw_schema_contract.py
tests/test_detection_normalization_contract.py
```

Steps:

- [ ] Add frozen typed containers for raw rows, raw objects, metadata, coordinate-token boxes, normalized objects, and normalized samples.
- [ ] Add `parse_raw_detection_row(row: Mapping[str, Any]) -> RawDetectionRow`.
- [ ] Add `normalize_detection_row(raw: RawDetectionRow, *, object_ordering: ObjectOrderingPlan) -> NormalizedDetectionSample`.
- [ ] Validate required top-level keys and object keys against the source-of-truth schema.
- [ ] Validate compact-v1 geometry as exactly four coordinate-token strings.
- [ ] Preserve object provenance: source object position, normalized object index, category fields, `coco_ann_id`, and stable object-instance ID.
- [ ] Add seeded object-ordering plans for `sorted` and `random_permutation`.

Acceptance:

- Raw source rows parse into typed containers.
- Normalized samples preserve object count, geometry tokens, object provenance, and selected ordering.
- Random-order plans record seed source and realized permutation.

## Phase 2: Template Registry And Render Spans

Target module:

```text
src/detection/template.py
```

Tests:

```text
tests/test_detection_template_registry.py
tests/test_detection_stage1_json_pretty_template.py
tests/test_detection_compact_full_template.py
```

Steps:

- [ ] Define `TemplateId = Literal["stage1_json_pretty", "compact_full"]`.
- [ ] Define `TemplateCapabilities`, `CharSpan`, `RenderedObjectEntry`, `RenderedAssistantSequence`, and `RenderedConversation`.
- [ ] Implement `Stage1JsonPrettyTemplate` as the current canonical Stage-1 JSON template, preserving strict structure: JSON root -> `objects` list -> object `desc` -> object `bbox_2d` -> next object -> strict ending closure.
- [ ] Pin the `stage1_json_pretty` v1 profile to `object_field_order=desc_first`, `coordinate_surface=coord_token`, `bbox_format=xyxy`, and the captured canonical spacing/closure fixture.
- [ ] Implement `CompactFullTemplate` grammar: `<|object_ref_start|>{desc}<|box_start|>{x1}{y1}{x2}{y2}` with newline separators and template-defined terminal stop span(s) after the final row, never JSON `]}` closure.
- [ ] Return assistant-local authoritative spans for object entries, desc, bbox opener, coordinate tokens, separators, terminal close, stop markers present in assistant text, structural tokens, and trie-eligible entry spans.
- [ ] Add strict parser round-trip tests for each template.
- [ ] Reject unsupported geometry, coordinate surface, or template/prompt mismatches before tokenization.

Acceptance:

- `stage1_json_pretty` is byte-identical to the captured canonical Stage-1 JSON fixture.
- `compact_full` renders the approved token grammar exactly.
- Both templates expose spans sufficient for SFT masks and trie target construction.
- No `coordjson_legacy` template exists in the greenfield target.

## Phase 3: Tokenization And Span Alignment

Target module:

```text
src/detection/tokenization.py
```

Tests:

```text
tests/test_detection_template_span_alignment.py
tests/test_token_span_masks_from_templates.py
tests/test_sft_preparation_contract.py
```

Steps:

- [ ] Apply the configured Qwen3-VL chat template once per rendered conversation.
- [ ] Add the full assistant span after chat-template rendering.
- [ ] Map renderer-owned assistant-local character spans to token spans.
- [ ] Fail if a span maps to zero tokens or crosses an unexpected tokenizer boundary.
- [ ] Build label masks and role masks from rendered spans.
- [ ] Add compact token-role tests for `<|object_ref_start|>`, `<|box_start|>`, and coordinate special tokens.
- [ ] Add canonical JSON token-role tests for object, desc, bbox, separator, and closure spans.

Acceptance:

- Training paths consume renderer-owned spans directly.
- No path guesses object spans by reparsing serialized text.
- Compact and canonical JSON both produce expected token-role masks.

## Phase 4: Latest Config Schema

Target files:

```text
src/config/schema.py
src/config/loader.py
src/config/__init__.py
src/config/resolve.py
src/sft.py
configs/base.yaml
configs/stage1/recursive_detection_ce.yaml
tests/test_latest_training_config_contract.py
tests/test_config_schema.py
```

Steps:

- [ ] Preserve framework runtime sections such as `model`, framework `template`, `training`, and `deepspeed`; these remain the home for Qwen/ms-swift chat-template settings, TrainArguments, optimizer, batch, checkpoint, and distributed runtime knobs.
- [ ] Add latest detection-task sections: `data`, `prompt`, `detection_template`, `objective`, `packing`, `evaluation`, `validation`, and optional local `debug`.
- [ ] Collapse old training-mode/loss/normalization concepts into `objective`.
- [ ] Reject obsolete keys: `custom`, `trainer_variant`, `stage1_set_continuation`, `prefix_conditioning`, `legacy_candidate_branch`, `candidate_balanced`, `branch_support_weight`, `branch_balance_weight`, and any explicit prefix-sampling knobs.
- [ ] Preserve strict unknown-key validation.
- [ ] Route Stage-1 latest-schema runs by `objective.id`, not `trainer_variant`; `recursive_detection_ce` selects the detection objective-preparation/loss adapter and still uses the standard teacher-forced causal-LM forward process.
- [ ] Record raw config and resolved canonical config in run manifests.
- [ ] Include template ID, template version, object ordering, objective variant, state-weighting policy, normalization policy, tokenizer ID, parser mode, and packing settings in cache fingerprints.
- [ ] Add a non-alias terminology table for manifests and cache audits: old config names are rejected, but their latest-schema replacements are documented for humans.

Latest-schema shape:

```yaml
model:
  # Existing model/runtime model settings.

template:
  # Existing Qwen/ms-swift chat-template settings, not detection output grammar.

training:
  # Existing framework TrainArguments-style settings.

deepspeed:
  # Existing runtime section when used.

data:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  image_root: public_data/coco
  max_objects: 60
  object_ordering: random_permutation

prompt:
  system_variant: stage1_detection
  user_variant: dense_detection
  include_template_summary: true

detection_template:
  id: stage1_json_pretty
  coordinate_surface: coord_token
  bbox_format: xyxy
  object_field_order: desc_first
  strict_parse: true

objective:
  id: recursive_detection_ce
  variant: random_permutation_et_rmp_ce
  trie_support_weight: 2.0
  trie_balance_weight: 1.0
  state_weighting: legacy_row_mean_prefix_mixture_equivalence
  normalization: legacy_row_mean_equivalence

packing:
  static_packing: false
  padding_free_packed: false

evaluation:
  expected_template: stage1_json_pretty
  parser_mode: strict_expected

validation:
  validate_span_alignment: true
  validate_template_capabilities: true
```

Compact candidate override:

```yaml
detection_template:
  id: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy

objective:
  state_weighting: uniform_permutation
  normalization: semantic_image_bucket_balanced

evaluation:
  expected_template: compact_full
  parser_mode: strict_expected
```

Terminology mapping for docs, manifests, and cache-fingerprint audits only:

| Previous surface | Latest surface | Notes |
| --- | --- | --- |
| `detection_sequence_format=coordjson` or JSON-like Stage-1 output | `detection_template.id=stage1_json_pretty` | Pinned v1 canonical JSON profile |
| `detection_sequence_format=compact*` for the approved full compact row | `detection_template.id=compact_full` | Coord-token xyxy only in v1 |
| `object_ordering=random` | `data.object_ordering=random_permutation` | Latest name makes dynamic object-order sampling explicit |
| `coord_mode` / coordinate prompt knobs | `detection_template.coordinate_surface` plus prompt profile | Manifest must record both rendering and prompt summary |
| `object_field_order` | `detection_template.object_field_order` for JSON templates | Pinned to `desc_first` for `stage1_json_pretty` v1 |

Acceptance:

- Latest configs do not need and cannot use `custom`.
- Obsolete knobs fail fast with clear messages.
- Configs distinguish equivalence profiles from new objective ablation profiles.
- Framework `template:` and detection `detection_template:` cannot collide.
- A latest-schema Stage-1 smoke config is runnable through `src/sft.py` without `trainer_variant`.

## Phase 5: SFT And Random-Order SFT Preparation

Target module:

```text
src/detection/objective.py
```

Tests:

```text
tests/test_sft_preparation_contract.py
tests/test_random_order_sft_contract.py
```

Steps:

- [ ] Build sorted SFT examples from normalized samples plus selected template.
- [ ] Build random-order SFT examples by applying one seeded random object permutation before rendering.
- [ ] Tokenize the full rendered conversation once.
- [ ] Apply hard CE to all assistant target positions.
- [ ] Attach rendered sequence, object-order metadata, masks, and provenance to each example.

Acceptance:

- Sorted SFT is a pure hard-CE full-sequence objective.
- Random-order SFT differs only by realized object permutation.
- Both templates support sorted and random-order SFT.

## Phase 6: Recursive Detection CE Target Builder

Target module:

```text
src/detection/objective.py
```

Tests:

```text
tests/test_recursive_detection_ce_target_builder.py
tests/test_random_permutation_et_rmp_ce_contract.py
tests/test_compact_et_rmp_span_contract.py
```

Steps:

- [ ] Add an internal `RecursiveDetectionState` with rendered object-entry token sequences, remaining-object multiset, current teacher object, teacher-forced entry prefix, and object-entry trie.
- [ ] Walk the teacher-forced full assistant sequence left to right.
- [ ] Outside the current object-entry span, emit hard-CE targets.
- [ ] Inside the current object-entry span, compute active remaining objects compatible with the teacher-forced prefix.
- [ ] Use hard CE when the valid trie child set has one token.
- [ ] Use support/balance CE when the valid trie child set has multiple tokens.
- [ ] Continue along the teacher-selected object entry.
- [ ] Remove exactly that object instance after its full entry span is emitted.
- [ ] Preserve exact duplicate rendered entries as separate object instances.
- [ ] Add desc-prefix collision tests proving `<|box_start|>` can be a trie child.
- [ ] Add same-class coordinate-divergence tests proving coordinate tokens can be first multi-child trie nodes.
- [ ] Compare frozen canonical JSON local-target parity fixtures against the new full-sequence builder: hard CE versus trie CE, valid child sets `V`, and multiplicity weights `q(v)` must match for every represented legacy sampled state.

Loss formula:

```text
P_valid = sum_{v in V} p_theta(v | context)
L_valid_support = -log(P_valid)
L_valid_balance = - sum_{v in V} q(v) * log(p_theta(v | context) / P_valid)
L_trie = trie_support_weight * L_valid_support
       + trie_balance_weight * L_valid_balance
q(v) = object_count(child_v) / object_count(current_node)
```

Acceptance:

- Random-Permutation ET-RMP-CE is token-level target-distribution replacement at true multi-child object-entry trie nodes.
- The objective does not score candidate chunks, compute candidate energy/logZ, or require branch-forward comparisons.
- Compact templates drive the objective without JSON keys or JSON closure fragments.
- The remaining-object multiset exists only as private objective-builder state.
- Exact-equivalence claims require both compiled position-weight parity and local trie-target parity.

## Phase 7: State Weighting And Loss Normalization

Target module:

```text
src/detection/objective.py
```

Tests:

```text
tests/test_recursive_detection_state_weighting.py
tests/test_length_insensitive_loss_normalization.py
```

Steps:

- [ ] Add `state_weighting=legacy_row_mean_prefix_mixture_equivalence`, compiling the old prefix exposure and row-mean reduction into deterministic per-position weights over a single sampled permutation.
- [ ] Encode the current prefix mixture `empty/random_subset/leave_one_out/full_prefix = 0.30/0.45/0.20/0.05`.
- [ ] Implement exact compiled weights as `alpha_t = E_K[1{t in U_{pi,K}} / |U_{pi,K}|]`, where `U_{pi,K}` is the supervised token set for the old sampled row.
- [ ] Add diagnostic exposure tests for `n > 1`: `entry_state_exposure(n, j) = 0.30 + 0.45 * (j - 1) / (n - 1) + 0.20 * 1[j = n]`.
- [ ] Test that opening schema/control tokens exposed only under an empty prefix have exposure `0.30`, then separately verify their exact `alpha_t` includes the old row-mean denominator.
- [ ] Test that separator tokens use the matching emitted-object exposure and exact row-mean denominator.
- [ ] Test that terminal close/EOS has exposure `1.0` and exact row-mean denominator.
- [ ] Add a singleton-object fixture for the `leave_one_out` degeneracy; do not use the `n > 1` formula for `n = 1`.
- [ ] Prove `alpha_t` parity is paired with identical local loss terms `l_t`; matching only position weights is not enough for objective equivalence.
- [ ] Add `state_weighting=uniform_permutation`, where every entry/separator state has weight `1.0`; label this as a new objective ablation.
- [ ] Add `normalization=legacy_row_mean_equivalence` for parity tests.
- [ ] Add `normalization=semantic_image_bucket_balanced` for length-insensitive compact-vs-JSON ablations.
- [ ] Report denominator diagnostics for every objective run: atom counts, object counts, semantic-role token counts, multi-child trie-token fraction, boundary/stop fraction, state-position weights, effective weighted trie contribution, and loss by GT-count bucket.

Semantic normalization formula:

```text
L_atom(a) = sum_{t in tokens(a)} token_weight(t) * loss(t)
            / sum_{t in tokens(a)} token_weight(t)

L_object(i,k) =
  sum_{a in object_atoms(i,k)} role_weight(role(a)) * L_atom(a)
  / sum_{a in object_atoms(i,k)} role_weight(role(a))

L_objects(i) = (1 / max(1, num_objects_i)) * sum_k L_object(i,k)

L_image(i) =
  lambda_objects * L_objects(i)
  + lambda_boundary * L_boundary(i)
  + lambda_schema * L_schema_control(i)

L_batch = sum_i bucket_weight(bucket(num_objects_i)) * L_image(i)
          / sum_i bucket_weight(bucket(num_objects_i))
```

Acceptance:

- Exact-equivalence tests preserve prior state exposure and row-mean denominator behavior.
- Equivalence fixtures cover opener/control sparsity, separators, close/EOS, and singleton samples.
- Semantic normalization is available and clearly artifact-labeled as a new objective profile.
- Pretty JSON and compact examples with equal semantic losses produce equal normalized image losses under the semantic profile.

## Phase 8: Packing Contracts

Target module:

```text
src/detection/packing.py
```

Tests:

```text
tests/test_packing_template_contracts.py
tests/test_packing_cache_fingerprints.py
```

Steps:

- [ ] Move static SFT packing eligibility and fingerprint construction out of `src/sft.py`.
- [ ] Define `PackedSequenceMetadata` for packed examples.
- [ ] Include template ID, template version, prompt profile, tokenizer ID, object ordering, objective variant, state-weighting policy, normalization policy, loss-mask version, and preprocessing version in fingerprints.
- [ ] Preserve static SFT packing for eligible full-sequence hard-CE examples.
- [ ] Reject recursive detection CE packing until trie target metadata preservation is implemented.
- [ ] Keep padding-free packed runtime as an explicit experimental runtime variant with validation.

Acceptance:

- Packing no longer depends on implicit template assumptions.
- Cache fingerprints change only when behaviorally meaningful resolved fields change.
- Unsupported packing combinations fail before expensive preprocessing.

## Phase 9: Strict Evaluation And Parsing

Target module:

```text
src/detection/evaluation.py
```

Tests:

```text
tests/test_detection_template_parsing_eval.py
```

Steps:

- [ ] Add strict expected-template parse helpers for `stage1_json_pretty` and `compact_full`.
- [ ] Keep salvage and auto-detect parsing diagnostic-only.
- [ ] Record expected template ID, parser mode, coordinate surface, and benchmark scope in artifact manifests.
- [ ] Add compact strict parse tests for valid output and invalid JSON-like output.
- [ ] Add canonical JSON strict parse tests for valid output and compact-like output.
- [ ] Label strict expected-template eval as a metric-surface break relative to historical salvage/auto-detect parsing; old and new metrics are not directly comparable unless parser mode is reported.

Acceptance:

- Metric-bearing eval does not silently accept the wrong template.
- Diagnostic parsing can inspect malformed outputs only when explicitly requested.
- Benchmark tables include `expected_template` and `parser_mode` wherever historical results are compared.

## Integration Smokes

Commands:

```bash
conda run -n ms python -m pytest \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_template_registry.py \
  tests/test_detection_template_span_alignment.py \
  tests/test_latest_training_config_contract.py
```

```bash
conda run -n ms python -m pytest \
  tests/test_sft_preparation_contract.py \
  tests/test_random_order_sft_contract.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_random_permutation_et_rmp_ce_contract.py
```

```bash
conda run -n ms python -m pytest \
  tests/test_recursive_detection_state_weighting.py \
  tests/test_length_insensitive_loss_normalization.py \
  tests/test_packing_template_contracts.py \
  tests/test_detection_template_parsing_eval.py
```

Smoke expectations:

- [ ] Canonical `stage1_json_pretty` sorted SFT materializes tokenized examples and masks.
- [ ] Compact sorted SFT materializes tokenized examples and masks.
- [ ] Random-order SFT materializes deterministic object-order metadata from a fixed seed.
- [ ] Trie-disabled recursive detection CE reproduces hard CE on the rendered random-order sequence.
- [ ] Canonical and compact Random-Permutation ET-RMP-CE materialize valid trie targets on toy samples.
- [ ] Legacy row-mean equivalence weighting and semantic normalization produce distinct, labeled denominator diagnostics.
- [ ] No smoke result is promoted as full validation without explicit scope.

## Adoption Ablation Matrix

Required before adopting compact Random-Permutation `ET-RMP-CE` with support/balance reweighting from the beginning of Stage-1:

| Run | Template | Objective | State weighting | Normalization | Purpose |
| --- | --- | --- | --- | --- | --- |
| R0 | existing checkpoint | eval-only | n/a | n/a | Anchor decoding controls and metric scope |
| J1 | `stage1_json_pretty` | sorted SFT | n/a | token mean | Stable canonical JSON baseline |
| C1 | `compact_full` | sorted SFT | n/a | token mean | Isolate compact-format shift |
| J2 | `stage1_json_pretty` | random-order SFT | n/a | token mean | Isolate random ordering under JSON prior |
| C2 | `compact_full` | random-order SFT | n/a | token mean | Isolate compact plus random-order interaction |
| J3 | `stage1_json_pretty` | trie-disabled recursive CE | legacy row-mean prefix-mixture equivalence | legacy row-mean equivalence | Verify scaffold equivalence under JSON |
| C3 | `compact_full` | trie-disabled recursive CE | uniform permutation | semantic image-balanced | Verify scaffold mechanics under compact target profile |
| J4 | `stage1_json_pretty` | ET-RMP-CE `1.0/1.0` | legacy row-mean prefix-mixture equivalence | legacy row-mean equivalence | Pure trie CE under JSON |
| C4 | `compact_full` | ET-RMP-CE `1.0/1.0` | uniform permutation | semantic image-balanced | Pure trie CE under compact target profile |
| J5 | `stage1_json_pretty` | ET-RMP-CE `2.0/1.0` | legacy row-mean prefix-mixture equivalence | legacy row-mean equivalence | Support-reweighted JSON control |
| C5 | `compact_full` | ET-RMP-CE `2.0/1.0` | uniform permutation | semantic image-balanced | Support-reweighted compact candidate |

Minimal ablation when compute is limited:

- Eval-only reference checkpoint.
- `stage1_json_pretty` sorted SFT.
- `compact_full` sorted SFT.
- `stage1_json_pretty` random-order SFT.
- `compact_full` random-order SFT.
- `stage1_json_pretty` ET-RMP-CE `2.0/1.0` with `legacy_row_mean_prefix_mixture_equivalence` and `legacy_row_mean_equivalence`.
- `compact_full` ET-RMP-CE `1.0/1.0` with semantic normalization.
- `compact_full` ET-RMP-CE `2.0/1.0` with semantic normalization.

Required parity probes:

- Current ET-RMP versus weighted full-sequence ET-RMP on `stage1_json_pretty`, same checkpoint init, same decode/eval scope, packing disabled.
- Frozen local-target parity on `stage1_json_pretty`: hard-CE versus trie-CE decisions, valid child sets `V`, and multiplicity weights `q(v)` match the captured legacy row-path fixtures.
- Weighted full-sequence ET-RMP versus unit-weight random-permutation ET-RMP, to isolate the effect of dropping runtime prefix sampling.
- Boundary/stop calibration on a fixed FN-heavy subset, comparing teacher-forced mass for `continue_with_next_object` versus `close_now`, plus predicted count and FN by GT-count bucket.
- Duplicate and late-divergence fixtures: exactly identical rendered entries, same-desc objects diverging at the first bbox token, and same-desc objects diverging at a later coordinate token.
- Template-pair probe for `stage1_json_pretty` versus `compact_full`, with packing disabled first and scope reported as `tiny`, `val200`, full-val, proxy, raw-text, or coord-token.

Metrics to collect:

- Strict parse success.
- Predicted object count.
- Count error by GT-count bucket.
- FN rate, especially on high-object-count images.
- Object coverage.
- Duplicate rate.
- Early-stop rate.
- Over-continuation rate.
- Coordinate-token loss.
- Structural-token loss.
- Entry-trie valid-support probability mass.
- Boundary continue-vs-stop teacher-forced mass.
- `loss/rmp_trie_support`, `loss/rmp_trie_balance`, `loss/rmp_boundary_ce`, `loss/rmp_close_ce`, and `loss/rmp_eos_ce` where applicable.
- `valid_child_mass_mean`, `p10`, `p50`, and `p90`, split by desc/coordinate/structural/other where possible.
- Multi-child trie-node count by token type.
- Multi-child trie-node entropy.
- Teacher trie-child top-1 accuracy.
- Valid child top-1 accuracy.
- Coordinate trie-child accuracy.
- Bbox quality for matched objects.
- Per-object-count buckets.
- State-position weight summaries by emitted-object index.
- Normalization denominator diagnostics.
- Standard detection metrics, with exact scope such as `val200`, full-val, proxy, raw-text, or coord-token.
- Throughput, peak memory, and cache hit rate.

## Risk Register

- Random object order may remove useful spatial ordering priors; the random-order SFT control is mandatory.
- Removing explicit runtime prefix sampling without state-position weights changes recursive-state exposure.
- Semantic image-balanced normalization changes the aggregation layer and must be interpreted as an objective profile.
- Support reweight may reduce FN but increase over-continuation or FP if terminal stop calibration is weak.
- Compact templates must provide precise token spans or MP can be applied to the wrong token positions.
- Same-class objects often branch first at bbox or coordinate tokens, not desc tokens.
- Desc-prefix collisions can make `<|box_start|>` a true trie child; do not hard-code it as always-hard.
- Duplicate rendered entries must remain separate object instances.
- Images with many objects produce more multi-child trie nodes; state weighting and normalization diagnostics must be reported by object-count bucket.
- Terminal stop and control CE must stay strong enough so the model learns when to stop.
- Static packing and future padding-free packing are separate runtime optimizations; do not resurrect candidate scoring or suffix rows for efficiency.

## Review Gates

- [ ] Gate 0 review after baseline docs hygiene and canonical JSON fixture capture.
- [ ] Gate 1 review after typed containers and source schema tests.
- [ ] Gate 2 review after canonical `stage1_json_pretty` and `compact_full` template spans pass.
- [ ] Gate 3 review after tokenization/span-mask alignment pass.
- [ ] Gate 4 review after latest config schema rejects obsolete knobs.
- [ ] Gate 5 review after sorted and random-order SFT pass.
- [ ] Gate 6 review after recursive trie target builder and duplicate handling pass.
- [ ] Gate 7 review after state weighting and normalization diagnostics pass.
- [ ] Gate 8 review after packing and strict parsing contracts pass.
- [ ] Gate 9 review before launching any full training ablation.
- [ ] Gate 10 review after smoke/preflight runs pass and all critical fixes are landed.
- [ ] Gate 11 review after production training is launched in `tmux` and early metrics are healthy.

Each gate review should include exact changed files, commands run, test output summary, any mathematical deviations, and unresolved risks.

## Final Deliverable 1: Smoke / Preflight Runs

Goal:

- Validate implementation correctness before production training.
- Catch implementation bugs early on narrow, debuggable surfaces.
- Debug training instability, decoding instability, malformed sequence modes, OOM risk, and throughput regressions.
- Verify that the final production configuration is safe to launch.

Operational scope:

- [ ] Use the local 8 GPUs with maximal safe parallelism, but cap concurrency when simultaneous smokes would obscure OOM root cause or saturate shared I/O.
- [ ] Run all smoke/preflight jobs under explicit config files or generated resolved configs; do not rely on ad hoc CLI-only settings.
- [ ] Record command, config path, output directory, GPU allocation, seed, commit SHA, and resolved config for every smoke.
- [ ] Keep smoke scopes explicit, for example `tiny`, `limit=32`, `limit=128`, `val200`, or another named preflight subset; do not present smoke metrics as full validation.
- [ ] Stop a smoke immediately on NaN/Inf, repeated malformed decoding, catastrophic eval collapse, persistent OOM, or impossible alignment/mask diagnostics.

Smoke matrix:

| Smoke | Template | Prompt | Ordering | Packing/runtime | Objective/loss | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | `stage1_json_pretty` | canonical | sorted | packing off | hard CE | Sanity baseline for latest pipeline |
| S1 | `compact_full` | compact prompt | sorted | packing off | hard CE | Isolate compact rendering and masks |
| S2 | `compact_full` | compact prompt | random per epoch | packing off | hard CE | Validate random object shuffling without trie MP |
| S3 | `compact_full` | prompt variant enabled | random per epoch | packing off with best batch acceleration | hard CE | Validate prompt variant plus random ordering |
| S4 | `compact_full` | prompt variant enabled | random per epoch | packing on, if implemented | hard CE | Validate static packing correctness and throughput |
| S5 | `compact_full` | prompt variant enabled | random per epoch | best validated runtime | ET-RMP-CE `1.0/1.0` | Validate trie targets without support reweight |
| S6 | `compact_full` | prompt variant enabled | random per epoch | best validated runtime | ET-RMP-CE `2.0/1.0` | Validate support-reweighted target profile |
| S7 | `compact_full` | prompt variant enabled | random per epoch | best validated runtime | ET-RMP-CE plus bidirectional gating, if implemented | Validate gating stability before production |

Hard-CE coverage checks:

- [ ] Verify hard CE supervises intended special/control tokens outside `trie_eligible_span`.
- [ ] Verify hard CE supervises open-vocabulary `desc` tokens when no multi-child trie target applies.
- [ ] Verify hard CE supervises coordinate tokens when the valid trie child set is singleton.
- [ ] Verify support/balance CE only replaces hard CE at true multi-child object-entry trie nodes.
- [ ] Verify terminal, assistant-end, EOS, chat-stop, separator, and JSON/compact closure positions remain hard CE by construction.

Bidirectional gating smoke, if implemented:

- [ ] Implement and validate the gate only as a latest objective/loss adapter; do not revive the archived set-continuation bidirectional gate module.
- [ ] Separate special-token positions from raw-text positions after `<|object_ref_start|>`.
- [ ] Penalize raw-text/open-vocabulary mass at special-token/control positions where the template requires a structural token.
- [ ] Penalize special-token/control mass at raw-text desc positions after `<|object_ref_start|>` where the template requires open-vocabulary text.
- [ ] Validate coordinate-token slots separately from desc/raw-text slots so the gate does not suppress valid coordinate special tokens.
- [ ] Report gating losses and mass diagnostics split by special-token positions, desc/raw-text positions, coordinate positions, separator positions, and terminal positions.
- [ ] Disable the gate for production if it causes malformed decoding, loss instability, coordinate degradation, or early eval collapse in smoke.

Correctness checks for every smoke:

- [ ] Loss decreases over the early training window, or any non-decrease is explained by a deliberately tiny/noisy scope.
- [ ] No NaN or Inf in total loss, CE loss, support loss, balance loss, gating loss, logits diagnostics, gradient norm, or optimizer state.
- [ ] No persistent OOM; isolated OOM probes must be followed by a smaller safe configuration.
- [ ] Token-position alignment passes for assistant span, object-entry spans, desc spans, coordinate spans, separator spans, and terminal spans.
- [ ] Loss-mask correctness is proven by sampled decoded token/mask dumps.
- [ ] Compact-template rendering contains `<|object_ref_start|>`, `<|box_start|>`, coordinate special tokens, expected separators, and no JSON key/closure leakage.
- [ ] Packing correctness is validated when `packing: true`: boundaries, label masks, position IDs, image/example boundaries, and template/trie metadata remain intact.
- [ ] Decoding stability is checked on a small fixed eval subset.
- [ ] Malformed decoding rate is reported by template and parser mode.
- [ ] Rollout sanity is checked: parsed object counts, duplicate rate, early-stop rate, over-continuation rate, and visible examples.
- [ ] Eval mAP and count/FN diagnostics do not collapse early relative to the relevant smoke baseline.
- [ ] GPU memory, throughput, tokens/sec, samples/sec, cache hit rate, and step time are reported.

Numerical-type calibration:

- [ ] Keep global training precision as `bf16` unless smoke evidence requires otherwise.
- [ ] Identify precision-sensitive operators in support/balance CE, log-sum-exp, valid-mass computation, gating losses, denominator normalization, and metric diagnostics.
- [ ] Explicitly cast precision-sensitive computations to `float32` where needed.
- [ ] Compare bf16-only versus targeted-fp32 smoke runs on the same tiny subset.
- [ ] Verify no bf16 operation causes unstable loss, incorrect logits/mass diagnostics, NaN/Inf, degraded training dynamics, or parsing collapse.
- [ ] Record the final dtype policy in the preflight report and resolved production config.

Memory and throughput target:

- [ ] Maximize GPU utilization without OOM.
- [ ] Target approximately 60-70 GB used out of 80 GB per GPU where feasible.
- [ ] If `packing: true` is validated and stable, prefer it for production.
- [ ] If `packing: true` is unavailable or unstable, choose the best validated `packing: false` batch-acceleration strategy and record why packing is rejected.
- [ ] Do not trade correctness, mask integrity, or stable decoding for higher memory utilization.

Expected output:

- [ ] Smoke/preflight report with one row per smoke: command, config path, output directory, GPU assignment, final status, key metrics, and failure/fix notes.
- [ ] Failed cases and fixes, including any code/config changes made after the failed smoke.
- [ ] Recommended final production configuration.
- [ ] Explicit production readiness statement: safe to launch or blocked with reasons.

Production launch block:

- [ ] Do not launch production training until all critical smoke/preflight failures are fixed.
- [ ] Do not launch production training until the final production config is selected, resolved, and linked from the smoke/preflight report.

## Final Deliverable 2: Production Training Launch

Launch condition:

- [ ] Smoke/preflight deliverable is complete.
- [ ] All critical correctness, stability, memory, alignment, mask, rendering, packing, and decoding bugs are fixed.
- [ ] Final production config is recommended by the smoke/preflight report.
- [ ] The owner explicitly approves launch after reviewing the smoke/preflight report.

Production configuration:

- [ ] Train for 4 epochs.
- [ ] Use base checkpoint `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`.
- [ ] Use effective batch size 128.
- [ ] Use `detection_template.id: compact_full`.
- [ ] Enable the validated prompt variant.
- [ ] Enable random object shuffling per epoch.
- [ ] Use hard CE over special tokens, open-vocabulary `desc` tokens, and coordinate tokens wherever the object-entry trie valid child set is singleton or the token is outside `trie_eligible_span`.
- [ ] Enable bidirectional gating loss only if smoke/preflight validated it as stable and beneficial or neutral.
- [ ] Use `packing: true` only if smoke/preflight validated packing correctness and stability.
- [ ] Otherwise use `packing: false` with the best validated alternative batch-acceleration strategy.
- [ ] Keep the final dtype policy from smoke/preflight, including targeted `float32` casts for sensitive operators where needed.
- [ ] Target approximately 60-70 GB used out of 80 GB per GPU where feasible, while avoiding OOM.

Launch procedure:

- [ ] Materialize the final production config under `configs/stage1/` or another checked-in config path selected by the implementation.
- [ ] Run a config-resolution dry run and save the resolved config.
- [ ] Start the production run inside a named `tmux` session.
- [ ] Save the exact launch command in the production report.
- [ ] Save output directory, run name, checkpoint root, logs path, resolved config path, and runtime manifest path.
- [ ] Record GPU allocation and early `nvidia-smi` memory usage.

Monitoring requirements:

- [ ] Monitor the first steps interactively for launch/config errors.
- [ ] Monitor early training loss trend, component losses, gating loss if enabled, gradient norm, NaN/Inf counters, and step time.
- [ ] Monitor decoding/eval sanity at the first configured eval point.
- [ ] Monitor malformed decoding rate, strict parse rate, predicted object count, FN/count diagnostics, duplicate rate, early-stop rate, and over-continuation rate.
- [ ] Monitor GPU memory usage and throughput against the 60-70 GB target.
- [ ] Stop and diagnose immediately if loss, decoding, eval, parser validity, memory, or throughput behavior becomes abnormal.

Expected output:

- [ ] Production launch report with tmux session name, command, config path, output directory, checkpoint/root path, GPU usage, early metrics, and current status.
- [ ] Confirmation that early behavior is healthy, or a stop/diagnosis report if the run is halted.
- [ ] No production run is marked successful until artifacts include resolved config, runtime manifests, logs, early metrics, and checkpoint/output paths.

## Final Definition Of Done

- Raw data parsing, normalization, rendering, tokenization, objective target construction, packing, and evaluation are separated by explicit typed contracts.
- `stage1_json_pretty` and `compact_full` both support sorted SFT, random-order SFT, and Random-Permutation ET-RMP-CE preparation.
- The full-sequence teacher-forced forward process is shared across SFT and ET-RMP-CE.
- Explicit runtime prefix sampling, suffix-row construction, candidate scoring, and archived-config compatibility are absent from the new target architecture.
- Remaining-object state exists only inside the recursive detection CE target builder.
- Loss masks are span-driven and validated.
- Config hierarchy rejects `custom` and other obsolete knobs in latest production configs.
- Framework `template:` remains separate from detection-output `detection_template:`.
- Legacy row-mean equivalence weighting, uniform state weighting, legacy row-mean normalization, and semantic image-balanced normalization are artifact-labeled.
- Packing eligibility and fingerprints are template-aware.
- Evaluation and inference enforce strict expected-template parsing for metric-bearing runs.
- Smoke/preflight report exists and confirms the final setup is safe to launch.
- Production run is launched in `tmux` only after smoke/preflight passes and owner approval is recorded.
- Production launch report records command, config path, output directory, GPU usage, and early metrics.

Approved first implementation slice:

- Typed raw detection containers and source-schema validation.
- Template registry base.
- Canonical `stage1_json_pretty` parity.
- `compact_full` template rendering, spans, and strict parser.
