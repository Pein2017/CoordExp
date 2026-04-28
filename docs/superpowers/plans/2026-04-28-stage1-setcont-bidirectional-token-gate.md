# Stage-1 Set-Continuation Bidirectional Token Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight Stage-1 set-continuation-native bidirectional token gate that penalizes non-coordinate vocabulary mass at coord slots and coord-token mass at non-coord supervised objective slots.

**Architecture:** Reuse the existing coord-vocabulary mass math where possible, but keep set-continuation mask ownership in `src/trainers/stage1_set_continuation`. The gate operates on `objective_label_mask` after the same next-token shift and supervised-suffix crop as candidate scoring, and logs compact gate metrics without enabling ordinary one-sequence Stage-1 mixins.

**Tech Stack:** Python, PyTorch, CoordExp strict dataclass config, Qwen3-VL coord-token tokenizer contract, Stage-1 set-continuation branch scorer, pytest, real-tokenizer smoke configs, and `rtk conda run -n ms python -m pytest` verification.

---

## File Structure

- Modify: `src/config/schema.py`
  - Add `Stage1SetContinuationBidirectionalTokenGateConfig`.
  - Parse it under `Stage1SetContinuationConfig`.
  - Validate nonnegative weights, `temperature > 0`, and `scope == "objective_tokens"`.

- Modify: `src/trainers/stage1_set_continuation/losses.py`
  - Add a small result dataclass for gate loss and metrics.
  - Add a pure helper that receives logits, labels, objective/coord masks, coord ids, and special token ids.
  - Compute gate masks after next-token shift.
  - Return token-summed losses and counts so the trainer can enforce the
    sample-level denominator contract.

- Modify: `src/trainers/stage1_set_continuation/branch_scorer.py`
  - Carry `objective_label_mask` through `TensorBranchScoreInput`.
  - Batch/pad/crop objective masks alongside existing candidate, coord, schema, and structural masks.
  - Return gate metrics from serial and smart-batched scoring paths.

- Modify: `src/trainers/stage1_set_continuation/trainer.py`
  - Include gate loss in the optimized sample loss when enabled.
  - Aggregate compact gate metrics as sample-level token means across scored
    objective branches, then add the gate once per objective-contributing
    sample.
  - Ensure prefix-only and special tokens contribute zero.

- Modify: `src/trainers/stage1_set_continuation/metrics.py`
  - Add compact gate metric keys and aggregation registration.

- Modify: `configs/stage1/set_continuation/production.yaml`
  - Add the explicit gate config under `custom.stage1_set_continuation`.
  - Update metadata from `schemafix` to a new gate-specific artifact name only after smoke passes.

- Modify: `tests/test_stage1_set_continuation_preflight.py`
  - Add real-tokenizer mask inspection tests for coord/text gate assignments.

- Modify: `tests/test_stage1_set_continuation_loss.py`
  - Add pure gate math and shift-alignment tests.

- Modify: `tests/test_stage1_set_continuation_branch_runtime.py`
  - Add retained-vs-smart-batched gate parity tests.

- Modify: `tests/test_stage1_set_continuation_metric_keys.py`
  - Assert gate keys are present only for gate-enabled set-continuation metrics.

- Modify: `tests/test_stage1_set_continuation_config.py`
  - Add strict config parser tests for the gate block.

- Modify: `docs/training/STAGE1_OBJECTIVE.md`
  - Document the objective-span-aligned gate.

- Modify: `docs/training/METRICS.md`
  - Document compact gate metrics.

## Stage 1: Red Tests

- [ ] **Step 1: Add config red tests**

Add tests that parse:

```yaml
bidirectional_token_gate:
  enabled: true
  coord_gate_weight: 0.5
  text_gate_weight: 0.1
  temperature: 1.0
  scope: objective_tokens
```

Expected initial result: fails because the config block is unknown.

- [ ] **Step 2: Add pure math red tests**

Create logits where coord-token rows have high non-coord mass and verify
`loss/coord_gate` increases. Create text rows with high coord-token mass and
verify `loss/text_gate` increases.

Also assert exact values:

```text
p_coord = sum softmax(full_vocab_logits / temperature)[coord_token_ids]
coord_gate = -log(p_coord)
text_gate = -log1p(-p_coord)
```

Add contrast tests proving this is not coord-vocab-normalized candidate CE:

```text
coord_vocab_CE(y) + w * coord_gate
  equals coord-vocab CE when w = 0
  equals full-vocab CE when w = 1
```

Add invariance tests:

```text
coord gate unchanged when probability moves inside coord vocab but p_coord is fixed
text gate unchanged when probability moves inside non-coord vocab but p_coord is fixed
```

Add numerical stability tests with `p_coord` near 0 and near 1. The helper must
return finite/clamped losses and finite gradients.

Expected initial result: fails because the helper does not exist.

- [ ] **Step 3: Add real-tokenizer mask red tests**

Use an existing real JSONL sample and real chat template branch. Assert:

```text
coord labels -> coord gate
schema opener -> text gate for empty prefix
candidate boundary -> text gate
prefix object labels -> neither gate
<|im_end|> / EOS -> neither gate
```

Add an adversarial synthetic branch where:

```text
objective_label_mask is strictly wider than candidate_object_label_mask
schema opener is objective-only
append/final boundary is objective-only
prefix object labels are present but outside objective_label_mask
```

The test must fail if the implementation uses `candidate_object_label_mask`, the
existing candidate score mask, or payload-only masks as the gate owner.

Expected initial result: fails because gate masks are not computed.

- [ ] **Step 4: Add runtime parity red tests**

Extend deterministic branch-runtime fixtures so retained, checkpointed-exact,
and smart-batched exact scoring return the same:

```text
candidate score
coord gate loss
text gate loss
coord gate token count
text gate token count
```

Backpropagate through both paths and compare model parameter gradients. Add a
full-logits versus supervised-suffix parity test for gate losses, mass metrics,
token counts, and gradients. Deliberately put extreme coord mass in physically
returned prefix logits to prove prefix logits are excluded.

Expected initial result: fails because scorer results lack gate fields.

Add a config/runtime guard test:

```text
branch_runtime.mode = smart_batched_exact
ddp_sync.candidate_padding = max_count
```

must fail fast. V1 supports smart batching only with
`ddp_sync.candidate_padding: none`; implementing a separate smart zero-padding
runtime is out of scope for this gate repair.

- [ ] **Step 4b: Add objective-composition red test**

At trainer level, freeze candidate scores and structural losses, enable the
gate, and assert:

```text
optimized_total =
  loss/candidate_balanced
  + coord_gate_weight * sample_coord_gate
  + text_gate_weight * sample_text_gate
  + enabled structural/json terms
```

Also assert `loss/mp_diagnostic` remains diagnostic-only when PEM is disabled.
For a PEM threshold-loss ablation, assert the gate still contributes even when
the PEM margin is satisfied.

- [ ] **Step 4c: Add coord-vocab scope red tests**

Test the pure helper and real tokenizer path:

```text
empty coord ids -> fail
duplicate coord ids -> fail
negative coord ids -> fail
id >= vocab_size -> fail
coord-labeled target not in coord ids -> fail
real tokenizer coord ids count == 1000
real tokenizer coord ids are unique
```

Tiny synthetic math fixtures may use a smaller coord-id set only when the helper
is explicitly told to skip the production-1000 invariant. The real-tokenizer
path must enforce the 1000-id contract.

## Stage 2: Minimal Implementation

- [ ] **Step 5: Implement strict config**

Add a frozen dataclass with:

```python
enabled: bool = False
coord_gate_weight: float = 0.0
text_gate_weight: float = 0.0
temperature: float = 1.0
scope: str = "objective_tokens"
```

Validation:

```text
weights >= 0
temperature > 0
scope == "objective_tokens"
enabled with both weights zero is invalid
```

- [ ] **Step 6: Implement pure gate helper**

The helper must:

```text
shift logits/labels/masks by next-token alignment
exclude label == -100
exclude special/stop token ids
index coord_token_ids from full vocab
compute p_coord with logsumexp
return weighted loss atoms and unweighted diagnostics
```

- [ ] **Step 7: Thread objective masks through scorers**

Carry `objective_label_mask` through serial, checkpointed, and smart-batched
paths. Crop it with the same suffix start used for candidate scoring.

- [ ] **Step 8: Add trainer aggregation**

For one training sample, aggregate all scored candidate branches first:

```text
sample_coord_gate = sum(coord_gate_nll over scored branches) / max(coord_gate_tokens, 1)
sample_text_gate = sum(text_gate_nll over scored branches) / max(text_gate_tokens, 1)
total_loss += coord_gate_weight * sample_coord_gate
total_loss += text_gate_weight * sample_text_gate
```

This prevents samples with more scored candidates from receiving a larger gate
weight solely because they have more branch rows. PEM-threshold ablations still
apply the gate after the PEM margin is satisfied.

## Stage 3: Verification

- [ ] **Step 9: Run targeted tests**

```bash
rtk conda run -n ms python -m pytest \
  /data/CoordExp/tests/test_stage1_set_continuation_config.py \
  /data/CoordExp/tests/test_stage1_set_continuation_loss.py \
  /data/CoordExp/tests/test_stage1_set_continuation_preflight.py \
  /data/CoordExp/tests/test_stage1_set_continuation_branch_runtime.py \
  /data/CoordExp/tests/test_stage1_set_continuation_metric_keys.py \
  -q
```

- [ ] **Step 10: Run broader Stage-1 set-continuation tests**

```bash
rtk conda run -n ms python -m pytest \
  $(rg --files /data/CoordExp/tests | rg 'stage1_set_continuation') \
  /data/CoordExp/tests/test_stage1_metric_key_parity.py \
  -q
```

- [ ] **Step 11: Run static checks**

```bash
rtk conda run -n ms ruff format --check \
  /data/CoordExp/src/trainers/stage1_set_continuation \
  /data/CoordExp/src/config/schema.py \
  /data/CoordExp/tests

rtk conda run -n ms ruff check \
  /data/CoordExp/src/trainers/stage1_set_continuation \
  /data/CoordExp/src/config/schema.py \
  /data/CoordExp/tests
```

If `rtk` is unavailable, use the same `conda run -n ms ...` commands directly.

## Stage 4: Smoke Gate

- [ ] **Step 12: Launch tiny real-data smoke**

Use the real dataset JSONL, real tokenizer, and real chat template. The smoke
must be tiny and procedural; it is not a training-trend claim.

Acceptance:

```text
gate losses finite
gate coord/text token counts nonzero on samples with objects
eval_det_parse_valid_rate >= 0.95 on first smoke eval
eval_det_start_bare_desc_rate <= 0.05
no special stop tokens contribute to gate masks
one-step probe: coord-slot coord mass increases or stays high
one-step probe: text-slot coord mass decreases or stays low
coord vocab provenance recorded: count, uniqueness, min/max, vocab size, hash
```

- [ ] **Step 12b: Dump tiny mask-inspection artifact**

For one empty-prefix branch and one non-empty-prefix branch, write a debug
artifact under `temp/` that contains decoded label tokens for:

```text
coord_gate labels
text_gate labels
prefix-only excluded labels
special excluded labels
first/last logits rows used
coord vocab provenance
```

- [ ] **Step 13: Production decision**

Only after unit tests, static checks, and the smoke gate pass, consider a new
production artifact name such as:

```text
coco1024_sota1332_setcont_candbal_bidirgate
setcont-coco1024-sota1332-candbal-bidirgate
```

Do not relaunch production from this plan until the smoke evidence is reviewed.

## Self-Review Checklist

- [ ] Gate masks are derived from `objective_label_mask`, not
      `candidate_object_label_mask`.
- [ ] Gate reduction is sample-level and not candidate-count weighted.
- [ ] Empty-prefix schema opener is included in text gate.
- [ ] Non-empty-prefix generated prefix objects are excluded.
- [ ] Candidate append/close boundary is included.
- [ ] Coord-vocab scope is validated.
- [ ] Full-logit and supervised-suffix losses match.
- [ ] Smart-batched and retained-graph losses match.
- [ ] Ordinary Stage-1 metric parity is preserved.
