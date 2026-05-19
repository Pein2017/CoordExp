# Recursive Detection Bucketing And Packing Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce compact-full prefix-rollin recursive detection training time by adding safe eval/runtime hygiene, row-atomic length bucketing, and a staged sidecar-safe static packing path without changing architecture, decode, tokenizer/template compatibility, geometry, or objective semantics.

**Architecture:** Keep latest recursive detection as a row-local objective by default. Add a deterministic sequence-length provider and a dedicated latest-detection bucketed sampler path that never rewrites sidecars. Add static packing only behind a new sidecar-packing adapter that rewrites every offset-bearing target, preserves per-original-sample normalization, and proves forward isolation before production use. Keep padding-free packed runtime as a later phase after static-packing parity tests pass.

**Tech Stack:** Python, PyTorch, dedicated latest-detection sampler helpers, ms-swift template encoding, YAML latest-detection configs, pytest under `conda run -n ms`.

---

Date: 2026-05-09

Spec: `docs/superpowers/specs/2026-05-09-recursive-detection-bucketing-packing-optimization-design.md`

Status: planning and audit only. Do not implement until the user explicitly approves implementation after audit findings are resolved.

## Hard Guardrails

| Guardrail | Requirement |
|---|---|
| Architecture | Do not edit upstream Qwen/HF model files and do not add a detection head. |
| Decode | Do not change the main autoregressive generate/decode path. |
| Objective | Preserve `prefix_rollin_et_rmp_ce` semantics, target weights, EOS trust, type gate, and trie multi-positive targets. |
| Geometry | Preserve image-root safety, object order, coord-token alignment, and `do_resize=false`. |
| Training quality | Do not accept a speedup if matched-scope training/eval quality regresses. Throughput improvements require sidecar/loss parity plus tiny or health-eval metric parity before production launch. |
| Config | Prefer typed config/schema/runtime artifacts over ad hoc CLI flags. |
| Packing | Do not enable recursive detection packing until sidecar offset rewrite and parity tests exist. |
| Logits | Do not use global `logits_to_keep` for recursive CE target optimization. |
| Worktree | Implement only in `/data/CoordExp/.worktrees/recursive-detection-bucketing-packing` on branch `codex/recursive-detection-bucketing-packing`. |

## Planned File Map

| Path | Role |
|---|---|
| `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2_a3_bsz8_ebs128.yaml` | A3 runtime hygiene: explicit eval batch size and cadence. |
| `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2_a4_eos_bsz8_ebs128.yaml` | A4 runtime hygiene: explicit eval batch size and cadence. |
| `src/config/schema.py` | Validate that `group_by_length` / `length_column_name` remain TrainArguments pass-through keys and preserve packing guardrails. |
| `src/config/loader.py` | Ensure training args and stripped packing keys do not hide bucketing/runtime fields. Do not add bucketing keys to stripped packing-only keys. |
| `src/sft.py` | Wire latest-detection length bucketing, effective-runtime artifact truth, and future packing owner handoff. |
| `src/detection/dataset.py` | Add deterministic latest-detection length provider and sidecar-safe helper methods. |
| `src/detection/length_bucketing.py` | New dedicated latest-detection length provider, sampler, distributed sampler, and sampler provenance objects. |
| `src/detection/packing.py` | Add sidecar-packing eligibility/contract objects and fingerprints for future static packing. |
| `src/data_collators/batch_extras_collator.py` | Preserve recursive sidecars through ordinary bucketed batches; later accept packed sidecar contract. |
| `src/data_collators/enrichers.py` | Keep packed recursive sidecars rejected unless contract marker is present. |
| `src/detection/loss.py` | Later support packed original-sample grouping while preserving unpacked loss identity. |
| `src/trainers/metrics/recursive_detection.py` | Later report original sample counts separately from packed row counts. |
| `tests/test_prefix_rollin_ablation_launch_smoke.py` | Config materialization and runtime hygiene tests. |
| `tests/test_latest_detection_length_bucketing.py` | New row-atomic bucketing, sequence-length, spy/no-encode, and DDP sampler tests. |
| `tests/test_recursive_detection_sidecar_packing_offsets.py` | New static-packing sidecar offset and loss-parity tests. |
| `tests/test_batch_extras_contract.py` | Collator packed/unpacked recursive sidecar contract tests. |
| `tests/test_recursive_detection_ce_loss_adapter.py` | Packed/unpacked recursive CE parity tests. |
| `tests/test_recursive_detection_ce_trainer_mixin.py` | Trainer metric and sidecar validation tests. |

## Task 0: Planning Preflight

**Files:**

- Read: `docs/superpowers/specs/2026-05-09-recursive-detection-bucketing-packing-optimization-design.md`
- Read: this plan
- Read: `src/detection/dataset.py`
- Read: `src/detection/loss.py`
- Read: `src/data_collators/enrichers.py`
- Read: `src/trainers/metrics/recursive_detection.py`

- [ ] **Step 1: Confirm branch and dirty state**

Run:

```bash
git -C /data/CoordExp/.worktrees/recursive-detection-bucketing-packing status --short
git -C /data/CoordExp/.worktrees/recursive-detection-bucketing-packing branch --show-current
```

Expected: current branch is `codex/recursive-detection-bucketing-packing`. The two A3/A4 bsz8 config edits may already be dirty as a user-authorized pre-implementation config delta. The first implementation commit must isolate those config edits from code/runtime changes. No other non-planning dirty files should be present before implementation begins.

- [ ] **Step 2: Use Serena for Python symbol inspection**

Inspect these symbols before editing:

```text
src/detection/dataset.py::DetectionTrainingDataset
src/detection/loss.py::compute_recursive_detection_ce_batch_loss
src/trainers/metrics/recursive_detection.py::RecursiveDetectionCEMixin
src/data_collators/enrichers.py::RecursiveDetectionTargetsEnricher
src/detection/packing.py::assess_packing_eligibility
```

Expected: write down which functions own length, collation, sidecar validation, loss normalization, and packing rejection.

## Task 1: Config And Eval Runtime Hygiene

**Files:**

- Modify: `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2_a3_bsz8_ebs128.yaml`
- Modify: `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2_a4_eos_bsz8_ebs128.yaml`
- Modify: `tests/test_prefix_rollin_ablation_launch_smoke.py`
- Modify: `src/sft.py`
- Modify: `src/bootstrap/experiment_manifest.py`

- [ ] **Step 1: Write config materialization tests**

Add tests that load both A3/A4 bsz8 configs and assert:

```python
assert cfg.training["per_device_train_batch_size"] == 8
assert cfg.training["per_device_eval_batch_size"] == 8
assert cfg.training["effective_batch_size"] == 128
assert cfg.training["eval_steps"] == 600
assert cfg.training["packing"] is False
assert cfg.training["eval_packing"] is False
assert cfg.packing.static_packing is False
assert cfg.packing.padding_free_packed is False
```

- [ ] **Step 2: Add run-artifact provenance tests**

Extend the same test owner so the materialized runtime path asserts:

```python
assert effective["runtime"]["eval_strategy"] == cfg.training["eval_strategy"]
assert effective["runtime"]["eval_steps"] == 600
assert effective["runtime"]["per_device_eval_batch_size"] == 8
assert effective["runtime"]["packing"]["enabled"] is False
assert effective["runtime"]["packing"]["eval_packing"] is False
assert experiment_manifest["runtime_summary"]["eval_strategy"] == cfg.training["eval_strategy"]
assert experiment_manifest["runtime_summary"]["eval_steps"] == 600
assert experiment_manifest["runtime_summary"]["per_device_eval_batch_size"] == 8
```

The test must exercise the real `_parse_packing_config()` path instead of
manually constructing `PackingRuntimeConfig(enabled=False, eval_packing=False)`.

- [ ] **Step 3: Run the tests and verify the intended failure**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_prefix_rollin_ablation_launch_smoke.py -q
```

Expected before runtime patch: config assertions may already pass because the
user-authorized config delta is present, but artifact assertions fail because
`eval_steps` is missing from effective runtime and disabled packing may still
serialize `eval_packing: true`.

- [ ] **Step 4: Isolate or apply A3/A4 config delta**

Add:

```yaml
training:
  per_device_eval_batch_size: 8
  eval_steps: 600
```

Expected: no objective, tokenizer, data, or packing fields change. If these
lines are already present, leave them as the first isolated config delta for the
implementation branch.

- [ ] **Step 5: Fix effective runtime packing and eval provenance**

Patch `src/sft.py` so disabled packing artifacts record:

```python
"packing": {
    "enabled": False,
    "eval_packing": False,
}
```

Also record:

```python
"eval_strategy": train_args.eval_strategy,
"eval_steps": train_args.eval_steps,
"per_device_eval_batch_size": train_args.per_device_eval_batch_size,
```

Patch `src/bootstrap/experiment_manifest.py` so the runtime summary mirrors the
same fields. Expected: artifact truth matches the effective runtime, not merely
schema defaults.

- [ ] **Step 6: Add health-eval provenance guardrails**

Add a health overlay only after implementation approval. For latest detection
configs, use:

```yaml
debug:
  enabled: true
  val_sample_limit: 64
experiment:
  claim_scope: health-val64-not-full-val
```

Do not set `debug.train_sample_limit` for any real training-health launch.
Add artifact tests that assert:

```python
assert train_data_provenance["sample_limit"] is None
assert eval_data_provenance["sample_limit"] == 64
assert resolved["resolved"]["experiment"]["claim_scope"] == "health-val64-not-full-val"
```

## Task 2: Row-Atomic Length Provider

**Files:**

- Modify: `src/detection/dataset.py`
- Create: `src/detection/length_bucketing.py`
- Create: `tests/test_latest_detection_length_bucketing.py`

- [ ] **Step 1: Write the failing length-invariance tests**

Create tests for a lower-level length provider that builds one compact-full
prefix-rollin sample with several objects, forces multiple `K` values, and
asserts:

```python
lengths = {
    provider.length_for_row(base_idx=0, forced_rollin_k=k)
    for k in range(object_count + 1)
}
assert len(lengths) == 1
```

Also assert random object order does not change length:

```python
dataset.set_epoch(0)
length_epoch_0 = provider.length_for_row(base_idx=0, epoch=0)
dataset.set_epoch(7)
length_epoch_7 = provider.length_for_row(base_idx=0, epoch=7)
assert length_epoch_0 == length_epoch_7
```

Use either the real tokenizer in an optional/gated test or a
BPE-boundary-sensitive fake tokenizer with unequal description tokenization.
The existing simple character tokenizer alone is not enough evidence.

- [ ] **Step 2: Run the tests and verify the intended failure**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_latest_detection_length_bucketing.py -q
```

Expected before implementation: no dedicated latest-detection length provider exists.

- [ ] **Step 3: Add a typed length provider**

Implement a small object-oriented owner in `src/detection/length_bucketing.py`
with this shape:

```python
@dataclass(frozen=True)
class LatestDetectionLengthProvider:
    dataset: DetectionTrainingDataset

    def length_for_row(
        self,
        *,
        base_idx: int,
        epoch: int | None = None,
        forced_rollin_k: int | None = None,
    ) -> int:
        return self.dataset.compute_encoded_length(
            base_idx=base_idx,
            epoch=epoch,
            forced_rollin_k=forced_rollin_k,
        )
```

`DetectionTrainingDataset.compute_encoded_length` may share internal
sample-building logic with `__getitem__`, but must not duplicate target
construction in a way that can drift. The production sampler must call
`length_for_row` with `forced_rollin_k=None`; `forced_rollin_k` exists to make
the K-invariance contract testable.

- [ ] **Step 4: Verify no sidecar mutation**

Add a test that compares `dataset[0]["recursive_detection_targets"]` before and
after length lookup:

```python
before = dataset[0]["recursive_detection_targets"]
_ = provider.length_for_row(base_idx=0)
after = dataset[0]["recursive_detection_targets"]
assert before == after
```

Expected: length lookup is observational and deterministic.

- [ ] **Step 5: Define length-cache provenance**

Keep v1 length precompute run-local only. Add tests and artifact assertions for:

```python
assert effective["runtime"]["dataloader"]["length_source"] == "latest_detection_run_local"
assert effective["runtime"]["dataloader"]["length_cache_policy"] == "run_local_only"
assert effective["runtime"]["dataloader"]["length_count"] == len(train_dataset)
```

Expected: no persisted length cache is created or reused across runs.

## Task 3: Row-Atomic Length Bucketing Sampler

**Files:**

- Modify: `src/config/schema.py`
- Modify: `src/config/loader.py`
- Modify: `src/sft.py`
- Create: `src/detection/length_bucketing.py`
- Modify: `tests/test_latest_detection_length_bucketing.py`
- Modify: `tests/test_prefix_rollin_ablation_launch_smoke.py`

- [ ] **Step 1: Write config tests for the bucketing switch**

Add a latest-detection config test for:

```yaml
training:
  group_by_length: true
  length_column_name: length
```

Expected materialized runtime through existing TrainArguments pass-through:

```python
assert train_args.group_by_length is True
assert train_args.length_column_name == "length"
```

Also assert these keys are not added to loader stripped packing-only keys:

```python
assert "group_by_length" not in stripped_packing_keys
assert "length_column_name" not in stripped_packing_keys
```

- [ ] **Step 2: Write sampler behavior tests**

Create a synthetic dataset with lengths `[100, 900, 110, 880]`, batch size `2`,
and assert the bucketed sampler forms lower-padding batches than plain shuffled
order under the same seed.

Expected invariant:

```python
plain_padding = sum(max(batch) * len(batch) - sum(batch) for batch in plain_batches)
bucket_padding = sum(max(batch) * len(batch) - sum(batch) for batch in bucket_batches)
assert bucket_padding < plain_padding
```

Add a spy dataset/template test that fails if sampler construction calls
`DetectionTrainingDataset.__getitem__` or encodes rows through the normal sample
path:

```python
dataset.getitem_call_count = 0
sampler = build_latest_detection_length_grouped_sampler(dataset, batch_size=2, seed=17)
assert dataset.getitem_call_count == 0
assert sampler.lengths == [100, 900, 110, 880]
```

- [ ] **Step 3: Run tests and verify the intended failure**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_latest_detection_length_bucketing.py tests/test_prefix_rollin_ablation_launch_smoke.py -q
```

Expected before implementation: config/sampler wiring is missing or does not use the dataset length provider.

- [ ] **Step 4: Add DDP sampler behavior tests**

Create non-training tests for `world_size=4` and non-divisible dataset lengths.
Assert:

```python
assert all(len(indices) == len(rank_indices[0]) for indices in rank_indices)
assert all(indices for indices in rank_indices)
assert same_seed_epoch0_rank0 == repeated_same_seed_epoch0_rank0
assert epoch1_rank0 != same_seed_epoch0_rank0
assert no_unexpected_duplicate_drop_behavior(rank_indices, drop_last=False)
```

Also test `drop_last=True` explicitly. Expected: DDP behavior is deterministic,
non-empty per rank, and documented before any training preflight.

- [ ] **Step 5: Wire bucketing without changing sidecars**

Use the dedicated latest-detection sampler path by default. Stock HuggingFace
`group_by_length` is allowed only if an implementation test proves it receives
the explicit length list without calling `DetectionTrainingDataset.__getitem__`
during sampler construction.

Required runtime behavior:

```text
bucketed batch item = original dataset item
batch collation = existing unpacked collator
recursive sidecars = unchanged row-local sidecars
labels/attention/image fields = unchanged
```

- [ ] **Step 6: Verify recursive sidecar alignment after bucketed collation**

Add a collated-batch test:

```python
for row, targets in enumerate(batch["recursive_detection_targets"]):
    for target in targets.token_targets:
        assert batch["input_ids"][row, target.position].item() == target.teacher_token_id
        assert batch["labels"][row, target.position].item() == target.teacher_token_id
        assert batch["attention_mask"][row, target.position].item() == 1
```

Expected: bucketing changes only which samples share a padded batch.

- [ ] **Step 7: Record sampler provenance**

Patch runtime artifacts and experiment manifest summary to include:

```python
"dataloader": {
    "group_by_length": True,
    "length_column_name": "length",
    "sampler": "LatestDetectionDistributedLengthGroupedSampler",
    "length_source": "latest_detection_run_local",
    "length_cache_policy": "run_local_only",
    "sampler_seed": train_args.seed,
    "epoch_policy": "sampler_epoch_and_dataset_epoch",
    "drop_last": train_args.dataloader_drop_last,
    "world_size": world_size,
}
```

- [ ] **Step 8: Add quality-preservation gate**

Add a matched-seed A/B smoke requirement before production use:

```text
baseline: group_by_length=false
candidate: group_by_length=true
same train/val sample scope
same model, seed, effective batch, objective, tokenizer, image roots
```

Acceptance is not based on speed alone. The candidate must preserve:

```text
loss/recursive_detection_ce
recursive_detection_ce/type_gate_loss
recursive_detection_ce/eos_trust_weight
recursive_detection_ce/target_mix/trie_multi_positive_fraction
recursive_detection_ce/entry/valid_child_entropy
eval_loss or eval recursive CE on the same health/full-val scope
```

Expected: no metric degradation beyond stochastic tiny-smoke tolerance, and any
throughput claim is reported with the exact sample/eval scope.

## Task 4: Sidecar-Safe Static Packing Contract

**Files:**

- Modify: `src/detection/packing.py`
- Modify: `src/data_collators/enrichers.py`
- Modify: `src/data_collators/batch_extras_collator.py`
- Modify: `src/detection/loss.py`
- Create: `tests/test_recursive_detection_sidecar_packing_offsets.py`
- Modify: `tests/test_batch_extras_contract.py`

- [ ] **Step 1: Preserve the current fail-fast guard**

Before adding packing support, add or keep this negative test:

```python
with pytest.raises(ValueError, match="recursive_detection_targets.*packing"):
    RecursiveDetectionTargetsEnricher()(packed_batch_without_contract)
```

Expected: recursive detection sidecars are still rejected for arbitrary packed batches.

- [ ] **Step 2: Write sidecar offset rewrite tests**

Create two synthetic unpacked examples with target positions `[3, 4]` and
`[2, 5]`. Pack both originals into one packed row with offsets `[0, 10]`.
Assert the public sidecar list is row-aligned, not original-sample-aligned:

```python
assert packed.packed_row_count == 1
assert packed.original_sample_count == 2
assert len(packed.recursive_detection_targets) == 1
row_targets = packed.recursive_detection_targets[0]
assert [target.position for target in row_targets.token_targets] == [3, 4, 12, 15]
assert packed.original_sample_groups[0].target_indices == (0, 1)
assert packed.original_sample_groups[1].target_indices == (2, 3)
```

Also assert every `LossAtom.token_positions` and every carried debug span token
position receives the same offset.

- [ ] **Step 3: Run tests and verify the intended failure**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_recursive_detection_sidecar_packing_offsets.py tests/test_batch_extras_contract.py -q
```

Expected before implementation: no sidecar packing adapter exists.

- [ ] **Step 4: Add a typed packing adapter**

Add an object-oriented adapter in `src/detection/packing.py`:

```python
@dataclass(frozen=True)
class RecursiveDetectionSidecarPackingAdapter:
    contract_version: str = "recursive_detection_sidecar_packing_v1"

    def pack(self, samples: Sequence[Mapping[str, object]]) -> PackedRecursiveDetectionBatch:
        return self._pack_validated_samples(samples)
```

The adapter must return a typed object containing:

```text
input_ids
attention_mask
labels
recursive_detection_targets  # one combined sidecar per packed row
original_sample_groups
original_sample_count
packed_row_count
pack_offsets
debug_spans_by_original
debug_spans_by_packed_row
contract_version
```

- [ ] **Step 5: Validate the contract marker, not just its string value**

Modify collator/enricher logic so packed recursive detection batches are
accepted only when `contract_version == "recursive_detection_sidecar_packing_v1"`.
The validator must also assert:

```python
assert pack_offsets == tuple(sorted(pack_offsets))
assert all(0 <= target.position < packed_sequence_length for target in row_targets.token_targets)
assert all(0 <= pos < packed_sequence_length for atom in row_targets.loss_atoms for pos in atom.token_positions)
assert all_labels_and_input_ids_match_shifted_targets(batch, row_targets)
assert original_groups_cover_targets_without_overlap(original_sample_groups, row_targets)
assert original_sample_count == sum(group.sample_count for group in packed_rows)
```

Expected: arbitrary packed recursive sidecars still fail fast.

- [ ] **Step 6: Add forward-isolation gate**

Before production static packing can be enabled, add tests or runtime guards for
segment-aware attention/position isolation:

```python
assert packed_batch.segment_ids is not None
assert sample_2_target_positions_cannot_attend_to_sample_1_tokens(packed_batch)
```

If the implementation only supports ordinary concatenation with cross-sample
causal attention, mark the mode as `diagnostic_static_packing_cross_context` and
keep production recursive detection packing disabled.

## Task 5: Packed/Unpacked Loss Parity

**Files:**

- Modify: `src/detection/loss.py`
- Modify: `src/trainers/metrics/recursive_detection.py`
- Modify: `tests/test_recursive_detection_ce_loss_adapter.py`
- Modify: `tests/test_recursive_detection_ce_trainer_mixin.py`
- Modify: `tests/test_recursive_detection_sidecar_packing_offsets.py`

- [ ] **Step 1: Write synthetic parity tests**

Build two original samples, compute recursive CE loss unpacked, then compute the
same loss after static packing with rewritten offsets. Use logits that give the
same target-token probabilities at the shifted positions.

Expected:

```python
assert packed_loss.item() == pytest.approx(unpacked_loss.item(), rel=1e-6)
```

Use unequal originals so the test catches merged-row normalization errors:

```python
original_a = build_targets(object_count=1, include_boundary=True, include_type_gate=True)
original_b = build_targets(object_count=5, include_boundary=False, include_type_gate=True)
unpacked_loss = (loss_a + loss_b) / 2.0
packed_loss = compute_loss_from_packed_row(two_original_groups_batch)
assert packed_loss.item() == pytest.approx(unpacked_loss.item(), rel=1e-6)
```

- [ ] **Step 2: Verify EOS/type-gate/trie preservation**

Add assertions that packed metrics equal unpacked metrics for:

```text
recursive_detection_ce/eos_trust_weight
recursive_detection_ce/type_gate_loss
recursive_detection_ce/target_mix/trie_multi_positive_fraction
recursive_detection_ce/entry/valid_child_entropy
detection_sequence/objective/recursive_detection_ce/loss_per_sample
detection_sequence/objective/recursive_detection_ce/batch_size
```

- [ ] **Step 3: Run tests and verify the intended failure**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_recursive_detection_sidecar_packing_offsets.py -q
```

Expected before implementation: loss code assumes one sidecar object per batch row and cannot normalize packed original samples.

- [ ] **Step 4: Preserve per-original-sample normalization**

Update loss code so packed recursive targets retain original-sample grouping.
Metrics must report both:

```text
recursive_detection_ce/batch_size_original_samples
recursive_detection_ce/batch_size_packed_rows
recursive_detection_ce/target_mix/targets_per_original_sample
```

Expected: optimizer signal matches unpacked per-sample semantics.

- [ ] **Step 5: Record effective-batch units for packing**

When static packing is enabled, keep existing `effective_batch_size` and
`actual_global_effective_batch_size` in packed-row units, and add:

```python
assert effective["runtime"]["effective_batch_unit"] == "packed_rows"
assert "actual_global_effective_original_samples_estimate" in effective["runtime"]
assert "observed_original_samples_per_optimizer_step" in effective["runtime"]
assert "packed_row_count" in effective["runtime"]
assert "original_sample_count" in effective["runtime"]
```

Mirror these fields into `experiment_manifest.json.runtime_summary`.

## Task 6: Tiny Smoke And Launch Gates

**Files:**

- Modify: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_prefix_rollin_adapter_tiny.yaml`
  only if the existing tiny config cannot express the new optimization mode.
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_prefix_rollin_eval_bsz8_memory.yaml`
- Modify: `tests/test_recursive_detection_ce_sft_wiring.py` only if no existing
  SFT wiring smoke verifies the new optimization mode in resolved runtime
  artifacts.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_prefix_rollin_ablation_launch_smoke.py \
  tests/test_latest_detection_length_bucketing.py \
  tests/test_batch_extras_contract.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_recursive_detection_ce_trainer_mixin.py \
  tests/test_recursive_detection_sidecar_packing_offsets.py \
  -q
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run non-disruptive launch gate**

Before any smoke or relaunch, run:

```bash
git -C /data/CoordExp/.worktrees/recursive-detection-bucketing-packing status --short
git -C /data/CoordExp/.worktrees/recursive-detection-bucketing-packing branch --show-current
tmux list-sessions -F '#{session_name}: #{session_windows} windows'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

Expected: branch is `codex/recursive-detection-bucketing-packing`; no unrelated
dirty code is present; target tmux session names are absent or explicitly idle;
target GPUs are free. Do not use `tmux kill-server`, broad `pkill`, broad
`rm -rf /data/CoordExp/outputs`, or cleanup of existing A3/A4 artifacts.

- [ ] **Step 3: Run tiny training smoke**

Use a tiny latest recursive detection config with:

```yaml
debug:
  enabled: true
  train_sample_limit: 8
  val_sample_limit: 8
training:
  max_steps: 2
  eval_steps: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
```

Expected: no traceback, no NaN/Inf, recursive CE/type-gate/EOS/trie metrics present, artifacts record the optimization mode.

- [ ] **Step 4: Run eval-bsz8 memory smoke**

Create `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_prefix_rollin_eval_bsz8_memory.yaml`
extending the A3 bsz8 config with:

```yaml
training:
  artifact_subdir: smoke_compact_full_prefix_rollin_eval_bsz8_memory
  run_name: smoke-compact-full-prefix-rollin-eval-bsz8-memory
  max_steps: 1
  eval_strategy: steps
  eval_steps: 1
  save_strategy: "no"
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 8
  effective_batch_size: 4
debug:
  enabled: true
  train_sample_limit: 8
  val_sample_limit: 32
experiment:
  claim_scope: smoke-eval-bsz8-not-throughput
```

Run the one-GPU memory smoke:

```bash
cd /data/CoordExp/.worktrees/recursive-detection-bucketing-packing
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=8 \
  conda run --no-capture-output -n ms torchrun \
  --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29641 \
  -m src.sft \
  --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_prefix_rollin_eval_bsz8_memory.yaml
```

Expected: no OOM, no NaN/Inf, eval recursive CE metrics present,
`effective_runtime.json.runtime.per_device_eval_batch_size == 8`, and
`eval_data_provenance.json.sample_limit == 32`.

- [ ] **Step 5: Run intended-shape DDP eval-bsz8 preflight**

After the one-GPU memory smoke passes, run the same smoke on the intended
four-GPU shape with a unique session and port:

```bash
tmux new-session -d -s coordexp_smoke_prefix_rollin_eval_bsz8_4gpu \
  'cd /data/CoordExp/.worktrees/recursive-detection-bucketing-packing && \
   CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. OMP_NUM_THREADS=8 \
   conda run --no-capture-output -n ms torchrun \
   --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29642 \
   -m src.sft \
   --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_prefix_rollin_eval_bsz8_memory.yaml'
```

Expected: no OOM, no NaN/Inf, every rank participates, eval metrics are present,
and the run is not treated as throughput evidence.

- [ ] **Step 6: Run DDP preflight before production relaunch**

Run 20-50 optimizer steps with the intended DDP shape before replacing A3/A4.

Expected:

```text
no OOM
no NaN/Inf
no malformed sidecar spans
no artifact/config mismatch
iteration speed improves or padding waste decreases under bucketing
matched-scope training/eval metrics do not regress versus non-bucketed baseline
```

## Audit Checklist Before Implementation Approval

- [x] Dataset/length assumptions audited.
- [x] Collator/sidecar/packing offset contract audited.
- [x] Loss normalization and metric semantics audited.
- [x] Config/runtime/artifact provenance audited.
- [x] Eval cadence/eval batch-size smoke risk audited.
- [x] All P0/P1 audit issues resolved in this plan and the design spec.

Resolution notes:

- Dedicated latest-detection bucketing replaces the unsafe default reliance on
  stock HF `group_by_length` for this torch map-style dataset.
- Static packing now requires one combined row-local sidecar per packed row plus
  `original_sample_groups`, not one sidecar per original sample at batch level.
- Packed loss parity must use unequal original samples and average original
  sample normalized losses.
- Eval-bsz8 is gated by one-GPU and intended-shape DDP memory smokes before A3/A4
  relaunch.
- Runtime artifacts must record eval cadence, eval batch size, dataloader mode,
  length source, disabled-packing truth, and effective-batch units.
- Efficiency claims must include a matched-scope quality check; faster is not
  acceptable if recursive CE, type-gate, EOS/trie health, or eval metrics regress.
