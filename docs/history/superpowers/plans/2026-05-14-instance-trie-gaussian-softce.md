# Instance-Trie Gaussian SoftCE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an instance-aware multi-positive Gaussian coordinate softCE objective for compact recursive detection, using the existing entry trie to preserve object-instance binding while applying bbox-size-aware smooth coordinate targets.

**Architecture:** Latest recursive detection remains the owner. Schema/control tokens use hard CE, description/entry-choice ambiguity uses existing ET-RMP support/balance, and coordinate tokens use pure softCE against a Gaussian mixture over active semantic-branch remaining candidate instances. The initial objective uses one teacher-forced forward pass: each coordinate slot computes a soft posterior over those candidates from previous teacher-forced coordinates, using unnormalized Gaussian mismatch energy for prefix compatibility, then mixes normalized bbox-size-aware Gaussian peaks for the current slot. This is a one-pass posterior-predictive training objective, not a duplicated latent-branch forward pass.

**Tech Stack:** Python dataclasses, PyTorch dense coordinate-token CE, strict latest-schema YAML configs, compact-full recursive detection sidecars, pytest under `conda run -n ms`, and Superpowers subagent-driven implementation in the isolated worktree.

---

Date: 2026-05-14

Status: implementation and smoke/preflight verification completed on the isolated worktree
`/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/instance-trie-gaussian-softce`
on branch `codex/instance-trie-gaussian-softce`. The objective is implemented
and has passed the focused test suite, no-training target-shape audit, tiny
smoke, and 8-GPU DDP preflight. It is not merged, stable, or current
production behavior yet; real production launch remains gated by explicit user
approval.

Design note: `docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md`

Branch/worktree:

```text
branch: codex/instance-trie-gaussian-softce
worktree: /data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/instance-trie-gaussian-softce
```

## Guardrails

| Guardrail | Requirement |
|---|---|
| Implementation status | Implementation is active only on branch `codex/instance-trie-gaussian-softce` in the isolated worktree; it is not merged, stable, or production-launched. |
| Training comparison | New config must extend `compact_full_support2.yaml` and keep model, data, optimizer, batch semantics, prompt, template, token rows, cache/packing, and non-coordinate behavior unchanged. |
| Template | Use `compact_full`, `coord_token`, `xyxy`, and the same dataset config as the previous A2/A5/A6 surface. |
| Initial scope | The first implementation and configs target the compact-full support2 latest recursive detection surface. Prefix-rollin support needs explicit tests before use. |
| Loss owner | Latest recursive CE owns this loss. Do not route through legacy `custom.coord_soft_ce_w1.*`. |
| Token groups | Schema/control hard CE; desc/entry-choice support+balance; coord pure softCE. |
| Coordinate candidate unit | Coordinates use the active desc/object-ref branch's object-instance candidate group, not recursive support weights, all image objects, or independent x/y coordinate pools. |
| Candidate ownership | Candidate groups must come from recursive target construction sidecars, never from decoded-string matching. They must contain active-branch remaining, not-yet-emitted instances only, excluding already emitted and roll-in-prefix objects. |
| Candidate priors | This objective uses uniform priors over valid candidate instances; non-uniform priors are future ablations. |
| Target shape | Use bbox-axis Gaussian peaks with `sigma_x^2 = width + 1` and `sigma_y^2 = height + 1`. |
| Prefix compatibility | Use unnormalized Gaussian mismatch energy for posterior prefix weighting; only current-slot coordinate target distributions are normalized over bins. |
| Coordinate domain | Resolve the coordinate-token vocabulary from token-row geometry. Current compact-full configs use 1000 value bins, `<|coord_0|>` through `<|coord_999|>`; do not hard-code a 1001-bin `[0,1000]` assumption. |
| No extra knobs | Do not expose or accept `tau`, `tau_source`, `weighting`, `replace_coord_hard_ce`, `apply_to_multi_positive`, sigma multiplier, truncation radius, min/max clipping, or IoU/CIoU switches for this config. |
| Structural legality | Mask impossible coordinate replacements such as `x1 >= x2` and `y2 <= y1`; do not use this mask as a hand-tuned smoothing radius. |
| Weighting | Coordinate positions use pure softCE. Recursive support/balance weights still apply to desc/entry-choice positions. |
| Log-probability surface | Coordinate softCE must use full-vocabulary `log_softmax` before indexing coordinate-token ids, so non-coordinate token leakage remains penalized. |
| Type gate | Preserve existing type-gate auxiliary behavior. "Pure coordinate softCE" means no recursive support/balance weighting on coordinates, not removal of existing type supervision. |
| Failure policy | Missing semantic-branch candidate metadata is a hard error for `instance_trie_gaussian`; do not fall back to selected-instance Gaussian or hard CE. |
| Inference/eval | Inference, decoding, confidence post-processing, and evaluator surfaces remain unchanged for this objective. Do not introduce oracle candidate constraints or reranking. |
| Evidence | Unit tests and smoke tests must label scope clearly; no tiny or smoke metric is full validation. |

## File Map

| Path | Role |
|---|---|
| `docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md` | Draft design and formula. |
| `docs/training/STAGE1_OBJECTIVE.md` | Contains current provenance for the active implementation draft, historical A5/A6 negative-result configs, and production-validation gates. |
| `src/config/schema.py` | Add strict schema acceptance for `target_distribution: instance_trie_gaussian`; reject obsolete tau/sigma/truncation knobs for this target. |
| `src/detection/coord_soft_targets.py` | Add Gaussian target builder, soft posterior weighting from teacher prefix, posterior diagnostics, and pure coordinate softCE helper. Keep old IoU/CIoU-Gibbs helpers for negative-result provenance unless explicitly removed later. |
| `src/detection/objective.py` | Audit existing sidecars. If `coord_soft_targets` is exact-prefix filtered, add candidate-only `coord_instance_candidates` for the active semantic-branch candidate group and keep legacy exact-prefix metadata intact. |
| `src/detection/loss.py` | Route coordinate positions to pure Gaussian softCE; keep desc/entry-choice positions on support/balance and schema/control positions on hard CE. |
| `src/detection/runtime.py` | Resolve the runtime coordinate token range for `instance_trie_gaussian` from token-row geometry. |
| `src/detection/__init__.py` | Export new dataclasses/helpers needed by tests. |
| `src/trainers/metrics/recursive_detection.py` | Log target distribution and coordinate diagnostics without assuming `tau` exists. |
| `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_gaussian_softce_a5.yaml` | Production-style `A5-instance-trie-gaussian` ablation config, launch only after approval and smoke gates. |
| `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_tiny.yaml` | Tiny smoke config. |
| `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_ddp8_preflight.yaml` | 8-GPU preflight config. |
| `tests/test_instance_trie_gaussian_coord_softce.py` | Unit tests for target math, uniform priors, soft posterior weighting, union-box non-reward, and pure softCE weighting. |
| `tests/test_recursive_detection_ce_target_builder.py` | Sidecar invariants: semantic-branch candidate ownership, no all-image candidates, and near-shared coordinate ambiguity carry-forward. |
| `tests/test_recursive_detection_ce_loss_adapter.py` | End-to-end loss replacement and metric checks. |
| `tests/test_latest_training_config_contract.py` | Schema/runtime/config parsing and obsolete-knob rejection. |
| `scripts/diagnostics/audit_instance_trie_gaussian_targets.py` | Reproducible no-training target-shape audit command. |
| `tests/test_instance_trie_gaussian_config_diff.py` | Structured resolved-config diff whitelist for fair-comparison configs. |
| `tests/test_instance_trie_gaussian_target_shape_audit.py` | Target-shape audit artifact schema and synthetic-fixture checks. |

## Naming Convention

Use clear, version-free names for all new code/config/docs introduced by this
work:

| Surface | Canonical name |
|---|---|
| Human-facing objective | Instance-Trie Gaussian SoftCE |
| Config `target_distribution` | `instance_trie_gaussian` |
| Production config suffix | `instance_trie_gaussian_softce_a5` |
| Ablation label | `A5-instance-trie-gaussian` |
| Numeric metric flag | `recursive_detection_ce/coord_soft_ce/is_instance_trie_gaussian` |

`A5` is retained only as the ablation-lineage label for comparison with
historical A5/A6 runs. It is not a code/API version and must not appear in
helper names, registry names, `target_distribution` values, or metric
namespaces except as run/config labeling.

Do not introduce new `*_v0`, `v_*`, or version-prefixed names for this
objective. Existing registry strings such as `iou_gibbs_v0` and
`ciou_gibbs_v0` may appear only when referring to historical implemented
negative-result paths; do not copy that naming pattern into new code.

## Infrastructure Refactor Scope

Keep the refactor surgical and correctness-driven. The implementation should
clean the seams that can silently change the objective, while avoiding broad
pipeline rewrites.

| Refactor | Required boundary |
|---|---|
| Pure coordinate target math | `src/detection/coord_soft_targets.py` owns Gaussian target construction, posterior weighting, structural legality, fp32/log-space math, diagnostics, and full-vocab coordinate softCE helpers. Recursive objective construction and trainer loss code must not duplicate this math. |
| Explicit candidate sidecar | `src/detection/objective.py` owns `TokenTarget.coord_instance_candidates` as the active semantic-branch remaining-instance sidecar. The sidecar uses a candidate-only `CoordInstanceCandidateSpec` shape with no coordinate-slot semantics. It must not overload legacy `coord_soft_targets`. |
| Legacy separation | Old IoU/CIoU-Gibbs paths keep their current metadata and tests for negative-result provenance. `instance_trie_gaussian` must require `coord_instance_candidates` and must not fall back to legacy exact-prefix metadata. |
| Coordinate vocabulary resolver | Runtime must resolve coordinate value count and token ids from token-row geometry rather than hard-coding `[0,1000]`. |
| Target-shape audit | A no-training target audit must run before tiny smoke training and must emit enough diagnostics to see whether ambiguity carries forward without rewarding union boxes. |
| Metric hygiene | Coordinate metrics use candidate/vocab/target terminology. Support/balance terminology is reserved for description and entry-choice positions. |
| Fair-comparison config diff | New smoke/prod configs must be diffed against `compact_full_support2.yaml` with a strict whitelist so the ablation changes only the coordinate objective and run identity. |
| No broad rewrite | Do not split or rewrite unrelated recursive detection, inference, eval, or trainer infrastructure unless a targeted invariant cannot be enforced otherwise. |

## Task 0: Approval Gate (Satisfied For Branch Implementation)

**Files:**

- Read: `docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md`
- Read: `docs/superpowers/plans/2026-05-14-instance-trie-gaussian-softce.md`

- [ ] **Step 1: Confirm canonical names before implementation**

Use these names unless the user explicitly overrides them before implementation:

```text
human objective: Instance-Trie Gaussian SoftCE
target_distribution: instance_trie_gaussian
config suffix: instance_trie_gaussian_softce_a5
production config: compact_full_support2_instance_trie_gaussian_softce_a5.yaml
ablation label: A5-instance-trie-gaussian
metric flag: recursive_detection_ce/coord_soft_ce/is_instance_trie_gaussian
```

Historical `A5-iou-gibbs` and `A6-ciou-gibbs` remain named historical
negative-result paths, not current recommended objectives.

- [ ] **Step 2: Record explicit user approval**

Implementation has started on the feature branch/worktree. Keep production
launches gated on target-shape audit, smoke workflow, diagnosis/audit review,
and a fresh explicit user approval.

- [ ] **Step 3: Confirm worktree**

Run:

```bash
git -C /data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/instance-trie-gaussian-softce status --short --branch
```

Expected: branch is `codex/instance-trie-gaussian-softce`; only approved
planning/doc files are dirty before implementation starts.

## Task 1: Target Math Tests

**Files:**

- Create or modify: `tests/test_instance_trie_gaussian_coord_softce.py`
- Modify after test failure: `src/detection/coord_soft_targets.py`

- [ ] **Step 1: Write failing tests for bbox-axis Gaussian width**

Add tests with this shape:

```python
def test_axis_gaussian_large_box_is_broader_than_tiny_box() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    tiny = (
        CoordSoftTargetCandidate(
            object_instance_id="tiny",
            slot_name="x1",
            bbox_xyxy=(100, 100, 109, 150),
            probability=1.0,
        ),
    )
    large = (
        CoordSoftTargetCandidate(
            object_instance_id="large",
            slot_name="x1",
            bbox_xyxy=(100, 100, 500, 700),
            probability=1.0,
        ),
    )

    tiny_dist = build_coord_soft_target(tiny, cfg)
    large_dist = build_coord_soft_target(large, cfg)

    assert large_dist.std.item() > tiny_dist.std.item()
    assert large_dist.entropy.item() > tiny_dist.entropy.item()
    assert large_dist.peak_prob.item() < tiny_dist.peak_prob.item()
```

- [ ] **Step 2: Write failing tests for localized multi-peak mixture**

Add:

```python
def test_instance_trie_gaussian_x1_mixture_has_separate_local_peaks() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "x1", (100, 100, 200, 200), 0.5),
        CoordSoftTargetCandidate("b", "x1", (500, 100, 620, 220), 0.5),
    )

    dist = build_coord_soft_target(candidates, cfg)

    assert dist.probs[100].item() > dist.probs[300].item()
    assert dist.probs[500].item() > dist.probs[300].item()
    assert dist.candidate_count.item() == pytest.approx(2.0)
```

- [ ] **Step 3: Write failing tests for uniform instance priors**

Add:

```python
def test_instance_trie_gaussian_uses_uniform_candidate_priors() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "x1", (100, 100, 200, 200), 0.99),
        CoordSoftTargetCandidate("b", "x1", (500, 100, 600, 200), 0.01),
    )

    dist = build_coord_soft_target(candidates, cfg)

    assert dist.probs[100].item() == pytest.approx(
        dist.probs[500].item(), rel=0.05
    )
```

Add a different-size component-normalization test:

```python
def test_instance_trie_gaussian_normalizes_each_candidate_before_mixing() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("tiny", "x1", (100, 100, 110, 200), 1.0),
        CoordSoftTargetCandidate("large", "x1", (500, 100, 900, 900), 1.0),
    )

    dist = build_coord_soft_target(candidates, cfg, return_components=True)

    assert dist.component_probs_by_id["tiny"].sum().item() == pytest.approx(1.0)
    assert dist.component_probs_by_id["large"].sum().item() == pytest.approx(1.0)
    assert dist.posterior["tiny"].item() == pytest.approx(0.5)
    assert dist.posterior["large"].item() == pytest.approx(0.5)
```

This test must fail for an implementation that sums raw unnormalized Gaussian
components and normalizes only after mixing; broad boxes must not receive extra
integrated mass under uniform priors.

- [ ] **Step 4: Write failing tests for coordinate-domain mapping**

Add a test that distinguishes token-id range from coordinate value bins:

```python
def test_coord_token_range_resolves_1000_value_bins() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )

    assert cfg.coord_token_end - cfg.coord_token_start + 1 == 1000
    assert cfg.coord_value_count == 1000
    assert cfg.coord_value_to_token_id(0) == 10
    assert cfg.coord_value_to_token_id(999) == 1009
```

- [ ] **Step 5: Write failing tests for soft posterior weighting**

Add:

```python
def test_previous_teacher_prefix_softly_downweights_incompatible_candidates() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "y1", (100, 100, 200, 200), 1.0),
        CoordSoftTargetCandidate("b", "y1", (500, 100, 620, 220), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="y1",
        teacher_prefix_values={"x1": 100},
    )

    assert dist.posterior_top1.item() > 0.99
    assert dist.posterior_entropy.item() < 0.1
```

Add a shared-prefix size-bias test:

```python
def test_shared_previous_coord_keeps_uniform_posterior_across_box_sizes() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("tiny", "y1", (100, 100, 109, 150), 1.0),
        CoordSoftTargetCandidate("large", "y1", (100, 100, 500, 700), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="y1",
        teacher_prefix_values={"x1": 100},
    )

    assert dist.posterior["tiny"].item() == pytest.approx(
        dist.posterior["large"].item(), rel=1e-6
    )
```

Add a near-shared case:

```python
def test_near_shared_top_left_carries_ambiguity_to_x2() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "x2", (100, 100, 200, 200), 1.0),
        CoordSoftTargetCandidate("b", "x2", (103, 101, 350, 260), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="x2",
        teacher_prefix_values={"x1": 100, "y1": 100},
    )

    assert dist.posterior_top1.item() < 0.99
    assert dist.probs[200].item() > 0.0
    assert dist.probs[350].item() > 0.0
```

Add posterior causality tests:

```python
def test_x1_posterior_uses_no_teacher_coordinates() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "x1", (100, 100, 200, 200), 1.0),
        CoordSoftTargetCandidate("b", "x1", (500, 100, 620, 220), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="x1",
        teacher_prefix_values={},
    )

    assert dist.posterior["a"].item() == pytest.approx(0.5)
    assert dist.posterior["b"].item() == pytest.approx(0.5)
```

Add slot-specific no-peeking cases:

```python
def test_y1_posterior_uses_x1_only_not_current_y1_or_future_slots() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "y1", (100, 100, 200, 200), 1.0),
        CoordSoftTargetCandidate("b", "y1", (100, 300, 700, 800), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="y1",
        teacher_prefix_values={"x1": 100},
    )

    assert dist.posterior["a"].item() == pytest.approx(0.5)
    assert dist.posterior["b"].item() == pytest.approx(0.5)


def test_x2_posterior_uses_top_left_only_not_current_x2_or_y2() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "x2", (100, 100, 200, 200), 1.0),
        CoordSoftTargetCandidate("b", "x2", (100, 100, 700, 800), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="x2",
        teacher_prefix_values={"x1": 100, "y1": 100},
    )

    assert dist.posterior["a"].item() == pytest.approx(0.5)
    assert dist.posterior["b"].item() == pytest.approx(0.5)


def test_y2_posterior_uses_x1_y1_x2_only_not_current_y2() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "y2", (100, 100, 200, 200), 1.0),
        CoordSoftTargetCandidate("b", "y2", (100, 100, 200, 800), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="y2",
        teacher_prefix_values={"x1": 100, "y1": 100, "x2": 200},
    )

    assert dist.posterior["a"].item() == pytest.approx(0.5)
    assert dist.posterior["b"].item() == pytest.approx(0.5)
```

- [ ] **Step 6: Write failing tests for union-box low-probability behavior**

Add:

```python
def test_union_box_coordinate_is_not_rewarded_after_prefix_disambiguates() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("a", "x2", (100, 100, 200, 200), 1.0),
        CoordSoftTargetCandidate("b", "x2", (500, 100, 620, 220), 1.0),
    )

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot="x2",
        teacher_prefix_values={"x1": 100, "y1": 100},
    )

    assert dist.probs[200].item() > dist.probs[620].item()
    assert dist.posterior_top1.item() > 0.99
    assert dist.probs[620].item() / dist.probs[200].item() < 0.05
```

- [ ] **Step 7: Write failing tests for structural legality and boundaries**

Add tests that the current-slot target is finite, normalized, and assigns zero
mass to structurally invalid bins:

```python
def test_structural_legality_masks_invalid_bins_and_normalizes() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidates = (
        CoordSoftTargetCandidate("tiny", "x1", (0, 0, 1, 1), 1.0),
    )

    dist = build_coord_soft_target(candidates, cfg)

    assert torch.isfinite(dist.probs).all()
    assert dist.probs.sum().item() == pytest.approx(1.0)
    assert dist.probs[1:].sum().item() == pytest.approx(0.0)
```

Also add hard-error cases for invalid bboxes:

```text
x1 >= x2
y1 >= y2
coordinate outside resolved value domain
non-integer coordinate value if the target builder receives raw values
```

- [ ] **Step 8: Write failing tests for pure coordinate softCE**

Add:

```python
def test_instance_trie_gaussian_uses_pure_softce_not_recursive_support_weight() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    logits = torch.zeros(1020, dtype=torch.float32)
    candidates = (
        CoordSoftTargetCandidate("a", "x1", (100, 100, 200, 200), 1.0),
    )

    result = full_vocab_coord_soft_ce(
        logits,
        candidates,
        cfg,
        support_weight=2.0,
        balance_weight=1.0,
    )

    assert result.weighted_loss.item() == pytest.approx(
        result.pure_soft_ce_equiv.item()
    )
```

Add a full-vocabulary pressure test:

```python
def test_coord_softce_uses_full_vocab_logsoftmax() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    logits = torch.zeros(1020, dtype=torch.float32)
    logits[0] = 20.0  # non-coordinate token
    candidates = (
        CoordSoftTargetCandidate("a", "x1", (100, 100, 200, 200), 1.0),
    )

    result = full_vocab_coord_soft_ce(logits, candidates, cfg)
    dist = build_coord_soft_target(candidates, cfg)
    coord_token_ids = cfg.coord_token_ids()
    full_vocab_expected = -(
        dist.probs * torch.nn.functional.log_softmax(logits, dim=-1)[coord_token_ids]
    ).sum()
    coord_only_wrong = -(
        dist.probs
        * torch.nn.functional.log_softmax(logits[coord_token_ids], dim=-1)
    ).sum()

    assert result.weighted_loss.item() == pytest.approx(
        full_vocab_expected.item()
    )
    assert result.weighted_loss.item() > coord_only_wrong.item() + 10.0
    assert result.weighted_loss.item() > 15.0
```

- [ ] **Step 9: Run tests and verify they fail**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_instance_trie_gaussian_coord_softce.py
```

Expected: fail because `instance_trie_gaussian`, `build_coord_soft_target`,
or `full_vocab_coord_soft_ce` is not implemented yet.

- [ ] **Step 10: Implement target math**

In `src/detection/coord_soft_targets.py`:

- extend `CoordSoftTargetDistributionName`
- allow `CoordSoftTargetRuntimeConfig(target_distribution="instance_trie_gaussian")` without `tau`
- add a generic
  `build_coord_soft_target(candidates, cfg, current_slot=None, teacher_prefix_values=None, device=None, return_components=False)`;
  unit fixtures may infer `current_slot` from `CoordSoftTargetCandidate.slot_name`,
  but production loss routing must pass the `TokenTarget` coordinate slot
- keep IoU/CIoU-Gibbs behavior intact for old configs
- ignore `CoordSoftTargetCandidate.probability` for `instance_trie_gaussian` and use uniform candidate priors
- pass only the causal teacher prefix into the helper; `x1` receives no teacher
  values, `y1` receives `x1`, `x2` receives `x1,y1`, and `y2` receives
  `x1,y1,x2`
- expose posterior diagnostics: `posterior_entropy`, `posterior_top1`, and `effective_candidate_count`
- expose per-candidate component diagnostics when `return_components=True`, so
  tests and target-shape audits can verify each candidate component is
  normalized before mixture
- compute all posterior and target-builder math in fp32/log-space before casting outputs as needed
- add normalized current-slot Gaussian log-weight math:

```python
variance = axis_length + 1.0
log_weights = -0.5 * torch.square(bins - center) / variance
log_weights = log_weights.masked_fill(~valid_mask, -torch.inf)
log_weights = log_weights - torch.logsumexp(log_weights[valid_mask], dim=0)
```

- add unnormalized prefix-compatibility math for posterior updates:

```python
prefix_energy = -0.5 * torch.square(teacher_value - candidate_value) / variance
```

Do not add `-log(sigma)` or a discrete normalizer to prefix compatibility.
That normalizer belongs only to current-slot target distributions.

- [ ] **Step 11: Run target math tests**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_instance_trie_gaussian_coord_softce.py
```

Expected: all tests pass.

## Task 2: Semantic-Branch Candidate Sidecar

**Files:**

- Modify: `tests/test_recursive_detection_ce_target_builder.py`
- Modify after test failure if needed: `src/detection/objective.py`

- [ ] **Step 1: Audit current sidecar behavior**

Read `src/detection/objective.py` around `TokenTarget`,
`_append_recursive_entry_targets`, `_append_hard_ce_targets`, and
`_coord_soft_targets_for_instances`.

Record in the implementation notes whether current `coord_soft_targets` means:

```text
legacy exact trie-node descendants
```

or:

```text
active semantic-branch coordinate-block candidates
```

Expected: if it is exact-prefix filtered, add a new
`coord_instance_candidates` sidecar rather than overloading legacy metadata.

Also record the exact coordinate-block snapshot point. For this objective it
must be after
the semantic branch reaches the coordinate block, including
`<|object_ref_start|>`, desc/object-ref tokens, and `<|box_start|>`, but before
any of `x1/y1/x2/y2` is consumed.

- [ ] **Step 2: Add semantic-branch candidate ownership test**

Add a compact-full sample with two same-desc objects:

```text
A: [100, 100, 200, 200]
B: [500, 100, 620, 220]
```

Assert:

```python
x1_target.coord_instance_candidates contains A and B
y1_target.coord_instance_candidates contains A and B
x2_target.coord_instance_candidates contains A and B
y2_target.coord_instance_candidates contains A and B
```

Also assert the sidecar does not include an unrelated object with a different
active desc/object-ref branch.

- [ ] **Step 3: Add remaining-instance exclusion tests**

Add a two-entry same-desc fixture and assert the second supervised entry's
coordinate sidecar excludes the first emitted object:

```text
entry 1 teacher: A cat [100, 100, 200, 200]
entry 2 teacher: B cat [500, 100, 620, 220]
```

Assert:

```python
second_entry_x1.coord_instance_candidates contains B
second_entry_x1.coord_instance_candidates does not contain A
```

If prefix-rollin is supported for this objective, add a fixture where roll-in
prefix objects are already consumed before the first supervised suffix object
and assert those roll-in objects are excluded. If prefix-rollin is not supported
for the initial implementation, add a config/runtime rejection test instead of a
target-builder test.

- [ ] **Step 4: Add near-shared coordinate carry-forward test**

Add a compact-full sample:

```text
A: [100, 100, 200, 200]
B: [103, 101, 350, 260]
```

Assert:

```python
x1_target.coord_instance_candidates contains A and B
y1_target.coord_instance_candidates contains A and B
x2_target.coord_instance_candidates contains A and B
y2_target.coord_instance_candidates contains A and B
```

This test should fail if the implementation uses exact-prefix trie descendants
for the new objective, because B would disappear after `x1=100`.

- [ ] **Step 5: Add teacher candidate presence and uniqueness test**

Assert every coordinate `TokenTarget` with `coord_instance_candidates` contains
the teacher object's `object_instance_id`. This is required so the loss can
derive `teacher_bbox_xyxy` without decoded-string matching.

Add explicit hard-error fixtures for zero or multiple matching teacher candidate
ids.

- [ ] **Step 6: Add unpacked collation and cross-record isolation tests**

Build a mini-batch with two records that reuse the same desc text and candidate
ids. Assert unpacked collation preserves each target's own
`coord_instance_candidates` tuple and never leaks candidates across records.
This objective does not add packed sidecar offset-rewriting support.

Also assert that a packed/raw batch containing `recursive_detection_targets`
continues to raise the existing packing-incompatibility error. Keep production
and smoke config diffs enforcing `training.packing: false` and
`training.eval_packing: false`.

- [ ] **Step 7: Run target builder tests and verify failures**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_recursive_detection_ce_target_builder.py
```

Expected: new tests fail until `coord_instance_candidates` exists or the current
sidecar is proven to already satisfy semantic-branch candidate ownership.

- [ ] **Step 8: Implement smallest sidecar change**

In `src/detection/objective.py`:

- add a candidate-only sidecar type such as:

```python
@dataclass(frozen=True)
class CoordInstanceCandidateSpec:
    object_instance_id: str
    bbox_xyxy: tuple[int, int, int, int]
    branch_key: str | None = None
    source_record_id: str | None = None
```

- add `coord_instance_candidates: tuple[CoordInstanceCandidateSpec, ...] = ()`
  to `TokenTarget`
- add `coord_slot_name: CoordSlotName | None = None` to `TokenTarget` so the
  loss can build x1/y1/x2/y2 distributions without reading slot names from the
  candidate sidecar
- populate it for coordinate positions from the active desc/object-ref branch
  candidate set snapshot after semantic selection and before `x1`
- attach the same candidate tuple to all four coordinate slots in that bbox
  block; posterior weighting, not sidecar recomputation, handles x1->y1->x2->y2
  disambiguation
- do not store coordinate slot names or legacy probabilities on
  `CoordInstanceCandidateSpec`; the current coordinate slot must come from the
  `TokenTarget`/loss context
- exclude emitted objects and roll-in-prefix objects from the candidate tuple
- keep legacy `coord_soft_targets` unchanged for old IoU/CIoU-Gibbs provenance
  unless a targeted compatibility audit approves replacing it
- never populate it from all image objects or decoded desc text

- [ ] **Step 9: Run target builder tests**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_recursive_detection_ce_target_builder.py
```

Expected: pass.

## Task 3: Loss Routing

**Files:**

- Modify: `tests/test_recursive_detection_ce_loss_adapter.py`
- Modify: `src/detection/loss.py`

- [ ] **Step 1: Add coordinate replacement test**

Add a test that builds a recursive target with
`coord_instance_candidates`, enables:

```python
CoordSoftTargetRuntimeConfig(
    target_distribution="instance_trie_gaussian",
    coord_token_start=10,
    coord_token_end=1009,
)
```

and asserts the per-position coordinate loss equals the manual helper call:

```python
manual = full_vocab_coord_soft_ce(
    logits[target.position - 1],
    tuple(
        CoordSoftTargetCandidate(
            spec.object_instance_id,
            target.coord_slot_name,
            spec.bbox_xyxy,
            1.0,
        )
        for spec in target.coord_instance_candidates
    ),
    cfg,
    current_slot=target.coord_slot_name,
    teacher_prefix_values=teacher_prefix_values_for_slot(
        target.coord_slot_name,
        teacher_candidate.bbox_xyxy,
    ),
    support_weight=2.0,
    balance_weight=1.0,
)
```

- [ ] **Step 2: Add desc and entry-choice support/balance preservation tests**

Use an existing desc multi-positive target and assert that enabling
`instance_trie_gaussian` does not change non-coordinate support/balance
losses.

Also add a control-looking entry-choice fixture, such as a branch where
`<|box_start|>` competes with a continuing description token. Assert this
position still uses ET-RMP support/balance, not deterministic schema hard CE.

- [ ] **Step 3: Add deterministic schema hard-CE preservation test**

Use a deterministic protocol token after the entry choice is fixed and without
`coord_instance_candidates`. Assert it still uses the hard teacher token CE
path. Do not use a special token that is still part of an entry-choice trie
branch as this fixture.

- [ ] **Step 4: Add missing-metadata hard-error tests**

Enable `instance_trie_gaussian` for a coordinate teacher target and assert a
clear `ValueError` when:

```text
coord_instance_candidates is missing or empty
only legacy coord_soft_targets is present
teacher object_instance_id is absent from coord_instance_candidates
teacher object_instance_id appears more than once
teacher coordinate token id is outside the resolved coord-token range
candidate object_instance_id is empty or non-string
candidate bbox is not a four-integer xyxy tuple
candidate bbox has non-integer coordinates
candidate bbox coordinate is outside the resolved coordinate value domain
candidate bbox violates x1 < x2 or y1 < y2
```

These failures should raise targeted `ValueError`s at sidecar ingestion or loss
routing, before constructing the coordinate distribution. Do not rely on
incidental `TypeError`, `IndexError`, non-finite math, or legacy fallback.

- [ ] **Step 5: Add type-gate preservation test**

If the current latest recursive CE runtime adds a coordinate/type-gate auxiliary
term, assert enabling `instance_trie_gaussian` preserves that auxiliary term.
The coordinate token CE replacement must only remove recursive support/balance
weighting from coordinate positions.

- [ ] **Step 6: Run loss adapter tests and verify failures**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_recursive_detection_ce_loss_adapter.py
```

Expected: new coordinate routing test fails until the loss helper is wired.

- [ ] **Step 7: Implement routing**

In `src/detection/loss.py`:

- treat a target as coordinate-softCE eligible only when config is enabled and
  `target.coord_instance_candidates` is non-empty for
  `instance_trie_gaussian`
- do not use legacy `coord_soft_targets` as a fallback for
  `instance_trie_gaussian`; those exact-prefix sidecars remain legacy
  provenance only
- raise a hard error if `instance_trie_gaussian` is enabled at a coordinate
  target but semantic-branch candidates or the teacher candidate bbox are
  missing
- call the pure coordinate softCE helper for eligible coordinate positions
- derive the teacher bbox from the candidate whose `object_instance_id` matches
  `target.object_instance_id`, then pass only the causal
  `teacher_prefix_values` for the current `target.coord_slot_name` into the
  helper
- require exactly one matching teacher candidate
- compute `log_softmax` over the full model vocabulary before indexing resolved
  coordinate-token ids
- keep desc/entry-choice trie targets on `support_balance_loss`
- keep hard targets on CE
- preserve existing type-gate auxiliary supervision
- emit coordinate softCE metric events, including slot-specific posterior
  entropy, posterior top1 weight, effective candidate count, target entropy,
  and target peak probability

- [ ] **Step 8: Run loss tests**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_recursive_detection_ce_loss_adapter.py
```

Expected: pass.

## Task 4: Schema, Runtime, Metrics, And Configs

**Files:**

- Modify: `tests/test_latest_training_config_contract.py`
- Modify: `src/config/schema.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/trainers/metrics/recursive_detection.py`
- Create: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_gaussian_softce_a5.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_tiny.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_ddp8_preflight.yaml`

- [ ] **Step 1: Add schema tests**

Add tests asserting this minimal config parses:

```yaml
objective:
  coord_soft_ce:
    enabled: true
    target_distribution: instance_trie_gaussian
```

Add tests asserting these fields are rejected for
`instance_trie_gaussian`:

```yaml
tau: 0.01
tau_source: ciou
weighting: gibbs
replace_coord_hard_ce: true
apply_to_multi_positive: true
sigma: 2.0
truncate: 16
target_sigma: 2.0
target_truncate: 16
```

Also assert that legacy targets such as `iou_gibbs_v0` and `ciou_gibbs_v0`
continue to parse with their existing required knobs. This protects
negative-result provenance while making the new objective surface strict.

- [ ] **Step 2: Run schema tests and verify failures**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_latest_training_config_contract.py
```

Expected: new tests fail until schema/runtime accept the new target.

- [ ] **Step 3: Implement schema/runtime**

Implement the smallest config surface:

```yaml
coord_soft_ce:
  enabled: true
  target_distribution: instance_trie_gaussian
```

Runtime should resolve:

```python
CoordSoftTargetRuntimeConfig(
    target_distribution="instance_trie_gaussian",
    coord_token_start=resolved_start,
    coord_token_end=resolved_end,
)
```

Do not require or accept `tau`, `tau_source`, or any stale weighting/truncation
knob for this distribution.

Add a runtime resolver integration test that constructs or loads a latest
recursive detection config with `token_rows.groups.coord_geometry`, resolves the
runtime, and asserts `CoordSoftTargetRuntimeConfig` uses token-row-derived ids.
Include an offset fixture where coordinate values map to token ids that are not
literal `0..999`; this must fail any implementation that hard-codes value ids
as token ids.

Trainer metrics should not assume `tau` exists. `target_distribution` is string
provenance and belongs in resolved config/manifests/artifacts, not flat numeric
metric payloads. If a numeric runtime flag is useful, use:

```text
recursive_detection_ce/coord_soft_ce/is_instance_trie_gaussian
```

Add flat reporter aliases for:

```text
recursive_detection_ce/coord_soft_ce/x1/posterior_entropy
recursive_detection_ce/coord_soft_ce/y1/posterior_entropy
recursive_detection_ce/coord_soft_ce/x2/posterior_entropy
recursive_detection_ce/coord_soft_ce/y2/posterior_entropy
recursive_detection_ce/coord_soft_ce/x1/posterior_top1
recursive_detection_ce/coord_soft_ce/y1/posterior_top1
recursive_detection_ce/coord_soft_ce/x2/posterior_top1
recursive_detection_ce/coord_soft_ce/y2/posterior_top1
recursive_detection_ce/coord_soft_ce/x1/effective_candidate_count
recursive_detection_ce/coord_soft_ce/y1/effective_candidate_count
recursive_detection_ce/coord_soft_ce/x2/effective_candidate_count
recursive_detection_ce/coord_soft_ce/y2/effective_candidate_count
recursive_detection_ce/coord_soft_ce/x1/target_entropy
recursive_detection_ce/coord_soft_ce/y1/target_entropy
recursive_detection_ce/coord_soft_ce/x2/target_entropy
recursive_detection_ce/coord_soft_ce/y2/target_entropy
recursive_detection_ce/coord_soft_ce/x1/target_peak_prob
recursive_detection_ce/coord_soft_ce/y1/target_peak_prob
recursive_detection_ce/coord_soft_ce/x2/target_peak_prob
recursive_detection_ce/coord_soft_ce/y2/target_peak_prob
```

Metric contract:

| Key or field | Owner | Trainer metric | Target audit | Type | Shape |
|---|---|---:|---:|---|---|
| `target_distribution` | resolved config / manifests | no | yes | string | aggregate |
| `recursive_detection_ce/coord_soft_ce/is_instance_trie_gaussian` | trainer metrics | yes | yes | numeric flag | aggregate |
| `candidate_count` | target builder / audit | optional aggregate | yes | numeric | aggregate + slot |
| `effective_candidate_count` | target builder / trainer | yes | yes | numeric | slot |
| `posterior_entropy` | target builder / trainer | yes | yes | numeric | slot |
| `posterior_top1` | target builder / trainer | yes | yes | numeric | slot |
| `target_entropy` | target builder / trainer | yes | yes | numeric | slot |
| `target_std` | target builder / audit | optional aggregate | yes | numeric | slot |
| `target_peak_prob` | target builder / trainer | yes | yes | numeric | slot |
| `effective_coord_bin_count` | target builder / audit | optional aggregate | yes | numeric | slot |
| `coord_vocab_bin_count` | runtime / audit | optional aggregate | yes | numeric | aggregate |
| `teacher_candidate_missing_count` | audit / smoke gate | optional aggregate | yes | numeric count | aggregate |
| `candidate_leak_count` | audit / smoke gate | optional aggregate | yes | numeric count | aggregate |
| `nonfinite_target_count` | audit / smoke gate | optional aggregate | yes | numeric count | aggregate |

- [ ] **Step 4: Add configs**

Create production and smoke configs extending the existing A2/support2 base.
Use canonical names containing `instance_trie_gaussian_softce_a5`. Keep the
same dataset, template, model, optimizer, token rows, batch semantics, and run
roots as the fair-comparison configs. Do not launch them in this task.

Add a resolved-config diff check against `compact_full_support2.yaml` with a
strict whitelist:

```text
/objective/coord_soft_ce/enabled
/objective/coord_soft_ce/target_distribution
/training/run_name
/training/artifact_subdir
/output_dir
/paths/output_root
/experiment/name
/experiment/tag
```

Smoke-only whitelist additions:

```text
/training/max_steps
/data/sample_limit
/data/max_samples
/eval/max_samples
```

DDP8-preflight-only whitelist additions:

```text
/distributed
/launch
/training/per_device_train_batch_size only if required by the existing preflight pattern
```

All packing and cache surfaces must remain equal to the base comparator,
including `training.packing`, `training.eval_packing`, padding-free packing,
encoded-sample cache, prompt/template, model path, dataset config, token rows,
optimizer, schedule, and effective batch semantics.

Create a structured config-diff test in
`tests/test_instance_trie_gaussian_config_diff.py` that resolves the base,
prod, tiny, and DDP8 preflight configs, computes JSON-pointer-like changed
paths, and fails if any changed path is not in the appropriate whitelist.

- [ ] **Step 5: Run config tests**

Run:

```bash
conda run -n ms python -m pytest -q tests/test_latest_training_config_contract.py
```

Expected: pass.

## Task 5: Documentation Update After Implementation Start

**Files:**

- Modify: `docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/superpowers/plans/2026-05-14-instance-trie-gaussian-softce.md`

- [ ] **Step 1: Update training docs after branch implementation starts**

Revise the existing `instance_trie_gaussian` provenance note to say the
objective is implemented on the feature branch/worktree but remains an
implementation draft until target-shape audit, smoke workflow,
diagnosis/audit review, merge acceptance, and production approval complete.

- [ ] **Step 2: Preserve negative-result provenance**

Do not delete progress notes or metric artifacts from the old A5/A6 runs. If
old configs remain, mark them as negative-result provenance rather than current
recommended configs.

- [ ] **Step 3: Disambiguate A5 naming**

Before any production launch, grep docs/configs for bare `A5` references and
make each occurrence carry an objective suffix:

```text
A5-iou-gibbs negative-result / superseded
A6-ciou-gibbs negative-result / superseded
A5-instance-trie-gaussian current successor
```

Use the canonical label from Task 0. Do not leave two different objectives
described as plain `A5`.

- [ ] **Step 4: Run docs/config grep sanity**

Run:

```bash
rg -n "instance_trie_gaussian|iou_gibbs_v0|ciou_gibbs_v0|coord_soft_ce" docs configs/stage1/recursive_detection_ce_latest
```

Expected: docs distinguish new draft/current config from old negative-result
configs.

## Task 6: Focused Test Matrix

**Files:** no new files expected.

- [ ] **Step 1: Run focused unit/config tests**

Run:

```bash
conda run -n ms python -m pytest -q \
  tests/test_instance_trie_gaussian_coord_softce.py \
  tests/test_iou_gibbs_coord_softce.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_latest_training_config_contract.py \
  tests/test_instance_trie_gaussian_config_diff.py \
  tests/test_instance_trie_gaussian_target_shape_audit.py
```

Expected: pass. Old IoU/CIoU-Gibbs tests must still pass unless the user
explicitly approves deleting that code path.

- [ ] **Step 2: Run post-implementation model-innovation risk audit**

After code implementation and focused tests pass, use
`model-innovation-risk-audit` as a read-only double check before any
target-shape audit, tiny smoke, DDP preflight, or production launch.

Audit implementation accuracy against:

```text
design docs and this plan
schema/runtime config
semantic-branch candidate sidecars
remaining/not-yet-emitted candidate ownership
roll-in prefix exclusion or explicit rejection
candidate-only CoordInstanceCandidateSpec sidecar
coord_slot_name ownership outside candidate sidecar
teacher-candidate bbox lookup
malformed sidecar hard-error behavior
legacy coord_soft_targets no-fallback behavior
soft posterior math
unnormalized prefix compatibility vs normalized current-slot targets
per-candidate normalization before mixture
no current/future coordinate peeking
full-vocabulary log-softmax pressure
loss routing
metric names
smoke configs
train/eval provenance
resolved-config diff whitelist
changed-file scope guard
```

Expected: no P0/P1 findings before the no-training target-shape audit or smoke.
Any P2 accepted for later work must be recorded with an owner, rationale, and
why it cannot change this ablation's training signal.

Record the audit result as a short artifact or note, for example:

```text
progress/audits/2026-05-14-instance-trie-gaussian-post-implementation-audit.md
```

The note must distinguish implementation-contract evidence from tiny/smoke
behavior evidence; do not present either as full validation.

- [ ] **Step 3: Run changed-file scope guard**

Run:

```bash
git diff --name-only
```

Expected changed files are limited to the planned surfaces:

```text
docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md
docs/training/STAGE1_OBJECTIVE.md
docs/superpowers/plans/2026-05-14-instance-trie-gaussian-softce.md
progress/audits/2026-05-14-instance-trie-gaussian-post-implementation-audit.md
progress/audits/README.md
src/config/schema.py
src/detection/coord_soft_targets.py
src/detection/objective.py
src/detection/loss.py
src/detection/runtime.py
src/detection/__init__.py
src/trainers/metrics/recursive_detection.py
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_gaussian_softce_a5.yaml
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_tiny.yaml
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_ddp8_preflight.yaml
scripts/diagnostics/audit_instance_trie_gaussian_targets.py
tests/test_instance_trie_gaussian_coord_softce.py
tests/test_instance_trie_gaussian_config_diff.py
tests/test_instance_trie_gaussian_target_shape_audit.py
tests/test_recursive_detection_ce_target_builder.py
tests/test_recursive_detection_ce_loss_adapter.py
tests/test_latest_training_config_contract.py
```

No files under `src/infer/`, `src/eval/`, upstream HF model code, or unrelated
artifact-pipeline surfaces should change. Any exception needs a short written
justification and a targeted test.

## Task 7: No-Training Target-Shape Audit

**Files:**

- Create: `scripts/diagnostics/audit_instance_trie_gaussian_targets.py`
- Create: `tests/test_instance_trie_gaussian_target_shape_audit.py`

- [x] **Step 1: Add executable audit runner and artifact schema test**

Create a checked-in no-training audit command:

```bash
conda run -n ms python scripts/diagnostics/audit_instance_trie_gaussian_targets.py \
  --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_tiny.yaml \
  --output temp/instance_trie_gaussian/target_shape_audit.json \
  --include-record-idx 27
```

The command must not initialize model weights or run optimization. It should
construct target distributions from synthetic fixtures and a fixed real-sample
probe, then write JSON with this shape:

```json
{
  "target_distribution": "instance_trie_gaussian",
  "config_path": "...",
  "synthetic_fixtures": [],
  "real_probe_records": [],
  "summary": {
    "teacher_candidate_missing_count": 0,
    "candidate_leak_count": 0,
    "nonfinite_target_count": 0
  },
  "slots": {
    "x1": {},
    "y1": {},
    "x2": {},
    "y2": {}
  }
}
```

Each synthetic fixture and slot entry must include:

```text
candidate_count
effective_candidate_count
posterior_entropy
posterior_top1
target_entropy
target_peak_prob
target_std
union_coordinate_probability_ratio
component_mass_by_candidate_id
```

Add `tests/test_instance_trie_gaussian_target_shape_audit.py` to validate the
artifact schema and synthetic pass/fail gates.

- [x] **Step 2: Build synthetic target-shape fixtures**

Before training, run a no-optimization audit that constructs target
distributions for synthetic cases:

```text
two far-apart same-desc objects
near-shared top-left objects
same previous coordinate with tiny and large boxes
already-emitted same-desc object excluded
structural boundary tiny boxes
```

Record for each coordinate slot:

```text
candidate_count
effective_candidate_count
posterior_entropy
posterior_top1
target_entropy
target_peak_prob
target_std
union_coordinate_probability_ratio
component_mass_by_candidate_id
```

- [x] **Step 3: Build fixed real-sample target-shape probe**

Use a small fixed real subset with repeated desc objects. Include `record_idx=27`
from the old negative-result diagnosis if that record is still present in the
current dataset manifest; otherwise record the replacement ids explicitly in
the audit artifact.

Expected: the target-shape audit demonstrates multi-peak x1 behavior where
appropriate and sharpening on later slots without requiring any model training.

- [x] **Step 4: Run the audit command**

Run:

```bash
conda run -n ms python scripts/diagnostics/audit_instance_trie_gaussian_targets.py \
  --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_tiny.yaml \
  --output temp/instance_trie_gaussian/target_shape_audit.json \
  --include-record-idx 27
```

Then run:

```bash
conda run -n ms python -m pytest -q tests/test_instance_trie_gaussian_target_shape_audit.py
```

Expected: the artifact exists, records fixture/sample ids, includes all required
keys, and passes synthetic gates.

- [x] **Step 5: Gate tiny training on target-shape audit**

Do not run tiny training until the target-shape audit passes these qualitative
checks:

```text
x1 effective_candidate_count can exceed 1 on repeated-desc fixtures
near-shared top-left ambiguity survives to x2
far-away incompatible candidates are downweighted after matching x1/y1 prefix
union-coordinate probability ratio stays low after prefix disambiguation
missing sidecar fixture raises a hard error
```

## Task 8: Smoke Plan After Approval

**Files:** no code files expected unless smoke reveals a bug.

- [x] **Step 1: Use full-pipeline-smoke**

After user approval and after the no-training target-shape audit passes, run the
tiny smoke config under the existing smoke workflow. Confirm at least:

```text
coord softCE config resolves
coord diagnostics log
slot-level posterior diagnostics log
loss is finite
semantic-branch coordinate candidate counts are nonzero
schema/desc/coord group metrics are not collapsed into one misleading scalar
inference/eval path is unchanged and uses no oracle candidate constraint
```

Completed evidence:

```text
tiny run:
  temp/recursive_detection_ce_latest/output/compact_full_instance_trie_gaussian_softce_a5_tiny/smoke-compact-full-instance_trie_gaussian_softce_a5-tiny/v0-20260514-121444

launcher log:
  temp/instance_trie_gaussian/train_logs/compact_full_support2_instance_trie_gaussian_softce_a5_tiny-20260514T121300Z.log

key tiny metrics:
  train_loss = 12.67943001
  coord_soft_ce/is_instance_trie_gaussian = 1.0
  coord_soft_ce/balance_loss = 0.0
  x1/effective_candidate_count = 2.77777767
  y1/effective_candidate_count = 1.11110139
  x2/effective_candidate_count = 1.0
  y2/effective_candidate_count = 1.0
```

- [x] **Step 2: Run model-innovation risk audit on smoke behavior**

After tiny smoke finishes, use `model-innovation-risk-audit` again to inspect
model training behavior before any DDP8 preflight or production launch.

Audit at least:

```text
tiny smoke resolved config and run manifest
target_shape_audit.json
trainer logs and metric keys
schema/desc/coord loss group contributions
coordinate position count and finite loss
posterior entropy/top1 trends by x1/y1/x2/y2
target entropy/peak/std by coordinate slot
hard-error counters all zero
candidate_count and effective_candidate_count nonzero where expected
absence of oracle decoding, reranking, or confidence post-op headline changes
tiny inference/eval artifacts if generated by the smoke workflow
```

Expected: no P0/P1 findings. Any warning that suggests diffuse targets,
immediate exact-prefix collapse, missing coordinate metrics, changed eval
surface, or suspicious coord loss contribution blocks DDP8 preflight until
resolved or explicitly waived by the user.

Append this behavior audit to the post-implementation audit note or create a
separate note:

```text
progress/audits/2026-05-14-instance-trie-gaussian-smoke-behavior-audit.md
```

Completed evidence:

```text
audit note:
  progress/audits/2026-05-14-instance-trie-gaussian-smoke-behavior-audit.md

target-shape audit:
  temp/instance_trie_gaussian/target_shape_audit.json

target-shape counters:
  candidate_leak_count = 0
  nonfinite_target_count = 0
  teacher_candidate_missing_count = 0

focused tests:
  174 passed in 7.73s
```

- [x] **Step 3: Run DDP8 preflight only after tiny smoke and behavior audit pass**

Use the DDP8 preflight config
`configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_gaussian_softce_a5_ddp8_preflight.yaml`.
Do not launch production training in this plan
without a fresh user instruction.

Expected: startup healthy, no non-finite loss, no sidecar alignment error.

Completed evidence:

```text
DDP8 run:
  temp/recursive_detection_ce_latest/output/compact_full_instance_trie_gaussian_softce_a5_ddp8_preflight/smoke-compact-full-instance_trie_gaussian_softce_a5-ddp8-preflight/v0-20260514-122519

launcher log:
  temp/instance_trie_gaussian/train_logs/compact_full_support2_instance_trie_gaussian_softce_a5_ddp8_preflight-20260514T122339Z.log

resolved batch shape:
  per_device_train_batch_size = 16
  world_size = 8
  global_effective = 128
  max_steps = 4

train loss series:
  11.4318, 11.7557, 11.0908, 10.4684

eval loss series:
  11.9755, 11.4867, 10.9156, 10.6649

coord weighted-loss series:
  train = 21.6260, 21.4509, 20.7999, 20.0667
  eval = 21.9242, 21.0397, 20.0072, 19.5494

coordinate objective flags:
  coord_soft_ce/is_instance_trie_gaussian = 1.0
  coord_soft_ce/balance_loss = 0.0

posterior behavior:
  x1/effective_candidate_count ~= 2.92-3.44
  y1/effective_candidate_count ~= 1.07-1.11
  x2/effective_candidate_count ~= 1.01
  y2/effective_candidate_count ~= 1.00

resource/runtime:
  max memory reported by trainer = 60.61 GiB
  A100 80GB GPUs free after run
  no non-finite metrics in logging.jsonl or train_heartbeat.rank0.jsonl
  no launcher errors, Traceback, RuntimeError, or CUDA OOM in the DDP8 log
```

## Self-Review

Spec coverage:

- token group separation: Tasks 3 and 5
- instance-aware coordinate support: Tasks 1 and 2
- semantic-branch candidate ownership: Task 2
- uniform instance priors: Task 1
- soft posterior from previous teacher-forced coordinates: Tasks 1 and 3
- bbox-size-aware Gaussian width: Task 1
- no union basin from independent marginals: Tasks 1 and 2
- no extra user knobs: Task 4
- production-prep configs: Task 4
- target-shape/no-training audit gate: Task 7
- audit/smoke gates: Tasks 6 and 8

Placeholder scan:

- This plan intentionally leaves implementation unexecuted, but each approved
  implementation task names concrete files, commands, and expected outcomes.

Type consistency:

- Draft helper names are `build_coord_soft_target` and
  `full_vocab_coord_soft_ce`; target construction takes `current_slot` and
  causal `teacher_prefix_values`, not a full teacher bbox. Loss routing derives
  the teacher bbox only to construct the causal prefix for the current
  `TokenTarget.coord_slot_name`. If implementation keeps the old helper name
  for compatibility, tests should import the final public name and
  `src/detection/__init__.py` should export it.
