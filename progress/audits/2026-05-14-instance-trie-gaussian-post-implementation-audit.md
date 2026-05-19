---
doc_id: progress.audits.instance-trie-gaussian-post-implementation-2026-05-14
layer: progress
doc_type: audit
status: concluded
domain: training
summary: Post-implementation risk audit for Instance-Trie Gaussian SoftCE before target-shape audit and smoke.
updated: 2026-05-14
---

# Instance-Trie Gaussian SoftCE Post-Implementation Audit (2026-05-14)

Scope: `/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/instance-trie-gaussian-softce` on branch `codex/instance-trie-gaussian-softce`.

Status: implementation-contract audit completed before target-shape audit, tiny smoke, DDP8 preflight, or production launch. Follow-up target-shape and smoke/preflight behavior evidence is recorded in `progress/audits/2026-05-14-instance-trie-gaussian-smoke-behavior-audit.md`.

## Contract

`Instance-Trie Gaussian SoftCE` adds `target_distribution: instance_trie_gaussian` for compact recursive detection. Schema/control tokens stay on hard CE, description and entry-choice ambiguity stay on ET-RMP support+balance, and coordinate tokens use pure full-vocabulary SoftCE against active-branch remaining-instance Gaussian mixtures.

Coordinate candidates must come from recursive target-construction sidecars, not all image objects or decoded-string matching. Candidate priors are uniform. Prefix posterior weights use previous teacher-forced coordinates only with unnormalized Gaussian mismatch energy. Current-slot candidate distributions are normalized over the resolved coordinate-token vocabulary before mixture.

## Findings

- P0/P1: none found in the read-only risk audit.
- P2: missing explicit teacher-token versus teacher-candidate bbox consistency guard. Fixed before smoke by adding a fail-fast check in `src/detection/loss.py` and a targeted regression in `tests/test_recursive_detection_ce_loss_adapter.py`.
- P2: no-training target-shape audit was not yet executable. Task 7 owns `scripts/diagnostics/audit_instance_trie_gaussian_targets.py` and `tests/test_instance_trie_gaussian_target_shape_audit.py`; smoke remains blocked until that audit passes.
- P3: direct construction of `CoordSoftTargetRuntimeConfig(target_distribution="instance_trie_gaussian", tau=...)` could carry a stale-looking ignored `tau`. Fixed before smoke by rejecting `tau` for `instance_trie_gaussian` in `src/detection/coord_soft_targets.py` and adding a regression in `tests/test_instance_trie_gaussian_coord_softce.py`.

## Confirmed OK

- Candidate sidecars are snapshotted once at the active semantic branch and reused across x1/y1/x2/y2.
- Already emitted instances are excluded before the next entry is supervised.
- Instance-Gaussian routing does not fall back to legacy `coord_soft_targets`.
- Uniform candidate priors, bbox-size-aware variance, current-slot per-candidate normalization, prefix causality, structural legality masks, and full-vocabulary log-softmax pressure are covered by focused tests.
- Schema/runtime/config surfaces expose only `enabled` and `target_distribution` for the new public YAML objective; stale YAML knobs are rejected.
- New A5 successor provenance is explicit: `A5-instance-trie-gaussian` supersedes historical negative-result `A5-iou-gibbs` / `A6-ciou-gibbs` naming for this successor direction.

## Verification

Focused implementation suite after the two audit fixes:

```bash
conda run -n ms python -m pytest -q \
  tests/test_instance_trie_gaussian_coord_softce.py \
  tests/test_iou_gibbs_coord_softce.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_latest_training_config_contract.py \
  tests/test_instance_trie_gaussian_config_diff.py
```

Result: `168 passed in 1.59s`.

`git diff --check` was clean after the guard fixes.

## Follow-Up Gates

- No-training target-shape audit and its tests completed.
- Tiny full-pipeline smoke completed after the target-shape audit passed.
- Behavior-focused model-innovation risk audit on smoke artifacts completed.
- DDP8 preflight completed after the tiny smoke and behavior audit.
- Do not treat target-shape, tiny, or smoke evidence as full validation.
