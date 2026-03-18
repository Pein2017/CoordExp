## 1. Stop-Branch Metadata

- [x] 1.1 Add canonical semantic stop-branch discovery to the teacher-forcing metadata path so packed segments expose the first terminal `']}'`, the closure-tail positions, and the `']},'` vs `']}'` branch pair without raw-brace heuristics.
- [x] 1.2 Add targeted tests for stop-branch discovery covering multi-object dense targets, closure-tail detection, packing-safe segment locality, and fail-fast behavior when the semantic branch is ambiguous.

## 2. Token CE Damping

- [x] 2.1 Extend `token_ce` config validation to accept `token_ce.config.stop_signal_damping` with the locked defaults and numeric constraints from the spec (`enabled=false`, `min_weight=0.2`, `max_weight=1.0`, `branch_temperature=1.0`, `curve_gamma=2.0`, `detach_gate=true`).
- [x] 2.2 Implement the branch-local stop-signal weighting path in `token_ce` so eligible `context=gt` semantic stop positions contribute `stop_signal_ce`, fall back to `struct_ce` when disabled, and never double-count the same token.
- [x] 2.3 Add targeted tests for the damping math and config behavior:
  - unknown-key and invalid-range fail-fast checks
  - `detach_gate=true` vs `detach_gate=false`
  - pair-local `p_stop`, `p_cont`, and `margin` semantics after `branch_temperature`
  - fallback to ordinary `struct_ce` when the feature is disabled

## 3. Canonical Registry And Metrics

- [x] 3.1 Update the unified-loss / projection path so `stop_signal_ce` is emitted canonically for Stage-1 `loss/gt_text/stop_signal_ce` and Stage-2 Channel-A `loss/A1_text/stop_signal_ce`.
- [x] 3.2 Add provenance-aware stop-signal diagnostics:
  - Stage-1: `stop_signal/gt/{eligible_seq_count,branch_count,weight_mean,p_stop_mean,p_cont_mean,margin_mean}`
  - Stage-2 Channel-A: `stop_signal/A1/{eligible_seq_count,branch_count,weight_mean,p_stop_mean,p_cont_mean,margin_mean}`
- [x] 3.3 Add or update tests so stop-signal counters aggregate additively, gauge metrics remain mean-like, and disabled / no-eligible-branch steps omit `stop_signal/gt/*` and `stop_signal/A1/*` keys.
- [x] 3.4 Update [METRICS.md](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/docs/training/METRICS.md) and [STAGE2_RUNBOOK.md](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/docs/training/STAGE2_RUNBOOK.md) so the new objective atoms, diagnostics, and downstream readout expectations are documented.

## 4. YAML Surfaces

- [x] 4.1 Add feature-enabled authored YAML examples for the relevant Stage-1 and Stage-2 surfaces without introducing new CLI flags.
- [x] 4.2 Add a dedicated Stage-2 smoke config under `configs/stage2_two_channel/smoke/` that extends `configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml`, enables `token_ce.config.stop_signal_damping`, sets a tiny reproducible run budget, and writes to smoke-specific `training.run_name`, `training.output_dir`, and `training.logging_dir`.
- [x] 4.3 Record reproducibility checkpoints in the smoke config and validation notes:
  - config path
  - `training.run_name`
  - output/logging directories
  - seed / deterministic knobs inherited from the resolved config
  - expected emitted stop-signal objective and diagnostic keys

## 5. Validation

- [x] 5.1 Run targeted config and trainer tests for the new stop-signal path, including the Stage-1 single-forward and Stage-2 Channel-A `A1` cases.
- [x] 5.2 Validate the OpenSpec change artifacts after code/config/doc updates:
  - `openspec validate --type change adaptive-stop-signal-damping --strict --no-interactive`
- [x] 5.3 Run the requested two-GPU smoke after the feature is enabled using the dedicated smoke config from task `4.2`:
  - `config=configs/stage2_two_channel/smoke/<stop-signal-smoke>.yaml gpus=0,1 bash scripts/train.sh`
- [x] 5.4 Verify the smoke artifacts:
  - rank-0 output contains the resolved `config_source.yaml`
  - training logs include `loss/A1_text/stop_signal_ce`
  - training logs include `stop_signal/A1/{eligible_seq_count,branch_count,weight_mean,p_stop_mean,p_cont_mean,margin_mean}`
  - existing parsing / detection eval families remain the downstream readout surface
