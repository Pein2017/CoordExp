# Repo Hygiene Constitution (CoordExp)

This document defines **where things go**, **what gets tracked**, and **how one-shot work gets promoted** so the repo stays reproducible and paper-ready.

## 1) Folder roles (authoritative)

Tracked, reviewable:
- `src/`: importable library code (training/infer/eval). No ad-hoc experiments here.
- `configs/`: YAML-first experiments. Any result worth keeping must be reproducible from a config.
- `scripts/`: stable entrypoints + small maintained utilities (thin wrappers, minimal logic).
- `tests/`: contracts + regressions (especially geometry + template invariants).
- `docs/`: runbooks and user-facing documentation.
- `public_data/`: dataset tooling and public artifacts (builders, validators, exporters).
- `openspec/`: design/contract governance (treat as source-of-truth for behavior changes).
- `progress/`: short research notes / decision logs (keep concise; link to `docs/` when something becomes stable).

Not tracked (workspace artifacts; safe to delete):
- `output/`: training/infer outputs (checkpoints, reports).
- `tb/`: TensorBoard event files.
- `vis_out/`: visualization outputs.
- `temp/`, `tmp/`: one-off scratch work.
- `model_cache/`: local cached models (large; delete only when you accept re-download/rebuild).

## 2) Config-first rule

If it changes model behavior or evaluation, it must be expressible via `configs/`:
- No “magic flags” hidden in scripts.
- Prefer adding a YAML field over adding a new CLI arg.
- Any experiment run should log `config_path` + resolved config dump + git SHA.

## 3) One-shot script lifecycle

### Stage A: scratch (allowed, untracked)
Put exploratory scripts in `temp/YYYY-MM-DD_<topic>/` and assume they can be deleted.

### Stage B: promoted utility (maintained)
If you use it **twice**, promote it into:
- `scripts/tools/` for utilities, or
- `scripts/analysis/` for analysis/report scripts.

Requirements to promote:
- Deterministic defaults (explicit seeds when sampling).
- Clear IO contract (`--input`, `--output`, or well-documented env vars).
- Writes outputs under `output/` or `vis_out/` (never beside code).

### Stage C: library (stable API)
If it becomes part of training/infer/eval, move logic into `src/` and add tests.

## 4) Deprecation & removal

When replacing an entrypoint:
1) Update docs/configs to the new canonical path.
2) Remove the old wrapper, or leave a short pointer stub **for one release window**.
3) Prefer “delete + git history” over keeping dead code indefinitely.

## 5) Reproducibility minimum bar

For any run you might cite:
- Encode hypothesis in `training.run_name` (dataset, base ckpt, decode, key knobs, seed).
- Log the exact git SHA and config used.
- Keep evaluation scripts deterministic and versioned.

## 6) Contract guardrails (do not violate)

- Preserve geometry: never drop/reorder coords; use `src/datasets/geometry.py`.
- Training uses `do_resize: false` unless explicitly justified in config/docs.
- Maintain Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model files (e.g., `modeling_qwen3_vl.py` is off-limits).

