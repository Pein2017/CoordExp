---
doc_id: docs.standards.repo-hygiene
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Source layout, promotion rules, and reproducibility hygiene.
updated: 2026-10-02
---

# Repo Hygiene Constitution (CoordExp)

This document defines maintained source layout, config conventions, and utility promotion. Physical output ownership and scratch placement are owned by [the output storage policy](../OUTPUT_STORAGE_POLICY.md); current research ownership is routed from [the research index](../../research/index.md).

## 1) Folder roles

Maintained source:
- `src/`: importable library code (training/infer/eval). No ad-hoc experiments here.
- `configs/`: YAML-first experiments. Any result worth keeping must be reproducible from a config.
- `scripts/`: stable entrypoints + small maintained utilities (thin wrappers, minimal logic).
- `tests/`: contracts + regressions (especially geometry + template invariants).
- `docs/`: runbooks and user-facing documentation.
- `public_data/`: dataset tooling and public artifacts (builders, validators, exporters).
- `openspec/`: design/contract governance (treat as source-of-truth for behavior changes).

Workspace artifacts and disposable scratch are ignored, but their physical
owner and location are defined only by [the output storage policy](../OUTPUT_STORAGE_POLICY.md).
Do not infer a backup destination or retention rule from `.gitignore`.

**Legacy and local state:**

- `progress/`: legacy historical evidence; do not add new records.
- `model_cache/`: machine-local cached models.
- `external/`, `.worktrees/`: local checkouts and worktrees, not vendored
  project source.
- `.codex/` except `.codex/skills/`: local agent runtime state. Memories,
  sessions, plugin caches, logs, auth, and app state must stay local-only.
- `.claude/`, `.gemini/`, `.history/`, `.gitnexus/`, `.vendor/`: local agent
  and workstation state.
- root-level credential or cookie files such as `baidu_net_cookie.txt` and
  `github_personal_token.txt`.

## 2) Config-first rule

If it changes model behavior or evaluation, it must be expressible via `configs/`:
- No “magic flags” hidden in scripts.
- Prefer adding a YAML field over adding a new CLI arg.
- Any experiment run should log `config_path` + resolved config dump + git SHA.

## 3) One-shot script lifecycle

### Stage A: disposable work
Use the task owner's scratch location as defined by the [output storage policy](../OUTPUT_STORAGE_POLICY.md).

### Stage B: promoted utility (maintained)
If you use it **twice**, promote it into:
- `scripts/tools/` for utilities, or
- `scripts/analysis/` for analysis/report scripts.

Requirements to promote:
- Deterministic defaults (explicit seeds when sampling).
- Clear IO contract (`--input`, `--output`, or well-documented env vars).
- Route generated files through the [output storage policy](../OUTPUT_STORAGE_POLICY.md), never beside maintained code.

### Stage C: library (stable API)
If it becomes part of training/infer/eval, move logic into `src/` and add tests.

## 4) Research and evidence

The root `research/` tree preserves earlier OKF-style source material and routes
current maintained research to its owner. See [its index](../../research/index.md).
`progress/` is a legacy historical archive; do not add new records there.
Physical artifact ownership remains defined by the [output storage policy](../OUTPUT_STORAGE_POLICY.md).

## 5) Config lifecycle

Keep configs in lifecycle folders:

- `prod/`: runnable current or comparator profiles that may be launched again.
- `smoke/`: cheap validation profiles for a current surface.
- `ablation/`: active or recently interpretable research variants.
- `negative/`: intentional failure fixtures or preflight guards.
- `profiles/`: legacy Stage-1 SFT profiles that are still documented.
- `analysis/` and `bench/`: historical or report-building configs, not default
  training authoring examples.

When a config stops being useful:

1. Move any useful conclusion to its existing research or documentation owner.
2. Remove the config if no current docs/tests reference it.
3. Keep a `negative/` config only when a test or preflight uses it to enforce a
   failure contract.

## 6) Agent and ops policy

Tracked agent assets should be capability surfaces, not state dumps:

- Track `.codex/skills/` when the skill encodes non-obvious repo workflow.
- Do not track `.codex/memories/`, sessions, logs, plugin caches, auth, app
  state, or auto-generated runtime bundles.
- Do not add auto-commit watchers for local agent memory. They mix source
  changes with private runtime state and make repository history noisy.
- Put workstation automation in `ops/`, not `scripts/`, unless it is a
  maintained CoordExp training/infer/eval entrypoint.

## 7) Deprecation & removal

When replacing an entrypoint:
1) Update docs/configs to the new canonical path.
2) Remove the old wrapper, or leave a short pointer stub **for one release window**.
3) Prefer “delete + git history” over keeping dead code indefinitely.

## 8) Reproducibility minimum bar

For any run you might cite:
- Encode hypothesis in `training.run_name` (dataset, base ckpt, decode, key knobs, seed).
- Log the exact git SHA and config used.
- Keep evaluation scripts deterministic and versioned.

## 9) Contract guardrails (do not violate)

- Preserve geometry: never drop/reorder coords; use `src/data/geometry.py`.
- Golden rule: all training and evaluation use offline-preprocessed images, and runtime vision processors must not resize them.
- Training uses `do_resize: false` unless explicitly justified in config/docs.
- Maintain Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model files (e.g., `modeling_qwen3_vl.py` is off-limits).
