---
doc_id: docs.standards.repo-hygiene
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Repo layout, promotion rules, and reproducibility hygiene.
updated: 2026-05-20
---

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
- `outputs/`: training/infer/eval outputs (checkpoints, reports, JSONL
  artifacts). **Do not delete automatically**; treat as valuable experiment
  artifacts and the canonical Baidu Netdisk sync surface.
- `output/`, `output_remote/`: legacy or transitional output roots. Do not
  create new canonical workflows here. `output_remote/` may exist during local
  migrations, but should not be part of durable repo structure.
- `tb/`: TensorBoard event files.
- `vis_out/`: visualization outputs.
- `temp/`, `tmp/`: one-off scratch work.
- `model_cache/`: local cached models (large; delete only when you accept re-download/rebuild).
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

## 4) Research asset lifecycle

Use four asset classes:

- **Core asset**: current library code, schema, config, docs, tests, and small
  provenance manifests. Keep tracked and routed from `docs/` or
  `docs/catalog.yaml`.
- **Historical reference**: dated design notes, audits, benchmark summaries,
  and failure analysis. Keep under `progress/` with a router entry when it is
  worth rediscovering.
- **Artifact**: checkpoints, rollout dumps, eval JSONL, large per-image tables,
  visual galleries, TensorBoard, and raw run logs. Keep under `outputs/` or a
  documented external artifact location; do not track by default.
- **Scratch**: one-off probes, temporary scripts, local staging data, copied
  logs, and debugging residue. Keep under `temp/` and delete after use.

`progress/diagnostics/artifacts/` may contain small curated evidence copies, but
avoid tracking large images, full per-image eval tables, launch logs, or copied
run directories there. Prefer a short `README.md`, metrics summaries, and links
or repo-relative pointers to the artifact root under `outputs/`.

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

1. Move its conclusion into `progress/`.
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

Retention note:
- `outputs/` is considered a persistent workspace artifact root
  (checkpoints + logs). Do not delete it via cleanup scripts.

## 9) Contract guardrails (do not violate)

- Preserve geometry: never drop/reorder coords; use `src/datasets/geometry.py`.
- Golden rule: all training and evaluation use offline-preprocessed images, and runtime vision processors must not resize them.
- Training uses `do_resize: false` unless explicitly justified in config/docs.
- Maintain Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model files (e.g., `modeling_qwen3_vl.py` is off-limits).
