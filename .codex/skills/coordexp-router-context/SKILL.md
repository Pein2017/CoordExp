---
name: coordexp-router-context
description: Use when CoordExp work needs current repo routing, docs/spec/code entrypoints, historical evidence boundaries, broad module maps, or guidance on RTK, raw shell, and Serena navigation.
---

# CoordExp Router Context

Use this as the compact routing layer for CoordExp. It replaces the old split between current codebase navigation, research-history context, broad maps, RTK selection, and Serena navigation.

## Mode Selector

- `current-route`: find current docs, configs, code entrypoints, tests, or artifact contracts.
- `history-pack`: connect current behavior to progress notes, memories, rollout summaries, or benchmark provenance.
- `map`: give a module/config/artifact map before returning to a narrow task.
- `navigation-tools`: choose between `rtk`, raw shell, and Serena.

Exact leaf skills win over this router: use `coordexp-infer-eval-workflow` for launch/repair/eval artifacts, `coordexp-public-data-provenance` for `public_data`, `model-diagnosis` for abnormal behavior, `model-innovation-risk-audit` for pre-launch trust gates, `audit-review` for findings-first audits, and `worktree-feature-loop` for isolation.

## Authority Spine

Use current repo truth in this order:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. relevant domain docs under `docs/`
5. `openspec/specs/` only for stable compatibility-sensitive contracts
6. `openspec/changes/<active-change>/` only when explicitly in scope
7. `progress/` only for history, diagnostics, benchmark evidence, or design derivation

Start with `docs/AGENT_INDEX.md` and `docs/catalog.yaml` before broad source search. When docs and progress disagree, answer current behavior from `docs/` and use `progress/` only to explain how the project got there.

## Current Route Loop

1. Name the exact surface: config, script, artifact root, metric, symbol, or doc claim.
2. Open the relevant docs/catalog route before source search.
3. Resolve config/schema ownership before code changes.
4. Trace the smallest useful path: JSONL/image -> config -> loader -> dataset/collator -> trainer/infer/eval -> artifact/metric.
5. Narrow with `rg` or `rtk grep`; use Serena only after candidate Python files are known.
6. Validate with the smallest check named by docs, specs, tests, or artifact manifests.
7. Update docs/specs only when behavior, schema, artifact names, metric semantics, entrypoints, or recommended workflows change.

## Current High-Signal Surfaces

- Stage-1 canonical compact teacher forcing: `configs/stage1/detection_teacher_forcing/`, `src/detection/runtime.py`, `src/detection/template.py`, `src/detection/objective.py`, `src/detection/loss.py`, `src/training/surfaces.py`, `src/training/pipelines/`.
- Stage-2 rollout correction: `configs/stage2/rollout_correction/`, `src/trainers/stage2_rollout_correction.py`, `src/trainers/rollout_correction/`, `src/training/stage2/`, `src/trainers/stage2_rollout_runtime.py`, `src/trainers/teacher_forcing/`, `src/launchers/stage2_vllm_server.py`.
- Infer/eval: `configs/infer/pipeline.yaml`, `configs/eval/detection.yaml`, `configs/postop/confidence.yaml`, `src/infer/pipeline.py`, `src/infer/runtime.py`, `src/infer/backend.py`, `src/infer/backend_sync.py`, `src/infer/backend_vllm_server.py`, `src/infer/artifacts.py`, `src/eval/detection.py`, `src/eval/orchestration.py`, `src/eval/artifacts.py`.
- Data/geometry: `docs/data/`, `src/datasets/geometry.py`, `src/datasets/dense_caption.py`, `src/datasets/builders/jsonlines.py`, `src/detection/template.py`, `src/common/detection_sequence.py`, `src/common/detection_compact_rows.py`.
- Artifacts/provenance: `docs/ARTIFACTS.md`, `src/bootstrap/`, `src/metrics/events.py`, `src/infer/artifacts.py`, `src/eval/artifacts.py`.

Treat archived recursive-detection configs under `configs/archive/detection_scene_clean_break/stage1/` and retired rollout-matching specs as historical comparators, not current public routes.

## History Pack Contract

For current-vs-historical reads, produce:

- question;
- current contract and authoritative docs/specs;
- current code/config/artifact handles;
- historical evidence with scope labels;
- mechanism read, explicitly labeled as inference;
- counterevidence, stale handles, repaired runs, missing baselines, or artifact-validity caveats;
- decision: `enough`, `one more probe`, `do not interpret yet`, `archive or pause`, or `hand off`;
- 2-5 targeted search seeds for continuation.

Search `.codex/memories/MEMORY.md` only when prior session context is relevant. Open at most 1-3 relevant rollout summaries or memory files before deciding whether more history is needed.

## Tool Choice

- Use `rtk` when output is noisy and a compact summary is enough: broad search, docs reads, git summaries, tests, logs, and file discovery.
- Use raw shell for exact stdout, machine-readable JSON/YAML, narrow `sed` reads, delicate quoting, or tiny commands.
- Use Serena for Python symbol overview, references, and precise symbolic edits after narrowing with `rg`/`rtk`.
- Do not run Serena repo-wide pattern scans with `relative_path` unset or `"."`.

## References

Open only when needed:

- `references/verification-matrix.md`: close-the-loop checks by surface.
- `references/grep-seeds.md`: current search seeds.
- `references/benchmark-interpretation.md`: score comparison, keep/drop, checkpoint-selection, and paused-direction evidence fields.
