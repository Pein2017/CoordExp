---
name: coordexp-router-context
description: Use when CoordExp work needs current repo routing, docs/spec/code entrypoints, historical evidence boundaries, broad module maps, or guidance on CodeGraph, RTK, raw shell, and Serena navigation.
---

# CoordExp Router Context

Use this as the compact routing layer for CoordExp. It replaces the old split between current codebase navigation, research-history context, broad maps, CodeGraph routing, RTK selection, and Serena navigation.

## Mode Selector

- `current-route`: find current docs, configs, code entrypoints, tests, or artifact contracts.
- `history-pack`: connect current behavior to progress notes, memories, rollout summaries, or benchmark provenance.
- `map`: give a module/config/artifact map before returning to a narrow task.
- `navigation-tools`: choose between CodeGraph, Serena, `rtk`, and raw shell by phase.

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
4. For code-heavy questions, use CodeGraph as the broad map when a correct local index exists, then switch to Serena once Python files or symbols are known.
5. Trace the smallest useful path: JSONL/image -> config -> loader -> dataset/collator -> trainer/infer/eval -> artifact/metric.
6. Use `rg`, `rtk grep`, raw shell, or structured parsers for config keys, YAML inheritance, docs/spec clauses, JSONL/artifact fields, metric names, and exact stdout.
7. Validate with the smallest check named by docs, specs, CodeGraph impact results, tests, or artifact manifests.
8. Update docs/specs only when behavior, schema, artifact names, metric semantics, entrypoints, or recommended workflows change.

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

- Route by phase, not by habit: CodeGraph answers "where should I look?", Serena answers "what exactly is this Python symbol and who depends on it?", and shell/tests answer "what is the exact current state?" Do not maximize MCP call count; maximize the value of the right tool at the right phase.
- Use CodeGraph for first-pass repository-scale orientation before token-heavy exploration: symbol search, file/package maps, call chains, grouped source context, and impact radius.
- Use CodeGraph CLI for index lifecycle and reproducible setup checks:
  - `codegraph init -i`
  - `codegraph status`
  - worktree-local index repair/rebuild
- Use CodeGraph MCP for exploration after the index exists:
  - start with `codegraph_explore` when one capped call can return grouped source/context;
  - then use `codegraph_search`, `codegraph_impact`, `codegraph_callers`, `codegraph_callees`, `codegraph_files`, or `codegraph_status` as needed.
- In linked worktrees, initialize CodeGraph inside the exact worktree and pass `projectPath=/absolute/worktree/path` to CodeGraph MCP calls whenever there is any ambiguity. If CodeGraph results mention a different worktree, stop relying on them until `codegraph init -i` and `codegraph status` confirm the local index.
- Use CodeGraph CLI as MCP fallback only for exploration commands. Prefer CLI `codegraph impact <symbol> --depth 2 --json` over `affected <files>` for CoordExp test selection.
- Treat CodeGraph as a fast tree-sitter graph, not a semantic type checker: it is strong for deterministic structure, imports, name-based calls, and local graph traversal; it is weaker for ambiguous method names, dynamic dispatch, config semantics, and true LSP-level reference precision.
- After edits that change indexed source/YAML, refresh the exact worktree index before relying on graph results. Use `codegraph sync <absolute-worktree-path>` when available, otherwise rerun `codegraph init -i` from that worktree.
- Use Serena MCP for precise Python/LSP work after narrowing with docs, CodeGraph, `rg`, or `rtk`: activate the exact project/worktree path, then use symbol overview, exact references, body reads, declarations/implementations, diagnostics, and precise symbolic edits.
- Expected handoff for nontrivial Python work: CodeGraph or `rg` to locate the area -> Serena `get_symbols_overview`/`find_symbol`/`find_referencing_symbols` for narrowed Python semantics -> patch or Serena symbolic edit -> narrow verification.
- Prefer Serena over another broad CodeGraph body read when the next step is a Python edit, reference-sensitive claim, inheritance/override check, or diagnostic-risk check.
- CodeGraph-heavy behavior is acceptable for read-only review when one or two capped `codegraph_explore` calls provide enough evidence and no Python edit or precise reference claim follows.
- Use `rtk` when output is noisy and a compact summary is enough: broad search, docs reads, git summaries, tests, logs, and file discovery.
- Use raw shell for exact stdout, machine-readable JSON/YAML, narrow `sed` reads, delicate quoting, or tiny commands.
- Use `rg`/raw parsers over CodeGraph/Serena for exact literal search in configs, docs, OpenSpec, progress notes, JSONL, logs, metrics, and artifact manifests.
- Do not run Serena repo-wide pattern scans with `relative_path` unset or `"."`.

## CodeGraph Patterns

- Broad code task: verify the exact worktree index, then MCP `codegraph_explore` with `projectPath` and `maxFiles` capped -> switch to Serena for exact Python symbols.
- Risky symbol change: MCP `codegraph_impact` with `projectPath`, or CLI `codegraph impact <ClassOrFunction> --depth 2 --json` from the worktree -> list affected files/tests -> inspect exact references and diagnostics with Serena -> run targeted tests.
- Function flow: MCP `codegraph_callers`/`codegraph_callees` with `projectPath`, or CLI `codegraph callers <function> --json`/`codegraph callees <function> --json`; if names are ambiguous, use Serena with an exact `Class/method` name path.
- Package map: MCP `codegraph_files` with `projectPath`, or CLI `codegraph files --json --filter src/<area>` from the worktree before opening large files.
- Config-driven change: start with docs/catalog and `rg` over `configs docs openspec src tests`; use CodeGraph only for the Python consumers of resolved config keys.

## References

Open only when needed:

- `references/verification-matrix.md`: close-the-loop checks by surface.
- `references/grep-seeds.md`: current search seeds.
- `references/benchmark-interpretation.md`: score comparison, keep/drop, checkpoint-selection, and paused-direction evidence fields.
