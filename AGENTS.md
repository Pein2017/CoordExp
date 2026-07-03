# Agent Guide - CoordExp

> One shared source for Claude, Codex, Gemini, and other coding agents.
> Distilled from the prior root/Codex guides, `.codex/RTK.md`, current repo docs,
> recent git history, and workflow memory. Tool-specific mechanics belong in
> short parentheticals; shared behavior belongs here.

## Precedence And Posture

- User request > nested instruction file > this file > personal/global defaults.
- Make the smallest reversible change that handles the request; inspect relevant files before editing.
- State assumptions when requirements are underspecified. Ask only for choices that affect research meaning, high cost, destructive cleanup, external publication, security/privacy, or irreversible compatibility.
- Prefer implementation plus verification over extended planning. For long work, set an explicit objective, scope boundary, and stop condition (Codex: `/goal`; no token budget unless asked).
- Use subagents only for independent lanes such as audits, subsystem exploration, disjoint implementation slices, or verification tracks. Respect platform limits (Codex: 6 active).
- Give direct verdicts when the user asks for a decision. For cross-agent prompts, set background and purpose, require evidence, keep read-only when appropriate, and leave room for each agent's own judgment.

## CoordExp Authority

- CoordExp is a grounding/detection research stack for reproducible data preparation, Stage-1 training, Stage-2 rollout-aware training, inference, evaluation, and artifacts.
- Route current-behavior reads through `docs/AGENT_INDEX.md` and `docs/catalog.yaml`, then `docs/PROJECT_CONTEXT.md` -> `docs/SYSTEM_OVERVIEW.md` -> `docs/IMPLEMENTATION_MAP.md` -> relevant `docs/` domain router.
- Use `openspec/specs/` only for stable compatibility-sensitive contracts: training/eval behavior, config schemas, loss semantics, artifact names, and normative metrics.
- Treat `progress/` as a legacy historical/evidence archive for unmigrated diagnostics, audits, benchmark context, and design derivation; do not treat it as current behavior authority.
- Prefer `research/` for new research ideas, investigations, mechanisms, interpretation, negative results, and continuation context. It is not authority for current coding, operator, schema, artifact, metric, or training/eval behavior.
- For deeper engineering posture, consult `docs/AGENT_ENGINEERING_CONSTITUTION.md` if still present and relevant.

## Hard Rules

- Config-first: prefer YAML/schema changes over new stable CLI flags.
- Default training surface: offline-prepared single-dataset JSONL; runtime fusion configs are legacy or experimental.
- Stage-1 baseline SFT and Stage-2 rollout-aware training are active first-class surfaces.
- Preserve image/geometry alignment end to end; never drop or reorder coordinates.
- Route bbox math through `src/datasets/geometry.py` unless editing detection serialization code.
- Training uses `do_resize=false`; do not introduce silent resizing.
- Do not edit upstream HF model files, including `modeling_qwen3_vl.py`.
- Do not add hidden agent memory stores, self-modifying persistence, or auto-commit watchers for local agent state.

## Navigation And Tools

- Docs route first, then narrow with `rg`, `rtk`, structured parsers, exact reads, or current artifact inspection.
- Inspect the smallest code/config/artifact surface that answers the task.
- CodeGraph: broad symbol/file/call maps when indexed for the exact worktree; keep docs/specs/configs/artifacts authoritative.
- Serena: Python symbol exploration, references, diagnostics, and precise edits after narrowing.
- Raw shell: exact stdout, JSON/YAML, NUL output, delicate quoting, pipelines, and small machine-readable checks.
- RTK: compact noisy output when exact stdout is not required, e.g. `rtk git status --short --branch`, `rtk git diff --stat`, `rtk grep "<pattern>" <path>`, `rtk pytest <target>`. If surprising, rerun with raw command or `rtk proxy`; see `.codex/RTK.md`.
- In this checkout, shells normally start in `ms`; use plain `python`, `pytest`, and repo entrypoints. If another checkout lacks that setup, follow its local guide.

## Change Governance

- Update docs when changing stable defaults, entrypoints, config schemas, artifact names, metric semantics, or recommended workflows.
- Use OpenSpec for stable compatibility-sensitive contracts, not ordinary experiment planning or implementation checklists.
- Keep code where future contributors would look first: contracts in `src/common/`, data/geometry in `src/datasets/`, config schema in `src/config/`, training in `src/training/` or `src/trainers/`, inference in `src/infer/`, eval in `src/eval/`, maintained utilities in `scripts/`.
- Prefer strict current schemas and fail-fast behavior. Keep compatibility shims visibly separate from canonical behavior.
- Do not add production dependencies, services, credentials, expensive jobs, destructive cleanup, or data deletion without explicit approval.

## Research And Git Routine

- Dirty worktrees are expected. Inspect state before broad edits, staging, committing, merging, or cleanup.
- Use worktrees for independent research directions or risky branch work; keep runs, notes, and artifacts isolated by direction.
- GPU use is opportunistic: all GPUs are available to every thread/task, concurrent use is expected, and the only constraint is managing GPU memory to avoid OOM.
- Keep experiments config-first and artifact-backed: record config, checkpoint, artifact root, parse/drop counters, metric files, and evidence scope before interpretation.
- Put one-off probes under `temp/`; promote repeated utilities to `scripts/tools/` or `scripts/analysis/` only when they become reusable.
- Keep outputs/checkpoints/rollout dumps/visual galleries/TensorBoard/raw logs under `outputs/` or documented external artifact roots. Do not delete them automatically.
- Prefer `research/` for new research writing. Use `progress/` only for old evidence that has not migrated yet or when current docs explicitly point there. Promote only stable current behavior into `docs/`.
- Track `.codex/skills/` only when a skill encodes non-obvious repo workflow. Do not track `.codex/memories/`, sessions, logs, plugin caches, auth, or app state.
- Stage narrowly by explicit path when dirty. Use small logical commits when requested. Run relevant checks and `git diff --cached --check` before commit. Confirm before retrying interrupted or unrequested pushes.

## Verification, Evidence, Reporting

- Every code/config/data/docs-contract/workflow change needs a verification path: targeted test, smoke run, config parse, artifact/manifest check, metric check, replay, residue grep, or explicit reason skipped.
- Narrow checks first; broaden only when shared contracts or user-facing workflows changed.
- Before production training, deployment, release, or benchmark claims, verify geometry/image alignment, prompt/template compatibility, config resolution, cache/packing, loss semantics, metric scope, artifact completeness, eval validity, and rollback/restore.
- For docs-only changes, verify referenced paths and links enough to avoid stale handles.
- Attach concrete handles to claims: paths, symbols, config keys, commands, artifact roots, metrics, or minimal I/O examples.
- Label evidence scope (`tiny`, `smoke`, `val200`, `limit=200`, `proxy`, `partial`, `full`) and never present partial evidence as full validation.
- When diagnosing model behavior, inspect the exact artifact tree the user names before explaining from config theory or memory.
- Preserve unrelated user work. Never revert changes you did not make unless explicitly asked.
- Final responses should include changed files, verification commands, skipped checks, residual risks, and useful next actions. Reviews should lead with findings; handoffs should include objective, current state, exact paths, commands, evidence scope, risks, and continuation seeds.
