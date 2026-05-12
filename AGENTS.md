# Codex Agent

## Mission
- Evolve CoordExp into a general grounding/detection research stack; favor reproducible, paper-ready workflows and compatibility-preserving changes.
- Follow precedence: `docs/PROJECT_CONTEXT.md` -> `docs/SYSTEM_OVERVIEW.md` -> `docs/IMPLEMENTATION_MAP.md` -> relevant domain docs under `docs/` -> `openspec/specs/` for legacy/stable-contract reference -> `openspec/changes/<active-change>/` only when an active change is explicitly in scope -> `progress/`.
- Use `docs/AGENT_INDEX.md` and `docs/catalog.yaml` for routing; use `progress/` only for historical context, diagnostics, or empirical evidence.

## Defaults
- Offline-prepared single-dataset JSONL is the default training surface; keep runtime transforms minimal and reproducible.
- Treat both Stage-1 baseline SFT and Stage-2 rollout-aware training as active first-class surfaces; fusion-config training remains legacy/experimental.
- Packing, cache reuse, and manifest/artifact completeness are primary operational levers.

## Guardrails
- Config-first; avoid new CLI flags; keep Qwen3-VL chat-template compatibility and current artifact contracts.
- Preserve geometry and image alignment end-to-end (never drop/reorder coords); use `src/datasets/geometry.py`; training uses `do_resize=false`.
- Do not edit upstream HF model files like `modeling_qwen3_vl.py`.
- Treat OpenSpec as downgraded governance. Use it only for stable, compatibility-sensitive contracts such as training/eval behavior, config schemas, loss semantics, artifact names, or normative metric semantics. Do not start or expand OpenSpec work for ordinary feature planning, experiment management, or implementation checklists when repo-local super-power plans and docs are sufficient.

## Workflow
- Explain decisions only when they affect correctness/reproducibility/eval validity/maintainability.
- State assumptions when underspecified; choose the smallest viable change; do not invent metrics/results.
- Fail fast on unexpected behavior, resolve root causes, and verify on the narrowest realistic surface first (targeted tests, caches, or artifacts before broad suites).
- Use repo-local super-power specs/plans for branch-specific implementation:
  code surfaces, config contracts, tests, smoke commands, artifacts, and
  merge-readiness evidence. Do not make super-power plans responsible for
  future research gates that require later evidence or benchmark artifacts.
- Use repo docs and checked-in progress notes for research memory,
  interpretation, decision logs, and claims; keep executable truth and
  reproducibility evidence in the repo, configs, and run artifacts.

## Research Management
- Use four repo-local surfaces with clear ownership:
  - `docs/` is the durable reference layer: motivations, interpretation, stable decisions, methods/protocols, and claims that have graduated from active work.
  - `super-power` plans/specs are the execution brain: detailed implementation stages, file-level tasks, command plans, verification checklists, and handoff notes.
  - `progress/` is the historical evidence layer: diagnostics, benchmark notes, and measured-run artifacts.
  - The repo is reproducibility truth: code, configs, tests, commands, manifests, artifact paths, checked-in docs, and checked-in `progress/` evidence.
- Keep redundancy low. Do not copy detailed implementation checklists into `docs/` when a repo-local super-power plan owns them.
- Prefer a promotion ladder for new work:
  - idea or brainstorm -> repo note or draft doc;
  - actionable small task -> super-power plan only if it needs coarse tracking;
  - implementation details -> super-power plan;
  - measured result -> repo artifact plus `progress/` note when warranted;
  - durable interpretation -> checked-in doc or final memo;
  - stable current behavior -> `docs/`;
  - stable compatibility contract -> OpenSpec only if truly needed.

## Routine
- Start a new research direction in a repo note or doc unless it is already an obvious one-command or one-file fix.
- Before coding on nontrivial work, locate or create the repo-local super-power spec/plan that owns detailed execution.
- Create or update a coarse tracking note only when there is coarse progress, a blocker, a gate transition, or a project-level decision worth separate tracking.
- After running experiments, record exact scope (`tiny`, `val200`, `limit=200`, full-val, proxy, raw-text, coord-token, etc.), configs, checkpoints, artifact roots, parse/drop counters, and metric files before writing interpretation.
- Promote results in order: artifact/manifests first, then `progress/` or benchmark/diagnostic notes when useful, then checked-in docs or final memo, then `docs/` only if behavior becomes current stable guidance.

## Style
- Keep management records short, link-rich, and scoped. Prefer one canonical entry plus links over repeated summaries.
- Use explicit statuses such as `Inbox`, `Triaged`, `Ready for Execution`, `Running`, `Analyzing`, `Concluded`, `Archived`, `Blocked`, and `Superseded` rather than prose-only state.
- Always distinguish hypothesis, plan, result, interpretation, and stable contract.
- Do not present `val200`, proxy, tiny, or partial-run evidence as full validation.
- When workflow notes disagree, preserve the repo as executable truth and keep docs/progress notes as navigation and interpretation layers.

## Repo Safety
- Never run destructive cleanup commands unless explicitly asked.
- Dirty changes from parallel work are expected; isolate your edits and do not revert unrelated work.
- Prefer small, logically scoped commits during large refactors or incident response.
- Do not add hidden agent memory stores, portable self-modification workflows, or any other agent-only persistence layer to this workspace.

## Navigation
- Use `coordexp-codebase` for entrypoints and workflow pointers.
- Use `coordexp-research-context` for broad background, read-order, and historical Stage-2 context.
- Treat `superpowers` as plugin-managed in this workspace: the active source of truth is the enabled `superpowers@openai-curated` plugin, not a repo-local vendored copy under `./.codex/skills/`.
- For current `superpowers` provenance or upgrade checks, inspect [`./.codex/config.toml`](./.codex/config.toml) for `[plugins."superpowers@openai-curated"]`, then inspect the cached plugin manifest at `./.codex/plugins/cache/openai-curated/superpowers/*/.codex-plugin/plugin.json` for the current packaged version and upstream repository.
- Prefer `rtk` for noisy shell workflows first: broad repo scans, multi-hit `rg`, `git`, logs, tests, long docs/prose reads, and other multi-line command output where compact summaries help.
- Do not force `rtk` into exact-output workflows. Prefer raw commands for narrow line reads (for example `sed -n`, `nl -ba ... | sed -n`), machine-readable stdout, delicate quoting, or commands where `rtk` could obscure exact interpreter/environment binding.
- When a command already depends on a project-specific environment wrapper, keep that wrapper under `rtk` instead of dropping it. In this repo, tests should prefer `rtk conda run -n ms python -m pytest ...` over bare `rtk pytest ...`.
- Serena MCP is available beyond Python whenever the target file type is supported and symbol-aware navigation or editing would reduce ambiguity or risk.
- For any `*.py` file, Serena MCP is mandatory for exploration and editing.
- For non-Python code, prefer Serena MCP when working in large files, doing cross-reference tracing, symbol-level edits, or nontrivial refactors. For plain-text or non-symbolic files such as Markdown, YAML, JSON, or exact line-based inspections, direct shell reads and `apply_patch` are usually the better fit.
- For code work, use `rg`/`rtk` first to narrow candidate files or directories, then switch to Serena MCP when symbol-level understanding or editing is useful.

## Environment
- Repo root: `.`
- Use `conda run -n ms python ...` for tests.
- Use `temp/` for one-off debug artifacts; clean up when done.
