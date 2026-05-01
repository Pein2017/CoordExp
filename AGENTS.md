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
- Treat OpenSpec as downgraded governance. Use it only for stable, compatibility-sensitive contracts such as training/eval behavior, config schemas, loss semantics, artifact names, or normative metric semantics. Do not start or expand OpenSpec work for ordinary feature planning, experiment management, or implementation checklists when `super-power`, Notion, Linear, and repo docs are sufficient.

## Workflow
- Explain decisions only when they affect correctness/reproducibility/eval validity/maintainability.
- State assumptions when underspecified; choose the smallest viable change; do not invent metrics/results.
- Fail fast on unexpected behavior, resolve root causes, and verify on the narrowest realistic surface first (targeted tests, caches, or artifacts before broad suites).

## Research Management
- Use four surfaces with clear ownership:
  - Notion is research memory: motivations, research-unit briefs, analysis notes, decision logs, methods/protocols, claims, interpretation, and final conclusions.
  - Linear is the coarse progress dashboard: active workstreams, big gates, blockers, phase/status, and links to Notion, super-power plans, repo branches, configs, and artifacts.
  - `super-power` plans/specs are the execution brain: detailed implementation stages, file-level tasks, command plans, verification checklists, and handoff notes.
  - The repo is reproducibility truth: code, configs, tests, commands, manifests, artifact paths, checked-in docs, and checked-in `progress/` evidence.
- Keep redundancy low. Do not copy detailed implementation checklists into Linear or Notion when a repo-local super-power plan owns them.
- Prefer a promotion ladder for new work:
  - idea or brainstorm -> Notion inbox/research-unit note;
  - actionable small task -> Linear issue only if it needs coarse tracking;
  - implementation details -> super-power plan;
  - measured result -> repo artifact plus `progress/` note when warranted;
  - durable interpretation -> Notion claim/decision/final memo;
  - stable current behavior -> `docs/`;
  - stable compatibility contract -> OpenSpec only if truly needed.

## Notion
- Treat the Notion `CoordExp` page as the global project/research entry.
- Prefer databases and filtered views over many top-level pages. Top-level Notion pages should stay few and durable.
- Use Research Units as the canonical record for ideas, ablations, investigations, and workstreams. Standalone experiment pages are supporting briefs or final memos, not competing sources of truth.
- Use Decision Log only for durable choices with rationale and consequence.
- Use Claims Ledger only for testable claims. Claims remain `Untested` until scoped evidence exists; every supported claim needs metric scope and artifact/config/checkpoint references.
- Use Methods & Protocols only for reusable procedures, not one-off execution checklists.
- Do not paste large generated artifacts into Notion; link exact repo paths, output roots, manifests, and metrics.

## Linear
- Keep Linear intentionally small. It is not the implementation task engine for CoordExp.
- Use Linear projects only for bounded workstreams with multiple gates or a final close condition. Do not create a project for every idea, prompt tweak, small bug, or implementation subtask.
- A research workstream should normally have:
  - one Linear project only when gate-level progress needs dashboard visibility;
  - one umbrella issue for big-picture progress and links;
  - at most one or two gate/blocker issues for high-risk checkpoints.
- Do not mirror every super-power plan item as a PEI issue. Renderer/parser/config/test/command details belong in the super-power plan unless they become independently blocked or decision-relevant.
- Linear issue descriptions should stay gate-level: purpose, current status, definition of done, blocker, and links to Notion/repo/super-power artifacts.
- Archive, cancel, or mark as superseded any Linear issues that only duplicate detailed super-power execution steps.

## Routine
- Start a new research direction in Notion first unless it is already an obvious one-command or one-file fix.
- Before coding on nontrivial work, locate or create the repo-local super-power spec/plan that owns detailed execution.
- Create or update Linear only when there is coarse progress, a blocker, a gate transition, or a project-level decision worth dashboard visibility.
- After running experiments, record exact scope (`tiny`, `val200`, `limit=200`, full-val, proxy, raw-text, coord-token, etc.), configs, checkpoints, artifact roots, parse/drop counters, and metric files before writing interpretation.
- Promote results in order: artifact/manifests first, then `progress/` or benchmark/diagnostic notes when useful, then Notion claims/decisions/final memo, then `docs/` only if behavior becomes current stable guidance.

## Style
- Keep management records short, link-rich, and scoped. Prefer one canonical entry plus links over repeated summaries.
- Use explicit statuses such as `Inbox`, `Triaged`, `Ready for Execution`, `Running`, `Analyzing`, `Concluded`, `Archived`, `Blocked`, and `Superseded` rather than prose-only state.
- Always distinguish hypothesis, plan, result, interpretation, and stable contract.
- Do not present `val200`, proxy, tiny, or partial-run evidence as full validation.
- When workflow tools disagree, preserve the repo as executable truth and keep Notion/Linear as navigation and interpretation layers.

## Repo Safety
- Never run destructive cleanup commands unless explicitly asked.
- Dirty changes from parallel work are expected; isolate your edits and do not revert unrelated work.
- Prefer small, logically scoped commits during large refactors or incident response.
- Do not add hidden agent memory stores, portable self-modification workflows, or any other agent-only persistence layer to this workspace.

## Navigation
- Use `coordexp-codebase` for entrypoints and workflow pointers.
- Use `coordexp-research-context` for broad background, read-order, and historical Stage-2 context.
- Prefer `rtk` for noisy shell workflows first: broad repo scans, multi-hit `rg`, `git`, logs, tests, long docs/prose reads, and other multi-line command output where compact summaries help.
- Do not force `rtk` into exact-output workflows. Prefer raw commands for narrow line reads (for example `sed -n`, `nl -ba ... | sed -n`), machine-readable stdout, delicate quoting, or commands where `rtk` could obscure exact interpreter/environment binding.
- When a command already depends on a project-specific environment wrapper, keep that wrapper under `rtk` instead of dropping it. In this repo, tests should prefer `rtk conda run -n ms python -m pytest ...` over bare `rtk pytest ...`.
- Serena MCP is available beyond Python whenever the target file type is supported and symbol-aware navigation or editing would reduce ambiguity or risk.
- For any `*.py` file, Serena MCP is mandatory for exploration and editing.
- For non-Python code, prefer Serena MCP when working in large files, doing cross-reference tracing, symbol-level edits, or nontrivial refactors. For plain-text or non-symbolic files such as Markdown, YAML, JSON, or exact line-based inspections, direct shell reads and `apply_patch` are usually the better fit.
- For code work, use `rg`/`rtk` first to narrow candidate files or directories, then switch to Serena MCP when symbol-level understanding or editing is useful.

## Model
- Allocate subagent model capacity by subtask type instead of using one fixed default.
- Use `gpt-5.4-mini` for pure information collection only: repo scans, file discovery, fact extraction, or status gathering with little synthesis or judgment.
- Use `gpt-5.4` with `medium` reasoning for bounded implementation, mechanical refactors, straightforward test updates, or execution tasks with clear acceptance criteria.
- Use `gpt-5.4` with `high` reasoning for the default frontier tier: debugging, code review, audit, nontrivial planning, cross-file reasoning, tradeoff analysis, and ambiguous implementation work.
- Use `gpt-5.4` with `xhigh` reasoning only for the hardest or highest-stakes work: deep audits, architecture/spec design, difficult root-cause analysis, research brainstorming, or any task where `high` is proving insufficient.
- When uncertain, prefer `gpt-5.4` with `high` rather than under-allocating; downgrade to `mini` only when the task is collection-only, and upgrade to `xhigh` only when extra depth is likely to change the answer.

## Environment
- Repo root: `.`
- Use `conda run -n ms python ...` for tests.
- Use `temp/` for one-off debug artifacts; clean up when done.
