# CoordExp Agent Guide

## Scope
- Applies to `./CoordExp`.
- Precedence: user request > nested `AGENTS.md` > this file > global/personal instructions.
- Keep this file repo-specific. Put reusable behavior in global instructions; add nested `AGENTS.md` only for subtree-specific commands or conflicting contracts.

## Project
- CoordExp is a grounding/detection research stack focused on reproducible training, inference, evaluation, and artifact workflows.
- Current reference order: `docs/PROJECT_CONTEXT.md` -> `docs/SYSTEM_OVERVIEW.md` -> `docs/IMPLEMENTATION_MAP.md` -> relevant `docs/` domain file -> `openspec/specs/` for stable contracts -> `progress/` for history/evidence.
- Routing: use `docs/AGENT_INDEX.md` and `docs/catalog.yaml` before broad source searches.

## Hard Rules
- Config-first: prefer YAML/schema changes over new stable CLI flags.
- Default training surface: offline-prepared single-dataset JSONL; runtime fusion configs are legacy/experimental.
- Treat Stage-1 baseline SFT and Stage-2 rollout-aware training as active first-class surfaces.
- Preserve image/geometry alignment end to end; never drop or reorder coordinates.
- Route bbox math through `src/datasets/geometry.py` unless editing detection serialization code.
- Training uses `do_resize=false`; do not introduce silent resizing.
- Do not edit upstream HF model files, including `modeling_qwen3_vl.py`.
- Do not add hidden agent memory stores or self-modifying agent persistence.

## Governance
- Update docs when changing stable defaults, entrypoints, config schemas, artifact names, metric semantics, or recommended workflows.
- Use OpenSpec only for stable compatibility-sensitive contracts: training/eval behavior, config schemas, loss semantics, artifact names, or normative metrics.
- Do not use OpenSpec for ordinary experiment planning or implementation checklists.
- For nontrivial branch work, use a repo-local super-power plan/spec when available; keep detailed checklists out of durable docs.

## Navigation
- Start with the docs route, then use `rg` or `rtk` to narrow files, config keys, symbols, tests, or artifacts.
- When `.codegraph/` exists, use CodeGraph as a local navigation accelerator for indexed symbol search, callers/callees, file maps, and impact radius; keep docs/OpenSpec/config/artifacts authoritative, and use Serena for Python symbol-level edits after narrowing.
- Inspect the smallest code/config/artifact surface that can answer the task.
- Python: use Serena MCP for symbol-level exploration and edits when available, after narrowing with `rg`/`rtk`.
- Non-Python: use Serena for large symbolic changes; use raw shell plus the agent's patch/edit tool for exact Markdown/YAML/JSON edits.
- If available, use `coordexp-router-context` for entrypoints, current-vs-historical context, zoomed-out maps, and RTK/Serena navigation choices.
- Treat `superpowers` as plugin-managed: source is `superpowers@openai-curated`, not a vendored repo-local copy. Verify provenance via `.codex/config.toml` and `.codex/plugins/cache/openai-curated/superpowers/*/.codex-plugin/plugin.json` when changing it.

## Commands
- Codex shells initialize the `ms` conda environment by default; use plain `python`, `pytest`, and repo entrypoints without `conda run -n ms`.
- Targeted tests: `python -m pytest <target>`.
- Noisy summaries: `rtk git status --short --branch`, `rtk git diff --stat`, `rtk grep "<pattern>" <path>`.
- Wrapped tests under RTK: `rtk pytest <target>` or `rtk python -m pytest <target>` when compact output is useful.
- Exact reads or machine-readable stdout: use raw `sed`, `nl`, structured parsers, or the underlying command.
- One-off debug artifacts: write under `temp/`; remove them when no longer needed.

## Verification
- For each change, run the narrowest relevant check: unit test, smoke run, config parse, artifact/manifest check, metric check, or replay.
- Before production-scale training, verify geometry/image alignment, prompt/template compatibility, config resolution, cache/packing behavior, loss semantics, metric scope, artifact completeness, and eval validity.
- Report commands run and any skipped verification with the reason.

## Experiment Records
- After experiments, record scope (`tiny`, `val200`, `limit=200`, full-val, proxy, raw-text, coord-token), config, checkpoint, artifact root, parse/drop counters, and metric files before interpretation.
- Do not present `tiny`, `val200`, proxy, or partial-run evidence as full validation.
- Promotion order: artifact/manifest -> `progress/` note when useful -> checked-in doc/final memo -> `docs/` only for stable current behavior.

## Git Safety
- Dirty worktrees are expected; isolate edits and never revert unrelated changes.
- Use small logical commits when requested or during large refactors/incident response.
- Ask before high-cost runs, publication/external release, or irreversible research/compatibility decisions.
