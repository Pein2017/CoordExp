# Repo Router

Role: top-level retrieval and precedence only. Do not use this memory to restate workflows, contracts, or test inventories.

Canonical entrypoints:
- `docs/AGENT_INDEX.md`
- `docs/catalog.yaml`
- `docs/PROJECT_CONTEXT.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/IMPLEMENTATION_MAP.md`

Rules:
- Precedence is `openspec/specs/` > `docs/` > `openspec/changes/<active-change>/` > `progress/`.
- For agent-first routing, start with `docs/AGENT_INDEX.md`; for machine-readable inventory, use `docs/catalog.yaml`.
- Core read spine after entry is `PROJECT_CONTEXT -> SYSTEM_OVERVIEW -> IMPLEMENTATION_MAP -> domain router -> relevant openspec/specs -> progress` only for history or evidence.
- Use `progress/` only for design history, diagnostics, benchmarks, or empirical evidence when current docs/specs do not answer the question.
- Route by domain: `docs/data/` for data contracts and preprocessing, `docs/training/` for training, `docs/eval/` for inference and evaluation, `docs/standards/` for repo policy.
- Treat `docs/ARTIFACTS.md` as the cross-cutting router for runtime artifacts, provenance, and logging surfaces.
- Treat `docs/catalog.yaml` as the canonical inventory; do not recreate its document list in memory.
- Keep this memory as a thin index, not a second documentation layer.

Search seeds:
- `stage2_two_channel|stage2_ab|stage2_coordination|stage2_rollout_aligned|rollout_runtime`
- `pipeline_manifest|run_metadata|trainer_setup|artifacts`
- `contract|jsonl|geometry|packing`
- `infer|engine|backends|orchestration|confidence|metrics`