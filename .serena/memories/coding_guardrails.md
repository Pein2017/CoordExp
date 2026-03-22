# Coding Guardrails (Memory)

Role separation:
- Memory role: coding-time guardrails only; use `docs/` for project background, workflows, and contracts.
- Canonical docs: `docs/standards/CODE_STYLE.md`, `docs/standards/REPO_HYGIENE.md`, `docs/PROJECT_CONTEXT.md`.
- Update trigger: when repo policy, code style, or architecture guardrails change.

Guardrails:
- Stay config-first: prefer YAML/schema changes over adding new CLI flags.
- Preserve geometry semantics; use `src/datasets/geometry.py` and never drop or reorder coordinates.
- Keep the offline-resized training path (`do_resize=false`) unless docs/specs explicitly change.
- Preserve Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model files such as `modeling_qwen3_vl.py`; extend through wrappers or adapters.
- Prefer the current narrow runtime seams over re-growing monolithic entrypoints:
  `src/bootstrap/`, `src/infer/{artifacts,backends}.py`, `src/eval/{artifacts,orchestration}.py`,
  `src/trainers/stage2_two_channel/`, `src/trainers/stage2_rollout_aligned.py`,
  `src/trainers/{rollout_aligned_targets.py,rollout_aligned_evaluator.py}`, and
  `src/trainers/rollout_runtime/`.
- Prefer small composable modules, explicit typed outputs, and deterministic targeted tests.
- Treat `docs/` and `openspec/specs/` as canonical; memories are retrieval notes only.
- Keep edits scoped and do not revert unrelated dirty changes.
- Route contract or architectural changes through OpenSpec.
- If current entrypoints, artifacts, or routing changed, sync `docs/catalog.yaml` and the CoordExp retrieval skills/memories in the same pass.
