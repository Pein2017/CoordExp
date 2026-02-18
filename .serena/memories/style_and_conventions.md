# Style + Process Guardrails (Memory)

Role separation:
- Memory role: cross-cutting guardrails for safe, reproducible implementation.
- Canonical docs: `docs/standards/CODE_STYLE.md`, `docs/standards/REPO_HYGIENE.md`, OpenSpec docs.
- Update trigger: when repository process rules or style standards change.

Guardrails:
- Prefer config-first (YAML/schema) changes over adding new CLI hyperparameter flags.
- Preserve geometry semantics; use shared geometry helpers instead of ad-hoc transforms.
- Keep upstream HF/Qwen modeling files untouched; extend through wrappers/adapters.
- Assume offline-resized data flow (`do_resize=false` path during training).

Process:
- Treat docs as canonical contracts; memories are retrieval notes.
- Route architectural/contract changes through OpenSpec governance.
- Keep edits scoped; never revert unrelated dirty changes.
