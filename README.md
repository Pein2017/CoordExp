# CoordExp infrastructure

This checkout owns infrastructure implementation and its runs. Use the current
`src/`, `configs/` and `tests/` for supported behavior, the local CLI help for
commands, and exact-checkout CodeGraph for relationships. Do not treat another
branch's documentation or a historical acceptance record as this version's
runtime capability.

[Durable invariants](docs/contracts/INVARIANTS.md) explain state, identity and
qualification boundaries. [OpenSpec](openspec/README.md) owns stable
compatibility-sensitive contracts. [Documentation](docs/README.md) is a small
asset collection; [storage](docs/OUTPUT_STORAGE_POLICY.md) binds this worktree's
outputs. Historical manuals use [verified recovery](docs/RETENTION.md), not a
parallel runbook with stale parameters.
