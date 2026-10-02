# Engineering design principles

Prefer a real owner with a small honest interface over pass-through facades or a
framework invented for hypothetical reuse. An interface includes units, ordering,
failure behavior and scientific meaning, not only function signatures. Share a
mechanism only when actual consumers agree on its contract; similar experiment
loops may have different populations, denominators or conditioning.

Use explicit dependencies and typed records where they clarify boundaries.
Keep low-level imports light and optional heavyweight backends behind their
runtime seam. Validate authored configuration before effects; report unknown keys,
invalid combinations and path origins rather than silently guessing a fallback.

Preserve logical versus packed positions, token/image alignment, geometry units,
object order, original versus edited labels and raw versus scored artifacts.
Do not hide a research policy inside a generic loader or alter a loss denominator
to make two implementations look uniform. Inference payload loading and exact
training-state continuation are distinct contracts.

Prefer focused, deterministic tests of real caller interfaces and failure modes.
A synthetic contract fixture is not measured model evidence. CPU tests, numerical
parity, resource cleanup and scientific efficacy establish different claims.
Expensive model experiments require their own explicit scope and resource budget.

Keep compatibility-sensitive behavior in its existing spec and executable tests.
Use [the asset policy](../RETENTION.md) for documentation: preserve durable design
reasons, not duplicate implementation inventories. Source and CodeGraph provide
the detailed map. An internal refactor does not create a new compatibility
contract merely because it changes file paths.
