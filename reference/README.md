# Historical source comparison

This existing mixed tree retains earlier source and OpenSpec history for source
comparison. It is not a documentation intake or a current implementation owner.

`legacy_src/` is the previous active implementation moved out of the import
root. It is reference-only code for source study, invariant recovery, and
debugging comparisons. New implementation code must live under the active
top-level `src/` package and must not import from `reference/legacy_src/` or
use path hacks to keep the legacy source executable.

`legacy_openspec_2026-06-29/` preserves the original framework history, including
its archive. Current requirements belong to this checkout's `openspec/specs/`;
historical proposals and completed tasks are not current execution grants.

The legacy source and its source-adjacent explanations remain together for
comparison; code retirement is outside the documentation cleanup. Do not add
snapshots or reorganize archived OpenSpec artifacts for naming consistency.
Shared retention rules belong to [the documentation owner](../docs/documentation.md).
