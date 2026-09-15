# Reference Code

This directory contains historical material for the CoordExp-swift rebuild.

`legacy_src/` is the previous active implementation moved out of the import
root. It is reference-only code for source study, invariant recovery, and
debugging comparisons. New implementation code must live under the active
top-level `src/` package and must not import from `reference/legacy_src/` or
use path hacks to keep the legacy source executable.

`legacy_openspec_2026-06-29/` is the archived OpenSpec tree. It remains useful
for historical comparison, but the active contracts for this rebuild come from
the current OpenSpec change and later approved changes.
