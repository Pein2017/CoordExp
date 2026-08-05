## 1. Decision And Contract

- [x] 1.1 Measure the real production cache and reject pure lazy decoding after observing about 7.39 seconds per chunk and about 32 minutes of extrapolated aligned in-loop stalls.
- [x] 1.2 Select final end-to-end wall clock as the sole performance objective and choose eager-once over an unproven prefetch subsystem.
- [x] 1.3 Define explicit `manifest` and `payloads` verification levels with no default.

## 2. Implementation

- [x] 2.1 Keep preparation, publication, completeness, reuse, and eager eval on `payloads` verification.
- [x] 2.2 Use `manifest` verification for distributed train resolution and retain the existing eager rank loader as the one validated payload pass.
- [x] 2.3 Add focused tests for explicit lifecycle intent and corrupt-after-preparation failure before forward.
- [x] 2.4 Remove the superseded lazy-reader, prefetch, and synthetic stream-probe artifacts from this change.

## 3. Verification And Backport

- [x] 3.1 Run focused pack-cache, pipeline assembly/rebuild, trainer, runtime, and eval tests in the source worktree.
- [x] 3.2 Obtain a bounded independent Opus review, reject the P1 lazy wall-clock regression, and resolve the finding with eager-once.
- [x] 3.3 Semantic-backport only the coordinate-independent eager-once OpenSpec/code/tests to `/data/CoordExp/.worktrees/CoordExp-swift`.
- [x] 3.4 Independently run focused tests and strict OpenSpec validation in the clean target and verify no resume/8-coordinate/config/prefetch contamination.
