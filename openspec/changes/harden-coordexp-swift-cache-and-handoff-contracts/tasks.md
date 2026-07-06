## 1. Spec Baseline

- [x] 1.1 Draft proposal, design, delta specs, and this task ledger.
- [x] 1.2 Validate the OpenSpec change strictly before source edits.
- [ ] 1.3 Commit the planning baseline separately from implementation changes.

## 2. Wave A: Pack Cache Semantic Identity

- [ ] 2.1 Use CodeGraph to inspect the current packing-cache fingerprint flow,
  source identity helpers, and impacted tests.
- [ ] 2.2 Add failing tests proving `src/qwen/positions.py`,
  `src/qwen/fa2.py`, and `src/qwen/forward.py` participate in the
  packing-cache fingerprint determinants.
- [ ] 2.3 Add or confirm tests proving worker count remains provenance only
  and does not change the semantic fingerprint.
- [ ] 2.4 Implement the minimal determinant-file update without changing cache
  payload shape or packing behavior.
- [ ] 2.5 Run targeted packing-cache tests and verify the new tests fail before
  implementation and pass after implementation.
- [ ] 2.6 Validate OpenSpec and run `git diff --check`.

## 3. Wave B: Checkpoint Handoff And Readiness

- [ ] 3.1 Inspect the current checkpoint writer, checkpoint metadata, inference
  loader, and existing handoff manifest behavior.
- [ ] 3.2 Add failing tests for canonical production handoff identity and
  research/dev manual composition provenance.
- [ ] 3.3 Add failing tests for a read-only readiness validator that returns
  pass/hold with concrete missing or mismatched handles.
- [ ] 3.4 Implement the minimal handoff/readiness contract without adding a
  broad reporting framework.
- [ ] 3.5 Run targeted checkpoint, inference handoff, and readiness-validator
  tests.

## 4. Final Verification And Cleanup

- [ ] 4.1 Run targeted test slices for packing cache, checkpoint handoff, and
  inference handoff behavior.
- [ ] 4.2 Run `openspec validate
  harden-coordexp-swift-cache-and-handoff-contracts --strict`.
- [ ] 4.3 Run `git diff --check`.
- [ ] 4.4 Update only directly affected docs or OpenSpec artifacts; defer broad
  docs sweep to the later authority-docs wave.
- [ ] 4.5 Commit implementation changes in logical groups.
