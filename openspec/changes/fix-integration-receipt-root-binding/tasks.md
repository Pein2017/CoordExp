## 1. Freeze the execution-root contract

- [x] 1.1 Add a focused interface-level regression that specifies the two
  allowed fixed execution roots, their distinct sealed receipt pairs, and
  unknown-root rejection, including canonical selection when the integration
  root is unavailable; run it RED against the current missing selector.
- [x] 1.2 Preserve the observed cross-root failure as a characterization:
  a receipt from the non-selected fixed root must still reject on path identity
  even when the corresponding bytes match.

## 2. Implement the closed selector

- [x] 2.1 Add the smallest explicit two-root receipt selector in the consumer
  adapter, resolve candidates independently, and route both default CPU receipt
  constants through it.
- [x] 2.2 Remove the superseded unconditional merged-target receipt constants;
  retain no filename/content/discovery fallback or automatic regeneration path.

## 3. Verify exact mechanics

- [x] 3.1 Run the focused selector regression and the CPU-only target-binding
  contract suite `tests/artifacts/test_research_probe_admission.py` plus
  `tests/research/test_research_probe_admission_consumers.py` from the
  integration lane (44 tests), then replay those same paths from the canonical
  target (its existing 41 tests); verify the selector covers both receipt pairs
  and each root accepts only its matching pair.
- [x] 3.2 Run strict OpenSpec validation, diff/residue checks, and a targeted
  no-GPU/no-external-write check; confirm the external receipt bytes, fixed
  paths, locks, and baseline tag remain unchanged.
- [x] 3.3 Freeze the infra diff and obtain fresh independent `claude-opus-5`
  leaf, `write:false` contract review; after a third same-class test finding,
  fix the shared root-relative test invariant and independently replay decisive
  checks without starting a fourth review loop.

## 4. Integration boundary

- [x] 4.1 Submit the verified infra change for a separate user decision on
  merge into `research-probes`; the user approved a fast-forward from
  `fffc3632c` to `74609d2b1`. No tag, push, move/unlock, receipt regeneration,
  or GPU smoke was performed.
- [x] 4.2 After the separately approved merge, rerun
  `tests/artifacts/test_research_probe_admission.py` plus
  `tests/research/test_research_probe_admission_consumers.py` CPU-only from the
  actual canonical target against its canonical-target receipt pair; record the
  expected 44-test result before any new fork baseline tag is considered.
  Receipt: `CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest -q` over those paths
  passed 44 tests in 3.20s at `74609d2b1`; external receipt digest stayed
  `c2fe7ad1e0271c8b20c6cfe6a72c0436ab8f0550bf56a95d4c8d0fd2b26c2bf1`.
