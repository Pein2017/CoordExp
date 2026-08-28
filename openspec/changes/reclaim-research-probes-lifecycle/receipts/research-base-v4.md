# research-base-v4 receipt (task 8.5)

- Tag object: `560088c5e549f05bd4287a21d8893429a6e2198c`
- Peeled commit: `23d2f03dfbc781e7b981c35379839f5082ad0750` (`docs(openspec): check off 8.1-8.3`)
- Created: 2026-08-28T05:41:19Z on `research-probes` at `/data/CoordExp/.worktrees/research-probes`

## Wave 8 (user approval 2026-08-28 "完全同意. test, scripts 都可以大改动")

- Lane merges: `lane/docs` → d23530f46, `lane/tests` → d058dcbed, `lane/scripts` → d26362525; lead follow-ups 473852962 (orphaned configs), 034a743ed (person25 cluster + artifact-dependent test skip), 4fdbf62ec (receipt-bound restore).
- Artifact reclaim (8.4): 39 roots, 3.2 GiB, manifest in `receipts/artifact-reclaim-manifest.md`; 3 skipped as still bound.
- Receipt-bound files are load-bearing bytes: the sealed CPU compatibility receipts bind twelve source files by SHA-256; one bound test module was split by 8.1 and admission failed closed (3/51); restored from v3 with its sealer dependency (see scripts-entropy-ledger addendum).

## Counts (v2 → v4)

- `git diff --shortstat research-base-v2..research-base-v4`: 673 files changed, 4596 insertions(+), 198800 deletions(-)
- Tracked files: 4608 → 4095
- `scripts/research/*.py`: 315 → 172; `scripts/analysis`: 64 → 13
- Test modules (all `tests/`): 413 → 257; `tests/research`: 230 → 146
- `configs/`: 564 → 470; `docs/`: 936 → 939; `progress/`: 231 → 157; `memories/notes`: 33 → 24

## Verification at tag

- `CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest tests/research tests/artifacts -q -p no:cacheprovider --no-header -rfE`: **82 failed, 4027 passed, 4 skipped in 584.78s**; failure-set diff vs `receipts/test-baseline.md` (88): **0 new**, 6 gone (deleted `test_build_sorted_image2299_native_ledger`, `test_run_image_12576_row_mediation_crossover` modules; `test_run_sorted_fn_successor_behavior::test_v2_attestor_analyzer_admission_builds_behavior_contract_cpu_only` fixed by the rescue-transition repair).
- Admission trio: 51 passed. `pytest tests --collect-only`: 5607 collected, 0 errors (54 errors before wave 8).
- `git worktree list`: 8 entries; local branches: 8; tags: 18 `probe-final/*`, 14 `archive/*`, research-base-v1..v4.
