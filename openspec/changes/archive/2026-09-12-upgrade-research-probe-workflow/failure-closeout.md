# Baseline failure closeout

Date: 2026-09-12. Base: `c38d488caad9902a991d8ccd387ae3c6535c3e24`.
Work was isolated in `/data/CoordExp/.worktrees/research-probes-test-closeout-20260912`, branch `codex/research-probes-test-closeout-20260912`.
Evidence root: `/data/CoordExp/outputs/research/upgrade-research-probe-workflow/20260912/failure-closeout`.

The user requested diagnosis of the 14 inherited failures, archive and canonical integration, then explicitly authorized legacy removal: “所有旧的/legacy的,都可以放心铲除”. This follow-up removes obsolete obligations and repairs current contract tests. It does not retroactively change the original upgrade acceptance or its independent design review.

## The original 14 failures

The lead reproduced the exact original 14 nodes on the isolated base: **14 failed in 17.10 seconds**, exit 1 (`fresh-14-red.log`, `fresh-14-red-exit.json`). These failures were already present before the shared-input upgrade.

| Original failures | Cause established by inspection and counterfactual | Final disposition |
| --- | --- | --- |
| 4 embedding source-gate/load cases | Tests searched ancestors for an untracked round-trip receipt before reaching the loader or payload assertions. | Keep the real loader and installation checks. Stage the existing checked-in source study and receipt under a temporary contract root. Exercise both explicit-root and default-cwd entry routes, direct and inference attachment, and fail before model mutation on missing/wrong evidence. |
| 1 DoRA source-gate case | The tracked default source study passes; the historical local round-trip output is missing from a clean worktree. | Keep the current loader/setup-plan contract. Use the real study plus the existing synthetic receipt builder, explicitly labelled as synthetic, and check rejection of missing evidence. This is no claim of a measured DoRA round trip. |
| 2 owner-commit/A3 cases | Both depend on the retired `/data/CoordExp/.worktrees/owner-commit-binding` checkout. | Delete the two experiment-specific tests. |
| 1 failure-matrix digest case | Naming commit `412877ab58e6441deaa3943540ad7c9283df2708` changed the receipt version but left the complete-receipt digest stale. Reverting only that string restores the old digest. | Delete the entire obsolete run-record test module (9 cases) and its JSON receipt. It records an old 99-test run, not a current execution obligation. Git `d883d3089` retains that revision; the older archived record is a different 94-test run. |
| 1 benchmark smoke-readiness case | Static config checks pass, but the test requires an untracked July step-917 embedding artifact to exist on every checkout. | Delete that one historical-checkpoint test. Retain smoke-config/fixture checks and the existing benchmark config/runtime checks. |
| 1 renderer snapshot case | The same naming commit changed the template fingerprint salt. Recursive comparison differs only at the two fingerprint fields; old/new salts reproduce the corresponding hashes. | Keep complete snapshot equality and update only those two fingerprint values. Rendered text, geometry, token strings, ordering and supervision spans stay equal. |
| 4 positive-progress D/C endpoint cases | Original baseline/integrated logs fail at rebuilding a frozen C endpoint packet: its producer identity uses the isolated checkout's `__file__` path. The fresh same-node reproduction fails earlier, after canonical integration, because C training admission checks live `runtime.py` and `selective_preservation_dense.py`. Historical staged sources still match their recorded hashes. | Delete the retired D=A17/C32 endpoint test module (7 cases) and its dedicated 1,345-line executable. It has no current Python callers; the historical renderer uses only its schema string. |

Thus **8 of the original failing cases are removed and 6 are repaired**. Removing entire obsolete modules also removes their previously passing cases. Current source gates, payload installation, complete renderer equality, scoring/owner burden and forced/free credit checks remain enforced.

The generic embedding fixture uses the existing checked-in `probes/logit_lens/configs/source-gate-receipt.json`. The default source study currently passes. The separately hash-frozen Logit real-model source study described in the original acceptance is a different contract; its old digest mismatch is not evidence of a defect in the default loader. No production gate, expected historical hash or scientific receipt was changed.

The deleted D/C endpoint is also preserved exactly in `retired-positive-progress-matched-endpoint.py`, SHA256 `55cf20d579aeb10dfb1dab030fae2278ca8a6c7bd87df71b783eb9cdc0b0a5a5`. Deleting its tests alone would leave its `cpu_preflight` pointing at a missing file; retiring that unreferenced executable completes the removal. Existing retained ledgers and the three hash-bound historical trainers were not changed.

## Subsequent canonical-source dependency

The first complete closeout suite produced **1 failed, 1,010 passed** (`final-suite.log`, `final-suite-exit.json`). Its sole failure was `test_package_import_verifies_real_smoke_but_endpoint_rejects_it` in `test_margin_preserved_endpoint.py`: the old C32 smoke receipt's verifier consults absolute live canonical sources, and canonical `runtime.py` has now advanced through the upgrade. The old upgrade comparison ran before that canonical merge, so this additional dependency was not in its 14-failure list. This failure is retained as evidence, not hidden as a green run.

The currently dirty `probes/parallel_owner_research/transfer.py` imports `margin_preserved_endpoint` for current transfer/scoring helpers. That executable and the other nine tests in its module remain supported and unchanged by the cleanup. Only the obsolete smoke-success test is removed: a historical execution is not required to remain admissible against today's live source files. In total, this closeout removes **20 legacy test cases**, adds **2 current root-route parameter cases**, and removes one retired executable.

The endpoint agent's earlier candidate receipt describes the fresh failure rather than the original failure stage, and predates the lead's subsequent executable retirement. The original full logs and this final addendum take precedence over those superseded candidate statements.

## Verification and archive

Lead-owned complete regression: **1010 passed, 8 warnings in 105.72s (0:01:45)**, exit 0 (`final-suite-02.log`, `final-suite-02-exit.json`, `final-suite.json`). The command is the same broad command used for the original before/after comparison:

```bash
python -m pytest -q tests/qwen tests/inference tests/templates tests/adapters probes/dora_owner_learning/tests probes/logit_lens/tests probes/human13/tests
```

The 1,028-case integrated collection becomes 1,010 cases after removing 20 obsolete cases and adding 2 current root-route cases. This is the named relevant suite, not a claim that every test in the repository was run. Existing dependency warnings remain; there are no failed or skipped cases in this suite. All staged changed-file hashes were checked after the run. The independent diagnostic workers' focused receipts and the lead's full run are retained together.

Normal validation of all main specs passed (26/26); the new input-preparation spec also passed its targeted strict validation. An additional all-spec strict check reports 8 passed and 18 failed because 18 unchanged older specs contain placeholder Purpose text. The old spec files are byte-identical to the closeout base; this existing documentation warning is not a Python test failure and is outside this cleanup. See `specs-validation.log`, `archive-specs-validation.log`, and `input-spec-strict-validation.json`. The delta/main requirement text was compared again in full before moving the completed change into `openspec/changes/archive/2026-09-12-upgrade-research-probe-workflow`. The live infra documentation link now targets that archive.

No additional GPU work is performed for this closeout. The original mechanical model acceptance remains bounded at 0.825686 cumulative GPU minutes and makes no model-quality or GPU-throughput claim.

The new formal `research-probe-input-preparation` spec copies all **5 added requirements and 10 scenarios** exactly from the accepted delta, with no requirement omitted or widened (`spec-sync.json`). All 17 original implementation tasks were already complete. The original proposal, design, review, acceptance and delta are retained together in the dated archive; this addendum records the later cleanup authority and evidence.
