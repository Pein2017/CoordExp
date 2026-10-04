# Validation and limits

## Executed CPU checks

All pytest runs used CUDA_VISIBLE_DEVICES empty, offline Hugging Face flags, one-thread math libraries, python -B and no pytest cache provider. Frontend validation selected the already installed Node 22 runtime per command; no dependency installation or persistent environment change occurred. Local tokenizer/processor preflight is not a model-forward or GPU qualification.

| Check | Passed | Failed | Errors | Skipped | Report |
|---|---:|---:|---:|---:|---|
| Initial default suite (stopped after three failures) | 442 | 3 | 0 | 0 | `.local/repository-upgrade/main-baseline.xml` |
| Expanded suite before harness-test retirement | 1909 | 6 | 0 | 1 | `.local/repository-upgrade/main-reproducible.xml` |
| Final retained CPU suite; explicit not-live selection | 1907 | 3 | 0 | 1 | `.local/repository-upgrade/main-final.xml` |
| Isolated unchanged model/render canaries | 0 | 3 | 0 | 0 | `.local/repository-upgrade/main-existing-canaries.xml` |
| Final explicit live data/application gate | 0 | 3 | 0 | 0 | `.local/repository-upgrade/main-live-environment.xml` |
| Continuation full retained CPU recheck; explicit not-live selection | 1907 | 3 | 0 | 1 | `.local/repository-upgrade/main-resumed.xml` |

The final retained run separately deselected the three explicitly marked live_environment tests. Those same tests were then executed as a separate gate; they were not silently skipped, deleted, or rebaselined. The initial suite was not a complete pre-change pass: it stopped after three failures. The continuation repeated the retained selection and reproduced the same three failures in 105.50 seconds, with the three live_environment cases explicitly deselected. Intermediate collection/environment failures (duplicate basenames and the runner Node selection) were fixed and rerun.

## Remaining failed gates

- `tests/qwen/test_positions.py::test_real_smoke_packed_positions_match_upstream_per_segment_helper`: the installed upstream get_rope_index requires mm_token_type_ids, whereas this retained canary calls the older signature.
- `tests/qwen/test_token_identity.py::test_real_local_qwen_components_load_without_model_and_preflight_tokens`: the observed tokenizer class is Qwen2Tokenizer, not the frozen Qwen2TokenizerFast expectation.
- `tests/templates/test_renderer.py::test_expected_rendered_snapshot_matches_real_renderer`: the current renderer observation disagrees with its frozen JSON snapshot. No golden artifact was regenerated to suppress this failure.
- The selected train source bytes disagree with the Label Studio source receipt; the selected validation data includes a negative annotation ID rejected by the old positive-ID source contract; the installed Label Studio checkout differs from the pinned revision. These are mismatches against frozen qualification, not a claim that the current datasets are corrupt.

The three model/render failures reproduce together in an isolated run. Main source implementation, their test bodies and their fixture files are unchanged by this upgrade. No parameter, upstream dependency, source receipt or dataset was edited to make them pass. Main is not certified as a fully passing release.

## Structural and recovery validation

- The deletion inventory is 366 tracked files, with original SHA-256, byte count and reason. Final comparison verified every original hash against the initial unchanged HEAD and verified that all task-owned deletions match the manifest exactly. Two later unrelated .codex/project-memory deletions are excluded; global Git status is not falsely attributed to this change.
- Main src/ has no diff; all tracked source Python parses, and the static absolute src imports resolve to existing modules/packages.
- Actual inference-module resolution and its CLI help remain exercised. Both annotation systems are now discovered, including spawned-worker, data-commit, recovery and geometry tests.
- Documentation checker with --verify-recovery passes: nine live docs; 784 Git-backed and 12 extra sealed historical items, 796 total. This is documentation recovery, not a public-data regeneration claim. The closed seals were not changed.
- The selected non-deletion diff and all deletion identities were reviewed. Scoped git diff --check passes. Final Git/OpenSpec status is reported at closeout.

## Safety and unexecuted claims

No GPU or expensive training experiment, public-data regeneration, runtime receipt publication, cross-worktree source transfer or historical artifact rewrite was performed. Existing research workload processes were not stopped. Research was held read-only during the initial active workload. The continuation revalidated closure and applied its separate non-runtime ownership/test-layout slice; review.md records the changed HEAD and protected parallel work. Existing/concurrent .codex changes remain outside this change. No index staging, commit, push, merge, rebase, reset, clean or stash was performed.

## Continuation Git scope and sibling evidence

Main remains at 52d7dd204449a5d610715666359c91ce243b758e. The task owns 383
physical status paths: 366 deletions, nine modified files and eight untracked
files (seven bounded OpenSpec files plus tests/__init__.py). Seven unrelated
.codex status entries are excluded, including two deletions; their bodies were
not inspected for task authority. The index is empty, source/protected data
consumers have no diff, and task-scoped git diff --check passes.

Infrastructure's separate config refactor retains all 26 before/after observations
exactly. Its installed-API FA2 repair has 29 passing focused tests; its complete
CPU suite is 2079 passed, 34 failed and five skipped. The unresolved broader gate
is detailed in that checkout's two change validation files.

Research's separate ownership/test-layout change preserves the same 41 passing
cases and the original file bytes. The knowledge checker validates 345 entries;
five external links remain unverified. Its wider suite timed out at 240 seconds
with failures/errors and no complete JUnit report; one missing historical-artifact
failure was isolated. No broad passing claim follows from the targeted success.
Parallel research commits and new coordinate-readout files were preserved.
