# Validation and remaining boundaries

## Completed checks

| Check | Outcome | Local evidence |
|---|---|---|
| Original operator modules plus knowledge contracts | 41 passed | .local/repository-upgrade/research-layout-before.xml |
| Same modules after relocation plus the same knowledge contracts | 41 passed | .local/repository-upgrade/research-relocated-targeted.xml |
| Wider probes/knowledge/artifacts suite | Timed out at 240 seconds with failures/errors; no complete JUnit report | WebCodex job wc_job_8CBAxfsUAvrKC0d1 |
| Isolated completed-row evidence-bound test | 1 failed: referenced historical artifact missing | .local/repository-upgrade/research-existing-failure.xml |

The 41 case identities match before/after after normalizing only their module
location. The three renamed files are byte-identical to their original Git
content (14453, 1687 and 1362 bytes). No test assertion was deleted.
Runs used CUDA hidden, offline resource access, one-thread math libraries,
python -B and no pytest cache provider. No GPU/model experiment was launched.

The existing knowledge checker passed: 345 catalog entries, 324 distilled,
21 current, 169 claim references. Five external links were not revalidated;
Git-backed knowledge recovery is not execution qualification. Strict validation
of this OpenSpec change passed. Scoped git diff --check passed; the index is empty.

## Failed wider gate

The wider suite is NOT accepted as passing. Its timeout supplies no final
executed-test count. A post-timeout process census found no residual pytest
process in this checkout. One representative failure was reproduced alone:
tests/probes/test_completed_row_crossover.py uses a retained evidence contract
whose accepted_raw_paths refer to a missing B-1.json in the historical
greedy-prefix-native-01 temporary worktree. Neither that test nor its producer
was changed here. Other wider-suite failures remain unclassified.

The next test-architecture slice should separate portable synthetic contracts
from explicit source-bound evidence replay. Do not delete scientific assertions,
recreate old worktrees, rewrite receipts or mark missing evidence as passing.

## Concurrent work and exact scope

The visualization commit ca7ec3c285 was preserved. During validation another
actor committed the five protected readout-audit knowledge files as ca29b8d43;
all five then exactly matched the recorded baseline hashes. During closeout,
that parallel work added coordinate_readout source/tests and changed its own
state.json in commit a6f236f38. The changed state was detected and left intact,
not restored to our old baseline. The other four recorded file hashes still
match. No concurrent source, test or state file was edited by this slice.
Final inspected HEAD: a6f236f38e2e60abb5a18d17e9c49534d5ada347.
Knowledge integrity was checked again after that commit and still passed.
The final task scope is 15 physical Git-status paths: four modifications,
three old test-path deletions, and eight untracked files (three relocated
modules and five OpenSpec files). No other dirty path was present at this check.

Our only non-OpenSpec changes are README.md, probes/README.md, pytest.ini and
three test renames. OpenSpec context is corrected in openspec/config.yaml.
Executable source, scientific records, qualification gates, released artifacts
and other worktrees remain untouched by this slice. No staging or Git publication.
