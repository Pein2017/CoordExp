# `research-base-v2` retroactive governance receipt

| Check | Resolved value |
| --- | --- |
| `git rev-parse research-base-v2` (tag object) | `960c627c67a4f8f9ca596c8330a4c976aae2217a` |
| `git rev-parse research-base-v2^{}` (peeled commit) | `8dac2d041b67c6a9a13f3fc52b5dd6046a543d7b` |
| Tag message (`git tag -l -n9`) | `Research probe baseline v2: canonical target-binding gate passed (44 CPU-only tests)` |
| Tagger date | 2026-08-24 14:17:25 +0000 |

Cut 2026-08-24 without the D1-predecessor pre-tag gate (candidate review,
target-binding replay) or a post-tag lifecycle receipt that `research-base-v1`
received. This retroactive receipt is written under D1: tags are milestones
recorded after the fact, not gated ceremony.

**Carries relative to `research-base-v1` (`258868571`):**
`git log --oneline research-base-v1..research-base-v2` returns `fffc3632c`
(record research-base-v1 tag), `74609d2b1` (`fix-integration-receipt-root-binding`
— adds the execution-root-keyed compatibility receipt selector to
`scripts/research/research_probe_admission_consumers.py`), and `8dac2d041` (record
the canonical receipt-binding gate). `74609d2b1` is the load-bearing change.

**Replay anchor:** `git ls-tree -r --name-only research-base-v2 --
scripts/research | wc -l` → **315**, matching the current `HEAD` count.
`research-base-v2` is the last tag containing every `scripts/research/` file
before this change's deletions; replay a deleted producer with
`git worktree add <tmp-path> research-base-v2`.
