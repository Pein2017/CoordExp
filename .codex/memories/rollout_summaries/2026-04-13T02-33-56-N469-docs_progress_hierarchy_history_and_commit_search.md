thread_id: 019d84b0-3fe8-7653-bbd7-b5a8c19424b5
updated_at: 2026-04-13T02:42:37+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/13/rollout-2026-04-13T02-33-56-019d84b0-3fe8-7653-bbd7-b5a8c19424b5.jsonl
cwd: /data/CoordExp
git_branch: main

# Traced why `docs/` and `docs/progress` appeared to lose files, and pinned the actual hierarchy-move commit.

Rollout context: The user asked to inspect git history to explain why `docs/` seemed to lose many files, then narrowed to why `docs/progress` only had one doc, then asked for the commit message that moved the hierarchy and clarified it should be a more recent change than the January docs reorg.

## Task 1: Trace docs file loss and identify the hierarchy changes

Outcome: success

Preference signals:
- The user asked to “refer to git history and track why my `docs/` folder seems to lose so many document files” -> future similar investigations should start from git history and deletion/rename commits, not from surface directory counts.
- After the January reorg was found, the user said “No, any other history? It shouldn't be that long ago” -> the user wanted the more recent hierarchy change, so future agents should not stop at the earliest plausible reorg and should search forward for newer architectural moves.
- The user’s follow-up on `docs/progress` (“Why my `docs/progress` only have one doc now?”) and later commit-message query show they care about the exact location of progress/history docs, not just the broad docs tree.

Key steps:
- Checked `git log --diff-filter=D --summary -- docs` to enumerate deletions under `docs/` and found clustered cleanup commits rather than continuous accidental loss.
- Compared file counts across revisions with `git ls-tree -r --name-only <rev> docs | wc -l`; the docs tree dropped sharply around cleanup commits but later expanded again.
- Used `git show --stat/--name-status` on the key commits to distinguish pure deletions from renames/consolidations.
- Pulled `git log --since='2026-03-01' --oneline --decorate --name-status -- docs progress` to locate the newer hierarchy migration and the surrounding refinement commits.

Failures and how to do differently:
- The initial January 31 docs reorganization (`a14cef2`) was real but was not the most recent hierarchy move the user meant; future agents should keep searching when the user says “it shouldn't be that long ago.”
- A naive reading of `docs/progress` is misleading because the canonical progress corpus is top-level `progress/`, not `docs/progress/`; future agents should verify with `git ls-tree` and `git status` before assuming a directory is tracked.

Reusable knowledge:
- The biggest apparent `docs/` file loss came from deliberate cleanup of temporary/generated artifacts, especially `docs/temp_packed_dataset/*`, including a very large `sample_stats.jsonl` removal in commit `b0705c2`.
- January/February docs changes were mostly restructuring and consolidation: flat docs were moved into subfolders and several overlapping runbooks were merged into canonical guides.
- The actual “docs vs progress” hierarchy split is in commit `39b4fbc1119d4a23bf86f1ca012275b2b696521a` (`2026-03-09`), message `chore(docs): migrate to scalable docs/progress architecture`.
- `docs/progress` is not a tracked canonical path in this repo; the tracked progress/history corpus lives under top-level `progress/`, and `git status --short docs/progress` showed `?? docs/progress/` for the local untracked directory.
- Follow-up hierarchy maintenance commits exist: `f718155` (`docs(progress): sync runtime architecture routing and runbooks`) and `ff0bc78` (`docs: refresh routing map`).

References:
- [1] `git log --diff-filter=D --summary -- docs` surfaced the deletion commits:
  - `1394c25` deleted `docs/data/PREPROCESSING.md`, `docs/data/PUBLIC_DATA.md`, `docs/training/STAGE2_AB.md`, `docs/training/STAGE2_ROLLOUT.md`
  - `6760e74` deleted `docs/refactoring/progress.md`
  - `b0705c2` deleted the `docs/temp_packed_dataset/` artifact bundle
  - `ffdce89` deleted `docs/centralized_infer/codex.md`
  - `cd67f30` deleted scratch notes in `docs/temp_packed_dataset/`
- [2] `git show --stat b0705c2 -- docs` showed the largest purge: `sample_stats.jsonl` alone removed `99388` lines, and the commit removed 12 tracked files from `docs/temp_packed_dataset/`.
- [3] `git show --name-status a14cef271b5f423bd4610fd92988eb420fc34a82 -- docs` showed the January restructure was rename-heavy, e.g. `docs/DATA_JSONL_CONTRACT.md -> docs/data/JSONL_CONTRACT.md`, `docs/STAGE2_AB_TRAINING_RUNBOOK.md -> docs/training/STAGE2_AB.md`, and `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md -> docs/training/STAGE2_ROLLOUT.md`.
- [4] `git show --stat --summary 39b4fbc1119d4a23bf86f1ca012275b2b696521a -- progress` showed the March 9 hierarchy migration, including renames of many `progress/` files, creation of `progress/index.yaml`, and deletion of `progress/stage2_rollout_matching_runbook.md`.
- [5] `git status --short docs/progress` returned `?? docs/progress/`, and `find /data/CoordExp/docs/progress -maxdepth 2 -type f` showed the lone local file `docs/progress/duplication_collapse_final_analysis_2026-04-13.md`.
- [6] `git ls-tree -r --name-only HEAD progress | wc -l` returned `44`, confirming the tracked progress corpus is top-level `progress/`.
- [7] `docs/README.md` and `docs/catalog.yaml` point readers to `progress/README.md` and `progress/index.yaml`, reinforcing that `progress/` is the canonical historical notes location.

## Task 2: Identify the newer hierarchy-move commit message

Outcome: success

Preference signals:
- When the user said “What's the commit message to move the hierarchy?” and then rejected the first answer with “No, any other history? It shouldn't be that long ago,” they were explicitly asking for the newer hierarchy shift, not the older docs reorg.
- Future agents should therefore prioritize the most recent relevant hierarchy move when the user frames the question as “any other history” or says the expected commit is more recent.

Key steps:
- Searched recent history with `git log --since='2026-03-01' --oneline --decorate --name-status -- docs progress` and `git log --grep='progress architecture\|hierarchy\|migrate to scalable docs/progress architecture' --all ...`.
- Confirmed the relevant commit was `39b4fbc1119d4a23bf86f1ca012275b2b696521a` dated `2026-03-09` with message `chore(docs): migrate to scalable docs/progress architecture`.

Reusable knowledge:
- The March 9 commit is the correct answer when asking for the hierarchy move that separated canonical docs from progress/history notes.
- Nearby commits `f718155` and `ff0bc78` refined the routing map after that migration, so if a future user asks for “the one after the split” those are the next commits to inspect.

References:
- [1] `git log --since='2026-03-01' --pretty=format:'%h %ad %s' --date=iso -- docs progress` included:
  - `39b4fbc 2026-03-09 09:02:28 +0000 chore(docs): migrate to scalable docs/progress architecture`
  - `f718155 2026-03-22 10:41:22 +0000 docs(progress): sync runtime architecture routing and runbooks`
  - `ff0bc78 2026-04-03 03:56:43 +0000 docs: refresh routing map`
- [2] `git log --grep='progress architecture\|hierarchy\|migrate to scalable docs/progress architecture' --all ...` returned the same March 9 commit as the hierarchy-move target.
