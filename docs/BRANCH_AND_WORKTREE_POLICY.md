---
doc_id: docs.branch-and-worktree-policy
layer: docs
doc_type: workflow
status: canonical
domain: repo
summary: Current production and research branch, worktree, and Codex-session routing for CoordExp.
tags: [git, branches, worktrees, codex, coordexp-swift, research-probes]
updated: 2026-08-28
---

# Branch And Worktree Policy

CoordExp has two current, non-substitutable routes: CoordExp-Swift production
work on repository `main`, and probe research on the fixed `research-probes`
worktree. This policy routes work between them; it does not merge their
authority or make either route disposable.

## Production routing

- `main` is the stable CoordExp-Swift branch. `/data/CoordExp` is its
  operational checkout for official training and evaluation.
- `coordexp-swift` is the active development branch. It is checked out at
  `/data/CoordExp/.worktrees/CoordExp-swift` for feature and experiment work.
- `ms-swift` is the preserved pre-promotion mainline, retained as a history
  archive and compatibility/reference branch. It has no active worktree and
  is not a target for new implementation or launches.
- `origin/coordexp-swift` tracks the active development branch; promote
  validated work from it into `main` through an explicit merge.

When a task asks for the current production repository, implementation, or
default production branch, resolve it against `main` at `/data/CoordExp`. For
production feature work, use the Swift development checkout. Use `ms-swift`
only for historical reconstruction, old-run reproduction, or explicit archive
maintenance.

The normal iteration is: develop and validate in `coordexp-swift`; merge the
accepted commits into `main`; launch official training/evaluation from the
root `main` checkout; then update the development branch from the promoted
`main` state before the next feature slice.

## Research-probe routing

- The current research authority is the fixed worktree
  `/data/CoordExp/.worktrees/research-probes`, currently on the
  `research-probes` branch. Resolve its live ref and commit at use time; branch
  names may change, but this fixed directory is never moved, recreated, or
  retired by probe lifecycle work.
- `/data/CoordExp/.worktrees/research-probe-infras` on branch
  `research-probe-infras` is the permanent integration lane for reusable probe
  mechanics. It merges accepted reusable mechanics into `research-probes` and
  is fast-forwarded from `research-probes` after each accepted lifecycle
  change; it is never a fork point for a new direction and never a retirement
  target.
- Both fixed research worktrees are protected by native Git worktree locks.
  Their current local protection is not an off-host backup, a remote branch
  promise, or approval to unlock, remove, prune, or rename either path.

For a new research direction:

1. Fork `probe/<direction>` from `research-probes` HEAD:
   `git worktree add /data/CoordExp/.worktrees/<direction> -b probe/<direction>
   research-probes`. The admission owner binds the exact commit and clean
   status at use time; no tag is required to fork. A direction worktree has no
   obligation to sync from `research-probes` during its life; run
   `git merge research-probes` inside it on demand, when the direction needs a
   new reusable mechanic.
2. Return records-only, not code. On `research-probes`:
   `git checkout probe/<direction> -- research/<unit-dirs>` for the unit
   directories, then hand-merge (append, do not overwrite) the shared routers a
   direction worktree diverges on — the investigation's `experiments/index.md`,
   `compass.md`, `research/index.md`, and any `research/decisions/` entry it
   touches. Code stays on the direction branch and is not returned by default
   (see promotion below). Follow the research-flow closeout order — result,
   then experiment router, then decision/compass, then `memories/current.md`
   — so a result never lives only in `memories/`.
3. Promote code only when a real second consumer exists; the default is
   experiment-local. When promotion is warranted, either
   `probe/<direction> → research-probes` directly or
   `probe/<direction> → research-probe-infras → research-probes` is
   acceptable; choose per case.
4. Cut a `research-base-vN` tag after a reusable-mechanics merge into
   `research-probes`, not after a records-only return, and record it in a
   one-paragraph receipt. `research-base-v2` (`8dac2d041`) is the replay
   anchor for every `scripts/research/` producer this repository has since
   deleted: replay by `git worktree add <tmp-path> research-base-v2` and read
   the producer from there.
5. Retire a direction lane once its worktree is clean
   (`git -C <wt> status --short` empty): tag first, then remove.
   `git tag -a probe-final/<direction> <tip> -m "..."` when the lifecycle
   completed and its records were returned; `git tag -a archive/<name> <tip>
   -m "..."` when the lane is untriaged and its content is preserved but not
   returned. Then `git worktree remove <wt>` and `git branch -D <branch>`.
   Recovery is `git branch <branch> <tag>^{}` followed by `git worktree add`.
   Remote-tracking refs are never touched. `image2299-mechanism-microscope` is
   the current live direction worktree and the reference specimen for this
   model.

## Codex sessions and task worktrees

Codex session transcripts, archived sessions, attachments, and app-managed
session indexes are historical or runtime-owned state. They may contain old
task-branch or worktree descriptions from when Swift was a candidate worktree.
Do not rewrite those transcripts to make past conversations appear current.

For a new or continued task, verify the live checkout with:

```bash
git branch --show-current
git worktree list --porcelain
git rev-parse --show-toplevel
```

For production work, the stable implementation checkout should report branch
`main` and path `/data/CoordExp`; the active development checkout should report
branch `coordexp-swift` and path `/data/CoordExp/.worktrees/CoordExp-swift`.
For research work, verify the fixed `research-probes` path and currently
resolved ref; use `git worktree list --porcelain` to confirm that both fixed
research worktrees remain locked. A Codex task that still displays an older task
branch is stale app-owned metadata; starting from or sending a new message in
the intended live checkout should refresh that association. Historical session
content remains unchanged by design.

## Legacy and upstream wording

References to `ms-swift`, `/data/ms-swift`, `src.sft`, legacy config roots, and
old mainline launchers are valid only when explicitly labeled as upstream
dependency, compatibility, historical evidence, or archive material. They must
not be presented as the current CoordExp entrypoint.

The current Swift entrypoints are documented in
[`COORDEXP_SWIFT.md`](COORDEXP_SWIFT.md) and use `src/train.py`, `src/infer.py`,
`src/inference/`, and `src/eval/detection_consumer.py`.
