---
doc_id: docs.branch-and-worktree-policy
layer: docs
doc_type: workflow
status: canonical
domain: repo
summary: Current production and research branch, worktree, and Codex-session routing for CoordExp.
tags: [git, branches, worktrees, codex, coordexp-swift, research-probes]
updated: 2026-08-24
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
- `/data/CoordExp/.worktrees/research-probe-infras` is the bounded integration
  lane for reusable probe mechanics. It is not a competing research authority
  or a source for a new research probe. Accepted generic mechanics may merge
  from this lane into `research-probes`; the two lines never imply a merge from
  `coordexp-swift` or production `main`.
- Both fixed research worktrees are protected by native Git worktree locks.
  Their current local protection is not an off-host backup, a remote branch
  promise, or approval to unlock, remove, prune, or rename either path.

For a new research probe:

1. Start an ephemeral `probe/<ticket>` worktree from the newest annotated
   `research-base-vN` tag. Before the first such tag exists, use only an
   explicitly recorded `research-probes` source commit; do not invent an
   implicit baseline from `main` or an arbitrary worktree.
2. Bind the source commit, worktree path, configuration, external-artifact
   locator, and replay entry to the probe's research record before interpreting
   a result.
3. Return the research unit, result/review, conclusion, and provenance manifest
   to `research-probes` whether the result is positive or negative. Promote
   experiment code only after a real second consumer has an owner-preserving
   shared need; otherwise leave it experiment-local.
4. Keep checkpoints, caches, raw model outputs, and other large artifacts
   outside Git with bound locators and checksums. Before retirement, reserve the
   `probe-final/<ticket>` annotated-tag namespace and record its final source,
   path, and replay entry in the merged provenance manifest.
5. Retire an ephemeral probe worktree only through a separately approved
   evidence-preserving lifecycle action. Baseline-tag approval, document merge,
   or a clean Git status does not authorize branch deletion, tag mutation,
   artifact reclamation, or worktree removal.

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
