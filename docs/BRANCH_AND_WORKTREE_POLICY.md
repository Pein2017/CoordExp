---
doc_id: docs.branch-and-worktree-policy
layer: docs
doc_type: workflow
status: canonical
domain: repo
summary: Current production and research branch, worktree, and Codex-session routing for CoordExp.
tags: [git, branches, worktrees, codex, coordexp-infras, research-probes]
updated: 2026-09-09
---

# Branch And Worktree Policy

CoordExp has two current, non-substitutable routes: coordexp-infras production
work on repository `main`, and probe research on the fixed `research-probes`
worktree. This policy routes work between them; it does not merge their
authority or make either route disposable.

## Production routing

- `main` is the stable coordexp-infras branch. `/data/CoordExp` is its
  operational checkout for official training and evaluation.
- `coordexp-infras` is the active development branch. It is checked out at
  `/data/CoordExp/.worktrees/coordexp-infras` for feature and experiment work.
- `ms-swift` is the preserved pre-promotion mainline, retained as a history
  archive and compatibility/reference branch. It has no active worktree and
  is not a target for new implementation or launches.
- `origin/coordexp-infras` tracks the active development branch; promote
  validated work from it into `main` through an explicit merge.

When a task asks for the current production repository, implementation, or
default production branch, resolve it against `main` at `/data/CoordExp`. For
production feature work, use the Swift development checkout. Use `ms-swift`
only for historical reconstruction, old-run reproduction, or explicit archive
maintenance.

The normal iteration is: develop and validate in `coordexp-infras`; merge the
accepted commits into `main`; launch official training/evaluation from the
root `main` checkout; then update the development branch from the promoted
`main` state before the next feature slice.

## Research-probe routing

`/data/CoordExp/.worktrees/research-probes` is the single permanent research
base and fork point. Its protected directory is never moved, recreated or
retired by probe lifecycle work. Resolve its current ref at use time.
Research-owned `src/` can evolve independently of production main/Swift.

Routine shared research changes happen in this base. Large or conflicting
changes can use a temporary development worktree, integrate accepted changes,
and retire it. The former `research-probe-infras` lane is no longer permanent;
retire it only after preservation and content integration. Do not unlock the
fixed research base or touch production worktrees, remote refs or shared
agent/runtime configuration as part of that retirement.

For a new direction:

1. Fork from research-probes HEAD into an isolated direction worktree when
   isolation is useful. A tag or clean-tree admission dossier is not a fork
   prerequisite. Record the actual code revision and dirty status; strict
   admission remains an explicit capability with its own unchanged contract.
2. Put maintained experimental code in `probes/<direction>/` with ordinary
   imports, local profiles/configs, documented module entries and explicit
   tests. A continuing direction can contain several units/runs. Shared code
   lives at its `src` owner and never imports direction packages; executable
   imports must not depend on another temporary worktree.
3. Return unique research knowledge by an explicit file list selected for
   purpose, not by checking out entire unit directories. Units can contain
   producer code. Preserve original results, then update routers and rewrite
   current synthesis; compass is not append-only history. Keep conflicting
   evidence scopes visible. Code promotion and knowledge intake are separate.
4. Share an operation when actual retained callers justify its behavior;
   preserve scientific choices and delete superseded implementations after
   consumer checks. Do not merge an entire experiment branch to acquire a
   helper. Research-base changes do not automatically promote production code.
5. Before retiring a direction, preserve relevant effective sources/configs,
   modified/untracked content and necessary ignored outputs; verify source
   recovery and evidence accessibility without its directory. Recheck clean
   status and relevant execution/write holders, preserve a recoverable ref,
   then remove the eligible worktree and branch. Never treat ignored outputs
   as disposable caches or a stopped job as scientific completion.

Historical producers use their specific saved versions. `research-base-v2`
remains an older replay anchor, not a universal source for subsequently removed
code. The current restructuring's per-worktree archive refs and output locators
are in [its preservation record](../openspec/changes/restructure-research-probe-development/preservation.md).
See [Research Probe Infrastructure Base](RESEARCH_PROBE_INFRA_BASE.md) for
lightweight direct execution and optional strict capabilities.

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
branch `coordexp-infras` and path `/data/CoordExp/.worktrees/coordexp-infras`.
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
[`coordexp_infras.md`](coordexp_infras.md) and use `src/train.py`, `src/infer.py`,
`src/inference/`, and `src/eval/detection_consumer.py`.
