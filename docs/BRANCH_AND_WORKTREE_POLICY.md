# Checkout and integration boundary

Canonical research: `research-probes` at `/data/CoordExp/.worktrees/research-probes`.
Named cleanup target: `research-probes-web-codex` at
`/data/CoordExp/.worktrees/research-probes-web-codex`.
Production and infrastructure worktrees are outside research cleanup ownership.
Resolve the registered Project and canonical path before every write phase.

Preserve all existing dirty work. Do not reset, clean, auto-stash or overwrite
unrelated changes. A local checkpoint commit preserves owned uncommitted work;
it is not a validation result. Do not push or rewrite history without explicit
separate authority. A process can still use a clean checkout; inspect holders.

For large integration, use the authorized registered temporary worktree based
on the exact current canonical HEAD. Resolve scientific document conflicts by
meaning and provenance, never by wholesale stale-index replacement. Re-run the
integrated surviving suite and integrity checks. Immediately before adoption,
canonical must still have that exact HEAD, clean tracked/untracked state and no
conflicting active consumer. Adopt the validated descendant with `git merge
--ff-only`; never force an unsafe base update.

Temporary integration evidence can remain ignored locally. It is not a new
permanent research lane. Existing external data/model/output directories and
remote refs are not cleanup targets.
