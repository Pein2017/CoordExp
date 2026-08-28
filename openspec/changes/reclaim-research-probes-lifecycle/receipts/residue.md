# Task 4.3 residue receipt

Checked from cwd `/data/CoordExp/.worktrees/research-probes`. `ps -eo pid,cmd | grep -F -- "<dir>" | grep -v grep` and `lsof +D "<dir>"` were both empty (no hits, no rows) for all four directories.

| dir | git check-ignore | size (du -sh) | process/lsof check | deleted |
|---|---|---|---|---|
| `.pi-worker/` | **NOT ignored** — `.gitignore` lines 62-67 re-include `.pi-worker/home/pi-worker/**`, and `git ls-files .pi-worker` lists tracked files (e.g. `.pi-worker/home/pi-worker/pi_worker_foreman.py`) | 1.7G | no process cwd/open-file hit | **no** — fails the ignored precondition; contains tracked content, out of scope for deletion |
| `.serena/logs/` | ignored | 72K | no process cwd/open-file hit | yes |
| `$evidence_dir/` (literal directory named `$evidence_dir`) | ignored | 4.0K | no process cwd/open-file hit | yes |
| `temp/` | ignored | 16K | no process cwd/open-file hit | yes |

Freed: 72K + 4.0K + 16K ≈ 92K. `.pi-worker/` (1.7G) left in place; `.serena/cache/` left in place per task 4.3 (lead deletes at close, task 7.1).

## Lead addendum (4.3)

`.pi-worker/` holds 8 tracked files under `.pi-worker/home/pi-worker/**`; only its ignored content was removed with `git clean -fdX -- .pi-worker` (runs/ 1.2G, results/ 151M, sandbox-base-v2/ 6.1M, sessions/, threads/, specs/, agent/): 1.7G -> 100K. No process referenced the directory.
