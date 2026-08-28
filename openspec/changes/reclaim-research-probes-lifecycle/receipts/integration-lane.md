# Integration lane receipt (task 2.4, 7.1)

## 2.4 — rename executed 2026-08-28 from `/data/CoordExp/.worktrees/research-probe-infras`

Before: branch `codex/research-probe-infra-foundation` @ `74609d2b1`; stale unmounted branch `research-probe-infras` @ `62274a97d`.

```
git tag -a archive/research-probe-infras-62274a97d 62274a97d -m "archive: stale unmounted research-probe-infras branch (2 AGENTS.md sync commits, 240 behind); superseded, name reused by the permanent integration-lane branch"
git branch -D research-probe-infras          # Deleted branch research-probe-infras (was 62274a97d).
git branch -m codex/research-probe-infra-foundation research-probe-infras
```

After (`git worktree list --porcelain`):

```
worktree /data/CoordExp/.worktrees/research-probe-infras
HEAD 74609d2b16908c3b7f36e99fd7dd558f610fd6e0
branch refs/heads/research-probe-infras
locked Bounded research infrastructure lane; unlock only by explicit lifecycle decision
```

`archive/research-probe-infras-62274a97d^{}` → `62274a97d`. This supersedes the "superseded HOLD" row for `62274a97d` in the archived `establish-research-probes-baseline-v1` design ledger; the content is preserved by the tag.

## 7.1 — fast-forward after close

(pending)
