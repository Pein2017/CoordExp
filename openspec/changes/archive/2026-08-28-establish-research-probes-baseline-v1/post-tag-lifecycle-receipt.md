# `research-base-v1` post-tag lifecycle receipt

Recorded on 2026-08-24 after the user authorized only local creation of the
annotated baseline tag. This later receipt does not move the tag.

## Exact tag result

| Check | Resolved value |
| --- | --- |
| Tag ref | `refs/tags/research-base-v1` |
| `git cat-file -t research-base-v1` | `tag` |
| Tag object | `0185a90e30813af7f256bec8acaaa73deba1741d` |
| `git rev-parse research-base-v1^{commit}` | `258868571e0e1347b3a3662a352d46aa336327a6` |

The peeled commit is the exact clean candidate reviewed after the one bundled
correction round. It descends from the revalidated infra anchor
`f337de5d0bd016b79aa012acfc491544e6313333`; the tag does not point at that
anchor.

## Gate disposition and retained boundaries

- A `claude-opus-5` leaf with `write:false` reviewed the exact tagged candidate
  and returned `candidate` with no P0/P1 findings. The lead independently
  replayed strict validation, the 41 target-binding contract tests, clean-status,
  fixed-path/ref/lock checks, and tag absence before creation.
- The user-approved CPU-only evidence remains mechanics-only. The separate
  bounded-GPU smoke in `harden-research-probe-target-binding` remains unchecked
  and unexecuted.
- This action created no remote publication or off-host copy. It did not unlock,
  move, delete, or recreate either fixed worktree; it did not delete refs,
  retire a worktree, reclaim artifacts, or authorize any such later action.

Future lifecycle actions, including generic-ref movement/deletion, tag deletion,
Git garbage collection, worktree retirement, and raw-artifact reclamation,
remain separately user-gated.
