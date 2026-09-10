---
doc_id: docs.standards.repo-hygiene
layer: docs
doc_type: standard
status: canonical
domain: repo
summary: Place source, research knowledge and runtime artifacts under their current owners.
updated: 2026-09-09
---

# Repository Layout And Retention

Use the selected checkout's [branch/worktree policy](../BRANCH_AND_WORKTREE_POLICY.md)
for source ownership and lifecycle. This page identifies storage roles; it does
not authorize deleting artifacts, changing scientific meaning or moving worktrees.

## Place content by its purpose

| Content | Maintained location |
| --- | --- |
| Reusable executable behavior | The existing `src/` concept owner; use the [implementation map](../IMPLEMENTATION_MAP.md) when it is unknown |
| Maintained research implementation | `probes/<direction>/` in the research base, with explicit profiles and dependencies; see [research mechanics](../RESEARCH_PROBE_INFRA_BASE.md) |
| Current production configuration | The selected `configs/coordexp_swift/` family and its schema; direction-owned profiles remain with their package |
| CLI entry or operational utility | An existing maintained script/package entry, without duplicating its underlying library behavior |
| Behavioral invariants | Tests at the relevant caller or contract owner |
| Current usage and explanation | `docs/`; stable compatibility requirements are in `openspec/specs/` |
| Research question, interpretation or negative result | `research/`, with source evidence and the owning scientific scope |
| Superseded documentation/provenance | `docs/history/`; old scientific records remain in their research reading path or archive |
| Checkpoints, rollouts, large tables, logs and galleries | The run's declared `outputs/` or external artifact root; docs keep handles rather than copied payloads |
| Disposable local scratch | A task-specific temporary location; confirm ownership and necessary evidence before removal |

`progress/`, `output/` and `output_remote/` are legacy locations, not defaults for
new records. Historical use of a directory does not prove its current contents
are disposable. Public-data layout and regeneration are owned by the
[output/data provenance standard](OUTPUT_SYNC_AND_DATA_PROVENANCE.md).

## Keep one maintained implementation

Place a new reusable operation at an existing compatible owner. Promote a
local utility when a retained consumer demonstrates the shared contract, not
after a fixed number of uses. Keep direction-specific scientific choices local.
Follow [CODE_STYLE.md](CODE_STYLE.md) for module and interface design.

Production runs use their declared config interface. Research packages may own
explicit local profiles or CLI options; do not invent global config fields or
move them into a production config family merely for uniformity. A citable run
needs enough effective code/config/input context to identify what executed;
its run name is not a substitute for that evidence.

When an entry or profile retires, check current imports, commands, tests,
document links and bound historical inputs. Preserve required source/evidence,
redirect current callers, then remove obsolete implementation. Keep a pointer
only when an actual reader needs the old location. A cleanup must not silently
change geometry, loss, parser, matching, denominators or artifact interpretation.

## Separate source from runtime state

Human-maintained guidance, skills or tooling source may be versioned under their
explicit owner. Credentials, sessions, memory stores, plugin caches, model
caches and generated runtime state are not source by proximity. Never infer
permission to stage or delete them from Git ignore or tracking status.

Use the current retention/backup contract before touching outputs or caches.
Preserve active work, external mounts and reproducibility dependencies; deleting
a documented directory is not safe merely because its name says scratch or cache.
This documentation workflow does not perform process cleanup or artifact deletion.
