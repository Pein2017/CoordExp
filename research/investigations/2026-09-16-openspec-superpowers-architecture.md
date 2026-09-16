---
type: investigation
date: 2026-09-16
status: closed
tags: [openspec, superpowers, architecture, tooling]
---

# OpenSpec and Superpowers architecture review

## Scope and decision

Reviewed the canonical project root and the three active CoordExp worktrees,
then compared the current upstream OpenSpec and Superpowers execution models.
The decision is to keep OpenSpec as the semantic and compatibility authority;
Superpowers remains an optional execution discipline linked to an active
OpenSpec change. They are complementary layers, not replacements.

## Verified local state

- Global `@fission-ai/openspec` is `1.13.0`; the canonical root's 12 generated
  OpenSpec skills were regenerated with `generatedBy: "1.13.0"`.
- Initial pass archived: five root changes (including one user-declared
  superseded change), one completed `coordexp-infras` change, and one completed
  `research-probes` change.
- Root active changes are `label-studio-coco-refinement` (28/37) and
  `coco-refinement` (36/50). The formerly complete
  `optimize-agent-instructions-serena-runtime` was user-declared superseded
  because `/data/deepseek-harness` is being removed; it was archived with a
  `SUPERSEDED.md` note and its delta specs were intentionally not synced.
- `coordexp-infras` retains `optimize-production-training-and-eight-gpu-smoke`
  (17/19). OpenSpec 1.13 correctly blocks it: its MODIFIED requirement omits
  four scenarios now present in the main `infra-base` spec. Repair the delta
  before archiving.
- `research-probes` retains three incomplete changes (24/29, 28/37, 13/14).
  `research-probes-web-codex` remains a stale duplicate and is not an
  authority; after explicit user authorization it was fast-forwarded to the
  same HEAD and its duplicate completed change was archived.
- Historical/back-up/snapshot `openspec/` directories and nested separate
  projects were inspected as provenance, not treated as current roots. Two
  already-archived historical changes still fail the archived validator because
  their old task lists are incomplete; they need a separate provenance-safe
  repair or retirement decision.

No commit or push was made. Archive moves therefore appear as old active
directories deleted plus dated archive directories untracked; no reset/clean
was performed and unrelated dirty paths were not targeted.

### Follow-up sweep

After the user authorized broader cleanup, `research-probes-web-codex` was
fast-forwarded from `6a48808e` to `df4890de` and its duplicate completed
research-probe change was archived. Additional complete changes archived were
three Web workflow changes, one wake-up plugin change, two codexhost changes,
and five HarnessDock changes in each of its `developer` and `main` worktrees.
The remaining active changes in those projects are incomplete or have stale
MODIFIED/ADDED deltas; OpenSpec validation is intentionally left as the repair
gate rather than bypassed.

### Research-lane closeout

The user authorized archiving the remaining research-probe tasks as no longer
important. The canonical `research-probes` tree archived all three remaining
changes with `--skip-specs`, committed only the OpenSpec paths as `ace16c8`, and
the web-codex tree was fast-forwarded to that commit. Both trees now have zero
active changes and identical HEADs. Their concurrent Source256 code,
experiments, and catalog edits remain uncommitted and untouched. Archived
validation reports 40 total / 35 passing in `research-probes` and 30 / 25 in
web-codex; the five failures in each are intentionally incomplete historical
archives, not active changes.

### Retirement decisions

The user subsequently retired `codex-host` and HarnessDock. Their remaining
active change folders were archived with `--skip-specs` so deprecated or stale
deltas would not overwrite stable contracts. `codex-host` now has no active
changes (53 archived; six older archives still fail the historical incomplete-
task check). Both HarnessDock worktrees now have no active changes (67
archived each; six newly retired incomplete-task archives are expected
validator findings). Project-level `RETIREMENT.md` files record that these are
provenance/superseded archives, not accepted feature claims.

## Layer comparison

| Concern | OpenSpec | Superpowers |
| --- | --- | --- |
| Authority | Durable proposal/spec/design/tasks, delta-to-main spec sync, validation, archive | Execution plan and sequencing derived from the approved scope |
| Execution | Assistant-driven, fluid artifact actions; CLI supplies deterministic scaffolding/status | Mandatory brainstorming/approval, worktrees, bite-sized plans, TDD, subagents, reviews, commits, branch finishing |
| Persistence | `openspec/changes/`, stable `openspec/specs/`, dated archives and cross-repo stores | `docs/superpowers/plans/` plus branch/worktree and review history; no equivalent normative spec registry |
| Best fit here | Research meaning, compatibility contracts, provenance and multi-worktree ownership | High-risk code implementation where test-first and review loops change acceptance quality |

Superpowers is stronger out of the box on implementation discipline, especially
mandatory RED/GREEN TDD and per-task/final review. It is not “better” overall:
it does not replace OpenSpec's semantic lifecycle or stable contract authority.

## Operating rule

Use one OpenSpec change per semantic surface. Invoke Superpowers only when the
implementation risk justifies its ceremony; its plans must point to the active
OpenSpec change and remain non-normative. Archive only after tasks, acceptance
evidence, and spec synchronization are closed. Do not mass-edit existing spec
warnings or merge incomplete changes without an owner decision.

## Sources

- [OpenSpec README](https://github.com/Fission-AI/OpenSpec)
- [OpenSpec v1.13.0 release](https://github.com/Fission-AI/OpenSpec/releases/tag/v1.13.0)
- [OpenSpec workflows](https://raw.githubusercontent.com/Fission-AI/OpenSpec/main/docs/workflows.md)
- [OpenSpec commands](https://raw.githubusercontent.com/Fission-AI/OpenSpec/main/docs/commands.md)
- [Superpowers README](https://github.com/obra/superpowers/blob/main/README.md)
- [Superpowers v6.3.0 release](https://github.com/obra/superpowers/releases/tag/v6.3.0)
- [Superpowers writing-plans skill](https://raw.githubusercontent.com/obra/superpowers/main/skills/writing-plans/SKILL.md)
- [Superpowers TDD skill](https://raw.githubusercontent.com/obra/superpowers/main/skills/test-driven-development/SKILL.md)
