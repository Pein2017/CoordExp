# Instruction ownership audit: implementation evidence

Date: 2026-10-05. Repository cleanup is materially implemented; full acceptance
remains HOLD for Runner integration and lead-worker consolidation. The independent
verification follow-up below closes the earlier OpenSpec CLI evidence gap. User
design/authority decisions are resolved. This is a change record, not a new
permanent policy owner.

## Scope and identity

| Project | Canonical path | Implementation baseline HEAD |
|---|---|---|
| coordexp | /data/CoordExp | 82c688499576a743e826f7b9fc9fbf3ffd7e1cef |
| coordexp-infras | /data/CoordExp/.worktrees/coordexp-infras | 4344a5e48d4f05c8d6453a1d26009a8f47c012cd |
| research-probes | /data/CoordExp/.worktrees/research-probes | 71d11b95733e6444da399cff7b4a9e1a3a9fc4ee |

Only this change's initial proposal/metadata were dirty at implementation start.
All checked changed paths are instruction, local guidance, owner documentation
or this change's files. No model/training source, data, research unit, external
receipt, service or other worktree was modified. At the implementation handoff,
no commit, push, integration, model experiment, runtime reload or restart had
occurred and changes remained unstaged. Local versioning was authorized separately
after the independent verification below; the branch histories own commit IDs.

## Implemented ownership map

| Rule/workflow | Action | Content owner after cleanup |
|---|---|---|
| User authority, research epistemology, high-cost validation and evidence integrity | KEEP / MERGE | .codex/AGENTS.md |
| Physical checkout, output/storage, research placement | MOVE / POINTER | Existing branch/storage policies and canonical research/CONVENTIONS.md |
| Local Python integration, instruction loading, shared skill source | MOVE | Existing main docs/AGENT_INDEX.md, read on demand |
| Tool invocation, edit/Git/Job mechanics | DELEGATE-TO-TOOL-CONTRACT | Current harness contract, not a copied tutorial |
| Ordinary review sequence | MERGE | audit-review; domain skill adds a lens, not another review |
| Worker brief, scoped repair/continuation and real handoff | MERGE | native-agent-team-guide; lead-worker still contains some duplication because its edit was blocked |
| Scientific execution and claim boundaries | KEEP / POINTER | Retained research-flow references and actual research/source/spec owners |
| Shared-memory capture and publication grant | RECONCILE | Constitution; external MCP skill owns caller identity and lifecycle |
| OpenSpec lifecycle | KEEP | All twelve external skills unchanged; local context no longer duplicates generic doctrine |

The three project AGENTS are thin routers with read-if-missing constitution
pointers. These pointers do not imply automatic Runner injection.

## Complete local skill disposition

Names below refer to the original 25 non-OpenSpec top-level entries. Resources
were inventoried in the audit; retained scripts are not executed by this cleanup.

| Skill | Actual disposition | Why retain or remove |
|---|---|---|
| ask-advisor | SHORTEN | Distinct read-only second opinion; native-team owns dispatch mechanics |
| ask-pro | KEEP | User-mediated prompt and identified Notion readings, without automatic submission |
| audit-review | SHORTEN / common owner | Artifact/claim decision audit and domain references |
| codebase-design | DELETE package | Generic vocabulary/design ceremony; no unique runtime resource |
| codex-session-cwd-migration | KEEP | Session/SQLite/rollout migration scripts and integrity contract |
| codex-usage-ledger | KEEP | Receipt deduplication, fork/cumulative usage and accounting scripts/references |
| coordexp-infer-eval-workflow | KEEP | Effective config, HF/vLLM qualification and scored artifact workflow |
| coordexp-public-data-provenance | KEEP | Real regeneration closure, manifest and active-root protection |
| coordexp-runtime-performance | KEEP | Rank/work units and observed wall time versus capacity estimates |
| detection-gt-vs-pred-visualization | KEEP | Coordinate/raw-row identity, actual image viewing and matcher distinctions |
| full-pipeline-smoke | SHORTEN | Actual caller, asymmetric distributed work, checkpoint reload and durable consumer |
| git-hygiene | SHORTEN | Discoverable scoped Git operations; delegate tool mechanics, retain retirement/provenance routing |
| grill-me | KEEP | Explicit user-selected interview interaction and recording reference |
| handoff | DELETE package | Transport boundary and necessary content absorbed by native-team |
| lead-worker | KEEP, shrink blocked | Persistent same-host transport scripts remain useful; original text retained |
| model-diagnosis | KEEP | Model symptom versus invalid evidence, data, objective and runtime distinctions |
| model-innovation-risk-audit | SHORTEN / lens | Unique scientific-fidelity references; common procedure belongs to audit-review |
| native-agent-team-guidance | CONSOLIDATE | Briefs, package continuation, handoff and existing routing-evidence scripts |
| qwen3-vl-execution | KEEP | Precision, causal mapping, packing, autograd and real encoded-input references |
| reclaim-code-entropy | DELETE package | Generic cleanup checklist duplicating existing owners |
| research-flow | SHORTEN | Routes to retained detailed scientific contract instead of repeating it |
| safe-runtime-maintenance | KEEP | PID/writer/lock/SQLite and exact cleanup boundaries prevent costly loss |
| scope-economy-review | DELETE package | Generic scope judgment does not need a separate review layer |
| shared-memory | KEEP external owner | Experimental cross-harness MCP; symlink discovery rejection is not bad YAML |
| workflow-self-distillation | SHORTEN | Explicit promotion from repeated work to one shared owner |

The twelve OpenSpec entries were each reviewed and retained byte-identically:
new-change, propose, ff-change, continue-change, update-change, explore,
apply-change, verify-change, sync-specs, archive-change, bulk-archive-change and
onboard (all prefixed openspec-). Native .system and synced collection ownership
remains unchanged. No archive or tombstone was created for deleted local skills.

## High-value safeguards preserved

The retained scientific reference still distinguishes native behavior from
forced/teacher-forced observations, nested populations and denominators,
technical-invalid/unexecuted evidence from a scientific null, and unmatched
neutrality from changed denominators or zero gradients. Reusing raw execution
after evaluator repair still depends on whether the evaluator affected sampling,
updates, selection or stopping. Clean current-source qualification and sealed
historical receipt integrity were not relaxed. Existing mechanism-promotion
thresholds were not redesigned during instruction cleanup.

The smoke procedure retains real entry-to-terminal-consumer evidence, fresh
checkpoint reload when implicated, nonempty publication/readback, intentionally
unequal multi-process work and measured scaling bounds. CPU substitutes are not
claimed as native model/distributed evidence. Worker interruption is not proof
that its external job stopped. Unknown memory/caller identities stay fail-closed.

## Measured size, not a token benchmark

| Surface | Before bytes | After bytes |
|---|---:|---:|
| Constitution | 14653 | 8766 |
| Main root AGENTS | 4257 | 1014 |
| Infrastructure root AGENTS | 0 | 924 |
| Research root AGENTS | 1154 | 1013 |
| All four resulting AGENTS files | 20064 | 11717 |
| Non-OpenSpec top-level SKILL.md bodies, including external shared-memory | 128132 | 76031 |
| External OpenSpec SKILL.md bodies | 134752 | 134752 |

Relevant AGENTS count is 3 to 4 because infrastructure lacked its entry.
Main's actual source payload is 18910 to 9780 bytes, about 48.3% smaller;
its latest bootstrap returned both bodies completely, without truncation.
Local skill entrypoint bodies are about 40.7% smaller. These counts exclude
references, scripts, tool schemas and Host prompts. No claim is made that all
skill bodies load every round or that byte reduction equals measured token cost.

## Actual loading and discovery

Fresh work_on_project responses after edits:

| Project | Automatically loaded source files | Returned text characters |
|---|---|---:|
| coordexp | AGENTS.md and .codex/AGENTS.md, each once | 9778 |
| coordexp-infras | AGENTS.md only | 923 |
| research-probes | AGENTS.md only | 1012 |

The other two Projects still need to follow the constitution pointer. The
approved automatic Runner-global injection is NOT implemented or verified.

Fresh skill_list reports 32 effective entries in each Project (formerly 36).
There are 33 structurally valid top-level definitions; shared-memory is not
admitted by the Runner because the package directory is a symlink. Existing
shared diagnostics remain three: .system and synced are collection directories
without a top-level definition, and shared-memory is a rejected linked package.
Main additionally rejects its linked .agents/skills project source. No fake
collection definition, copied package or weaker symlink rule was introduced.

## Verification evidence

- Native skill-creator quick_validate: all 21 retained non-OpenSpec entries pass.
- All 33 retained top-level frontmatters parse; all twelve OpenSpec definitions
  match HEAD; sampled baseline native/system/synced definitions are unchanged.
- Local Markdown file paths and heading anchors: no broken references found.
- Four removed packages had no external references in the three current tracked
  non-archive surfaces before deletion. Exact file bytes matched HEAD; empty
  package directories were subsequently removed with exact rmdir arguments.
- All three git diff --check calls passed; changed-path allowlist and unstaged
  state checks passed. Hygiene reports only the expected dirty-worktree warning.
- OpenSpec instructions apply returned ready with proposal/design/tasks present.
  This is planning readiness, NOT OpenSpec validation success.
- No independent fresh-agent behavioral benchmark or model run was performed.

Task-local evidence files under main's ignored scratch are baseline.json,
structure-validation.json and scope-validation.json in
.local/scratch/instruction-audit-20261005/. They support the recorded measurements;
the source changes and current loading must be rechecked after future edits.

## Implementation-phase tool blockers

1. Runner config read_files returned sensitive_path for its startup-bound
   configuration. Current runtime discovery exposes sanitized check/reload but
   no field-edit tool. runner_config_check returned valid=true, generation=1,
   restart_required=false. No alternate filesystem route or config reload was
   used. A sanctioned instruction-field edit capability is needed to implement
   the already-approved injection, then verify deduplication in all Projects.
2. The whole-file lead-worker text consolidation was blocked by the Host safety
   check before a result. A later comparison confirms the file remains exactly
   HEAD. Its existing functionality/resources are preserved; some duplication
   remains. No alternate mutation route was attempted.
3. The final OpenSpec validate request was also Host-blocked before a result.
   No validation exit status or successful CLI receipt exists for that call;
   do not relabel it as passed from YAML parsing or apply readiness.

No further user architecture decision is pending. Keep tasks open where the
literal acceptance criterion is unmet; keep this change unarchived and its Goal
incomplete. No automatic continuation or background work is promised.

## Independent verification and local versioning (2026-10-05)

Fresh registered Project paths and physical checkouts matched the three roots
above. Their HEADs still matched the implementation baseline before local
versioning, with 28, 2 and 3 dirty files respectively and empty indexes. This
follow-up used native Codex checks and fresh WebCodex observations; it did not
continue the blocked implementation or reload the Runner.

- All 33 retained frontmatters parse; installed skill-creator validation passes
  for all 21 non-OpenSpec entries. All 12 OpenSpec entrypoints remain byte-identical
  to HEAD. All 77 retained tracked package resources exist; 68 local Markdown
  links, including 19 heading anchors, resolve. Active tracked text in the three
  checkouts contains no inbound references to the four deleted packages.
- Fresh work_on_project responses return complete instruction bodies matching
  disk: main has AGENTS.md plus .codex/AGENTS.md (9778 characters), infrastructure
  has AGENTS.md only (923), and research has AGENTS.md only (1012). The global
  constitution injection criterion remains unmet.
- Guarded skill catalog pagination completes at 32 entries per Project, with
  stable catalog revisions; all 32 definition revisions match disk. A complete
  research-flow skill_load also matches the current definition. The external
  shared-memory source is structurally valid but remains excluded by WebCodex's
  linked-package admission rule. Existing collection/link diagnostics remain;
  no trust rule was weakened.
- runner_config_check reports valid=true, current_generation=1 and
  restart_required=false. lead-worker/SKILL.md remains unchanged from HEAD.
- OpenSpec 1.13.0 ran
  `openspec validate simplify-agent-instruction-ownership --strict --no-interactive`
  with actual exit 0 and `Change 'simplify-agent-instruction-ownership' is valid`.
  The CLI explicitly accepts zero spec deltas because skip_specs=true. This is
  fresh CLI evidence, separate from the implementation-phase blocked request.
- All three git diff --check calls pass. A bounded scan of added/new text found
  no credential patterns or secret-like candidate paths; this is a scoped check,
  not a claim that arbitrary repositories contain no secrets. Recomputed byte
  totals match the size table above. No behavioral-equivalence, model-quality,
  latency or actual token-cost benchmark was performed.

Primary observations and terminal command results are retained in Codex chat
01a10b0d-69cc-7b82-89f1-9194cfd7e392. Temporary checkers and the bounded receipt
under /tmp support this check; they are scratch rather than permanent owners.

The user selected separate local commits on main, coordexp-infras and
research-probes, including this verification follow-up. Stage only this change's
explicit paths. Keep tasks 2.2, 3.1 and 3.2 open and the change unarchived; local
versioning does not complete their acceptance criteria. Push, branch integration,
runtime configuration changes and service operations are outside this selection.
