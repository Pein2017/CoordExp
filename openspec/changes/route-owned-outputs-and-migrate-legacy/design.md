## Context

See proposal.md. Canonical main is 73822b0030934d28784b72a8a5325387b2f5cd41; canonical Research Probes is 19af7b15db17a861ae73310b377b9002d889e0df. Main has five pre-existing dirty entries; Research Probes was clean. The main guides still prescribe root launches and source archives; Research Probes has retired those archive surfaces. Those histories are not interchangeable.

Observed sessions establish real producer ownership: 01a0e2e2-ef29-7291-9045-f631930359c1 (root main) created the COCO/LVIS aggregate script on 2026-09-27, line 854. 01a0eec2-93f9-7242-b1bd-8e2a92cf584b (Research Probes) created the HF compatibility script on 2026-09-29, line 220, and corrected it at 258. Worker session 01a0f0cc-423f-7773-a1ee-0a2d6c848424 sends root-output Markdown reports through worker_turn.py (lines 1317/1561). Session metadata is historical producer evidence, not proof of a clean source commit. No transcripts are modified.

## Goals / Non-Goals

Goals: split source, scientific interpretation, transport messages and runtime data by their existing owners; absorb the remaining human research notes; migrate or retire every legacy root run family; update current consumers and reject new root-run writes; route detailed evaluation operations to the existing skill.
Non-goals: change model/data semantics, resume a historical run, qualify a GPU execution, preserve old frozen scripts as supported operators, or claim deletion of assets whose active owner or consumer is unresolved.

## Decisions

1. Retain exactly root/main and `.worktrees/{coordexp-infras,research-probes,research-probes-web-codex}` under the latest user lifecycle ruling. Integrate useful temporary commits and records into these owners, qualify the minimum payloads, then retire temporary branches/worktrees. Infrastructure runs use the physical coordexp-infras checkout. Do not recreate main-runs or a universal legacy depot. Source integration and dataset/environment qualification remain separate; no launch is performed.
2. Keep policy at existing owners. Main OUTPUT_STORAGE_POLICY owns root-sharing and main route; Research Probes specializes its branch and links to that shared-root rule. Lead-worker separates message files from receipts and binds output ownership in assignments. Preserve the dirty shared AGENTS and same-host transport reference byte-for-byte.
3. Retain small, evidence-backed operators at their real source owners. Record original identities and remove obsolete one-off generators and frozen source snapshots; do not retain executable historical copies or runtime archive fallbacks.
4. Absorb each remaining human-authored research record into its current question/unit/catalog owner, keeping candidate/failure/null status, denominators, counterevidence, version boundaries and exact source path/hash. Do not create a duplicate report archive. After the owner verifies the synthesis, remove the redundant output Markdown.
5. For each legacy output family, first establish a concrete current use or named necessary reproduction. Integrate useful maintained code and scientific conclusions into existing Git owners, retire obsolete bytes, and relocate only the justified minimum to its actual owner. Moving unknown legacy families into any retained worktree does not satisfy acceptance. Update current dereferencing consumers and remove superseded paths. The latest user ruling defers annotation relocation: Gate A retains its live root; old Label Studio is retired with its two drafts exported. Frozen receipts retain provenance fields, not a right to keep obsolete paths. Do not use compatibility symlinks or historical binary/source backups.
6. Record frozen source/script identity (exact path, SHA-256, size and disposition) without retaining executable copies. Remove only the exact recorded files after their report/evidence owner is reconciled; do not execute or reuse them.
7. Use this finite inventory and migration receipt. The small inventory summary is maintained with this change; its exact full machine receipt is hash-bound under coordexp-infras outputs/maintenance/output-storage-closeout-20261002. Preserve producer identities, explicit unknowns and immutable receipt bytes. Record new physical locators separately; never relabel old evidence as produced by a storage checkout's HEAD.
8. Keep `docs/eval/` for stable contract, interpretation and entrypoint overview; put command-level inference/evaluation procedures in the existing `.codex/skills/coordexp-infer-eval-workflow/` skill and its focused references. Do not duplicate the same detailed command recipe in both owners.

## Risks / Trade-offs

- Active readers/writers or concurrent source changes -> rescan exact holders before mutation, close service owners normally, and verify labels/drafts/counters on restart. Open log descriptors may follow a same-filesystem inode move only when no path reopen is required; never stop an ambiguous research producer.
- Embedded old absolute paths -> preserve immutable historical provenance, replace current consuming locators, and verify new inputs. A legacy binding is a migration task, not a permanent layout exception.
- Generated adapter README files -> allow only exact native-manifest-bound package metadata, including nested payloads; loose prose and symlink escapes remain forbidden. Do not rewrite sealed payload identities.
- Uncommitted source extraction -> keep verified task-scoped originals and leave them available through acceptance; do not claim Git recovery or permit decision-bearing execution from dirty source.
- Legacy root tree is large -> use metadata and same-filesystem inode-preserving moves where possible; verify both byte hashes before deduplicating an existing independent copy. Do not traverse symlink targets. Unknown producers remain unknown while their physical storage gets an explicit owner.

## Migration Plan

Keep the accepted source/report extraction and eval routing. Apply the user's
2026-10-02 retirement ruling: reject new root-run destinations; retain the exact full
machine inventory under the current infrastructure maintenance owner; migrate retained root families into
their actual owners after usefulness qualification; preserve the explicitly
deferred annotation roots and live Gate A state; retire old Label Studio while
exporting its two drafts; update current consumers; remove originals and obsolete
historical executable copies. Finish with focused consumer/negative checks,
knowledge/OpenSpec validation, exact move/dedup receipts, live state, diff/status
and layout verification. Never overwrite a concurrently recreated path.

The finite inventory is evidence for this maintenance pass. Temporary blockers
require exact resolution and are not an accepted final storage layout.
