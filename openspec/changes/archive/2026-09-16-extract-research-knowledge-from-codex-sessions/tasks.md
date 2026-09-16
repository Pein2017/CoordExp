## 1. Freeze corpus and contracts

- [x] 1.1 Freeze the capture timestamp, complete session source list, research-document source list, worktree identities, and package boundaries.
- [x] 1.2 Create the common manifest/synthesis contract and developer-session exclusion rules for all Luna packages.

## 2. Luna session packages

- [x] 2.1 Classify and extract research-relevant sessions from May–June 2026 into `evidence/session-early/manifest.tsv` and `synthesis.md`.
- [x] 2.2 Classify and extract research-relevant July 2026 sessions into `evidence/session-july/manifest.tsv` and `synthesis.md`.
- [x] 2.3 Classify and extract research-relevant August 1–25 2026 sessions into `evidence/session-august/manifest.tsv` and `synthesis.md`.
- [x] 2.4 Require each package lead to replay coverage counts, inspect its skip boundary, and return a compact acceptance packet.

## 3. Luna research-document package

- [x] 3.1 Inventory and deduplicate root, canonical research-probes, Image2299, Human13, OwnerBridge, coverage, geometry, and infrastructure research documents.
- [x] 3.2 Extract current findings, negative results, lifecycle/authorization changes, contradictions, and unowned knowledge into `evidence/research-docs/manifest.tsv` and `synthesis.md`.
- [x] 3.3 Verify exact paths, worktree identities, dirty/uncommitted boundaries, and current semantic owners.

## 4. Cross-source integration

- [x] 4.1 Merge package manifests into `coverage.tsv`, preserving every source disposition and duplicate group.
- [x] 4.2 Reconcile session findings against current research owners into `research-findings.md` with scientific/technical disposition and not-claimed boundaries.
- [x] 4.3 Sample high-value, skipped, duplicate, historical, and needs-adjudication rows to test classifier and synthesis sensitivity.
- [x] 4.4 Produce `notion-update-plan.md` containing only unique `needs-summary` or `needs-adjudication` changes; do not mutate Notion in this wave.

## 5. Verification

- [x] 5.1 Verify counts, required columns, source-path existence, session IDs, duplicate groups, lifecycle labels, and owner mappings.
- [x] 5.2 Confirm no raw transcript/document dump, development-only synthesis, experiment launch, or local write outside this change directory.
- [x] 5.3 Run strict OpenSpec validation and reconcile checkboxes with accepted receipts.
