# Standalone COCO Refinement Wave 1 Gate

## Verdict

**Passed at commit `c94f66da` on 2026-07-17.** OpenSpec task 1.9 has no
unresolved P0/P1/P2 findings. This receipt is the durable fixed-point evidence
for the Wave 1 engineering and intent/contract gates.

## Executed Gate

- Wave 1 targeted tests (`canonical`, `repository`, `bootstrap`, and
  `preflight`): 74 passed.
- `openspec validate coco-refinement --strict`: valid.
- Django, Label Studio model, and `AnnotationDraft` vendor-import residue in the
  standalone Wave 1 surface: none. Imports from
  `src.label_studio_coco_refinement` are the explicitly approved compatibility
  boundary for the existing COCO registry and working-store core, not vendor UI
  or Django state integration.
- The full Gate A `state.sqlite3` at
  `outputs/coco_refinement/gate-a-20260717/` opened successfully through the
  hardened repository at this fixed point.

## Engineering Audit

The initial fixed-point audit found one confirmed P1: schema-v1 validation
checked columns, foreign keys, and named indexes but did not attest required
`CHECK` and table-level `UNIQUE` constraints. An executed malformed-v1 probe
showed that a database missing those constraints was accepted and could contain
invalid splits, negative values, and duplicate identities.

Commit `c94f66da` closed the finding by comparing a quote-aware normalized
identity of every canonical `CREATE TABLE` definition. The regression test
reproduces the malformed schema and invalid rows, requires
`coco_refinement.schema_constraints`, and proves rejection does not self-heal
or rewrite schema/data. The closure engineering audit was **CLEAN**, with no new
P0/P1/P2.

The intentional residual behavior is fail-closed: a hand-authored schema with
semantically equivalent but non-canonical DDL is rejected. Runtime databases
are created by this repository, so this is the preferred safety boundary.

## Intent and Contract Audit

The independent intent/contract audit and its post-fix closure review were both
**CLEAN**, with no P0/P1/P2. The fixed point preserves the approved exact
max_len12000 train/val sources, norm1000 COCO-80 objects, compact complete task
indexes plus sparse SQLite Drafts, shared image links without copies, immutable
source files, standalone build-free service, legacy port-8080 isolation, and
the explicit prohibition on real ROI work before operator Gate A approval.

No corresponding known-finding ledger existed. The gate therefore closes with
no inherited unresolved finding.
