---
name: coordexp-public-data-provenance
description: Audit, update, or hand off CoordExp processed-data provenance manifests, JSONL-only checksums, active-root cleanup guardrails, or cross-node regeneration from raw datasets.
---

# CoordExp Public Data Provenance

The portable contract is: raw data is fetched locally, processed data is
regenerated from Git-tracked manifests, and routine cross-node validation uses
model-facing JSONL checksums rather than mirroring data trees.

## Workflow

1. **Bound live data.**
   - Identify requested roots, keep-set, delete-set, and active training roots.
   - Preserve every active root unless the user gives fresh explicit authority
     to rewrite, move, regenerate, or delete it.
   - Create manifests only for materialized data; record absent variants as
     absent.

2. **Create one manifest per processed owner.**
   - Classify it as a processed dataset, image store, or annotation view.
   - Record source identity, generation command and environment, logical output
     root, schema/version, counts, and sidecars needed for regeneration.
   - Record the executable dependency closure: tracked or retrievable command
     owner, required configs, transforms, parameters, upstream inputs, and
     expected outputs. A command string alone is not a regeneration contract.
   - Image stores normally record identity and counts without routine per-image
     checksums; tighten this only for an explicit image audit.

3. **Keep checksum scope narrow.**
   - Hash model-facing JSONL samples and aggregate sorted path, digest, byte
     size, record count, and line count.
   - Exclude raw images, resized images, caches, and whole processed trees from
     routine checksum scope.

4. **Validate the contract.**
   - Start from an actual manifest in scope. Read its schema/version and checksum
     scope, then search tracked files for those identifiers to resolve one
     current schema owner and one validator. Stop on zero or multiple plausible
     owners instead of choosing by filename familiarity.
   - Inspect the resolved schema and validator, then run that validator through
     the repository's configured test runner. When changing the contract, add
     or run a negative case that proves wrong JSONL content is rejected.
   - When materialized data is present, require every listed JSONL and sidecar
     to exist and match. When it is absent, distinguish a supported skip from a
     validation success.
   - Complete when the resolved owner is unambiguous, the manifest can detect
     wrong content, and it can guide regeneration without copying the processed
     tree.

## Cross-Machine Handoff

Give a pull-first route: update tracked manifests, run validation, obtain the
required raw dataset locally, execute the manifest's generation command for
missing roots, then validate again. In a fresh checkout, verify that every
declared executable dependency resolves and that the command reaches a dry-run,
help, or minimal generation boundary before calling the handoff complete.
Report tiny, proxy, selected subset, or full scope only after naming the
manifest identity, logical data root, counts, and checksum status.

## Cleanup Guardrails

- Distinguish raw inputs, processed owners, caches, and tracked manifests before
  deletion.
- Remove data only after the keep-set is explicit and active roots are excluded.
- Preserve tracked manifests as the portable contract.
- Keep one-off cleanup helpers disposable and outside durable repository
  surfaces.
- Keep shared run-artifact transport outside this skill. Resolve its current
  owner from `docs/standards/OUTPUT_SYNC_AND_DATA_PROVENANCE.md` and live
  operator tooling; do not revive a removed transfer skill or treat a public
  data manifest as transfer authorization.

Report changed manifests, source and materialized scope, validation evidence,
active-root exclusions, regeneration route, and unresolved provenance gaps.
