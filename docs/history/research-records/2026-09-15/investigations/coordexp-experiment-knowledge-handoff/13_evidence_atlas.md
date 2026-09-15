---
title: Experiment Knowledge Handoff Evidence Atlas
description: Provenance and coverage index for recovered CoordExp experiment Markdown, claims, content hashes, and source identities.
type: investigation
role: evidence-atlas
authority: non_normative_research
status: historical-synthesis
domain: historical-experiment-handoff
updated: 2026-07-18
---

# Experiment Knowledge Handoff: Evidence Atlas

This atlas is a provenance and coverage index for the recovered CoordExp
experiment record. It is deliberately not a scientific review: it does not
promote a source to a claim, adjudicate conflicting measurements, or declare a
mechanism successful.

## Coverage contract

`source_coverage.tsv` is the row-level source oracle. It contains exactly
4,058 unique content rows:

| source domain | unique contents | authority input |
| --- | ---: | --- |
| `progress` | 684 | `manifest.tsv` rows with `progress_content=yes` |
| `output` | 3,374 | `output-markdown-union/2026-07-17/contents.tsv` |

The SHA-256 in every row is the content identity. `source_path` is a
semicolon-separated union of aliases: `all_known_paths` from the recovery
manifest for progress, and `all_source_paths` from the output contents index
for output. This preserves aliases without creating duplicate evidence rows.
The row also retains source-kind, branch, worktree, commit, date, lifecycle,
availability, and review-state provenance where the inventories provide it.

The source branch machine checker at
`historical-markdown-recovery@93deed826:scripts/research/check_experiment_knowledge_handoff.py`
was executed before import. It verified the two denominators, exact Secure Hash
Algorithm 256-bit (`SHA-256`) universes, alias inclusion, non-unknown fields,
destination links, and bidirectional claim references against `claims.tsv`.
The raw source-union inputs used by that checker were deliberately not copied
into this worktree.

## Disposition and destination semantics

Disposition is a preservation decision, not a truth judgment:

| disposition | meaning |
| --- | --- |
| `retain_current` | The recovered progress content is identical at its same current path. |
| `archive_historical` | The progress content is a distinct historical version or path and remains recoverable through the history bundle. |
| `archive_output` | The content is an output markdown record retained in the output snapshot inventory. |

`destination_docs` points to the package document that provides the relevant
context, limitations, and reading route. The destination is intentionally
coarse; it must not be read as a claim classification. `claim_ids` carries an
explicit `claims.tsv#<claim_id>` handle only when the source is a useful,
non-misleading representative for that claim; otherwise it is `-`.
`claims.tsv` keeps a bounded representative source list for compact agent
consumption. This sparse claim layer favors recoverability and reading quality
over exhaustive evidence adjudication.
`artifact_handles` carries the recovery snapshot handle for progress records
and the output snapshot handle for output records.

## Source groups and authority boundaries

The recovery inputs are external, immutable evidence inventories retained in
the historical recovery branch:

- progress content: commit `12467ca8e`, path
  `docs/history/worktree-union/2026-07-17/manifest.tsv`;
- progress source associations: the sibling `sources.tsv` in that recovery
  bundle;
- output content: commit `816fbb0a1`, path
  `docs/history/output-markdown-union/2026-07-17/contents.tsv`;
- output path-level provenance: the sibling `manifest.tsv` in that output
  bundle.

The atlas does not copy those source bodies. It records their content hashes,
aliases, and recoverability so that later claim documents can cite the exact
record without silently merging branch versions. Current research documents
remain the semantic owners for active directions; historical progress and
output rows are evidence only.

## Claim-link layer

Claims are intentionally kept separate from this coverage table. Each claim row
cites one or more `source_coverage.tsv` `source_id` values and states its
scope, denominator, conditions, artifact handles, and limitations. The atlas
therefore supplies the complete source graph while avoiding scientific
verdicts. The validator checks both directions: every source-row claim handle
must exist, and every claim must be referenced by at least one source row.

## Reading routes

- [Stage-2 rollout failure registry](07_stage2_rollout_failure_registry.md)
  for failure, invalid, stopped, and superseded rollout source groups.
- [Coordinate objective and decode negatives](08_coordinate_objective_and_decode_negatives.md)
  for coordinate-surface and decode-limitation source groups.
- [Training and runtime lessons](09_training_and_runtime_lessons.md) for
  operational lessons with explicit evidence handles.
- [Inference, evaluation, and artifact lessons](10_inference_evaluation_and_artifact_lessons.md)
  for artifact and evaluator boundaries.
- [Historical result registry](11_historical_result_registry.md) for measured
  result rows and Gaussian coordinate soft-target plus Ranked Probability Score
  lineage pointers.
- [Legacy design lineage](12_legacy_design_lineage.md) for old names and
  design proposals without treating them as current contracts.

These routes are navigation aids only; the row-level manifest remains the
provenance authority.
