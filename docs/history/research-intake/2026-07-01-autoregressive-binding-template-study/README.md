---
doc_id: docs.history.research_intake.autoregressive_binding_template_study_2026_07_01
layer: docs
doc_type: history-intake
status: raw-intake
domain: research-history
summary: Raw manifested intake for Codex autoregressive binding and duplication experiment records before OKF research synthesis.
tags: [history, research-intake, autoregressive-binding, diagnostics, okf]
updated: 2026-07-01
---

# Autoregressive Binding Template Study Research Intake: 2026-07-01

This bundle preserves the raw Markdown evidence used to absorb Codex experiment records into an OKF-style research investigation. It is raw provenance, not current-behavior documentation.

## Scope

Two source waves are included:

- `june10_12_deleted`: 141 Markdown records deleted by commit `9d668607494f39ff59bc0756bdc81cc5a8b79969` (`Consolidate June mechanistic diagnostics`) and recovered from `progress/diagnostics/artifacts/june_10_12_consolidated_sources/source_notes_2026-06-10_12.tar.gz`. The parent commit before deletion was `df922431396db3da33925e2569a01449ae868e32`.
- `june20_27_branch`: 82 tracked Markdown records from `/data/CoordExp/.worktrees/autoregressive-binding-template-study` at commit `e026b290e15c09f91357329bac8dfd41fa74d7f9` on branch `codex/autoregressive-binding-template-study`.

## Manifest

Use [manifest.tsv](manifest.tsv) as the source of truth for this intake. Each row records source wave, source kind, branch, commit, original path, source date, SHA-256, extracted title, `doc_type`, `evidence_scope`, compact artifact handles, snapshot path, synthesized destination, and raw status.

## Snapshot Policy

Raw Markdown is copied under [snapshots/](snapshots/) in SHA-prefix directories:

```text
snapshots/<sha12>/<original/source/path>.md
```

The snapshot path is content-addressed enough to avoid collisions while preserving the original relative path for provenance.

## Counts

- Manifest rows: 223
- June 10-12 deleted-wave records: 141
- June 20-27 branch-wave records: 82
- Unique snapshot files: 223

Rows by source date:

- `2026-06-10`: 28
- `2026-06-11`: 78
- `2026-06-12`: 35
- `2026-06-20`: 15
- `2026-06-21`: 23
- `2026-06-22`: 24
- `2026-06-23`: 2
- `2026-06-25`: 1
- `2026-06-26`: 5
- `2026-06-27`: 12

## Governance

Do not cite this bundle as current behavior. Use it to reconstruct raw evidence, audit the OKF absorption, or recover exact source records. The synthesized reading path is `research/investigations/autoregressive-binding-template-study/`.
