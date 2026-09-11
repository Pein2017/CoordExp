---
doc_id: docs.training.index
layer: docs
doc_type: historical-router
status: historical-router
domain: training
summary: Historical router for legacy MS-Swift/mainline training notes and old-run interpretation.
tags: [training, historical, stage1, stage2]
updated: 2026-07-11
---

# Training History And Legacy Runbooks

This folder is retained for historical MS-Swift/mainline training contracts,
old-run reproduction, and empirical context. It is not the current training
route on `main`.

For current implementation work, start with the named config or source owner.
Use [`../IMPLEMENTATION_MAP.md`](../IMPLEMENTATION_MAP.md) only to locate an
unknown owner, then the relevant `openspec/specs/coordexp-infras-*` contract.
For Qwen execution semantics, use the
[Qwen manual](../standards/upstream/QWEN_VL.md); current launch code is under
`configs/coordexp_infras/` and `src/train.py`. The research base also maintains
[direction packages](../RESEARCH_PROBE_INFRA_BASE.md#maintained-direction-entries)
with their own objectives. Do not read all architecture pages before a local edit.

## Historical surfaces

- [`STAGE1_OBJECTIVE.md`](STAGE1_OBJECTIVE.md): legacy Stage-1 objective and
  comparator vocabulary;
- [`STAGE2_RUNBOOK.md`](STAGE2_RUNBOOK.md): legacy rollout-correction runbook;
- [`METRICS.md`](METRICS.md): historical metric and loss interpretation;
- [`../data/PACKING.md`](../data/PACKING.md): current packing ownership and contract
  links (not a historical training surface);
- [`drafts/`](drafts/): explicitly non-canonical experiment drafts.

The old `configs/stage1/`, `configs/stage2/`, archived recursive-detection
configs, `src/sft.py`, `src/trainers/`, `src/detection/`, and `src/infer/`
references remain useful only when interpreting historical artifacts or
comparator results. They are not current Swift config or source ownership.

## Contract boundary

Do not copy a historical Stage-1/Stage-2 requirement into a current doc or
config without checking the live `src/config/` models, current source, tests,
and stable `coordexp-infras-*` specs. If a historical behavior must become a
supported contract, use a separate OpenSpec change and update current docs only
after the behavior is implemented and verified.

## Use this router for

- identifying the provenance of an old training run;
- comparing legacy Stage-1 or Stage-2 objective terminology;
- finding historical metric names and artifact layouts;
- locating superseded research notes before migrating a durable interpretation
  into `research/`.

Do not use this page to authorize a current launch, infer a current config
schema, or decide current source ownership.
