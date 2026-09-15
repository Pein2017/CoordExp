---
title: Legacy Training and Inference Design Lineage
type: investigation
role: lineage
authority: non_normative_research
status: historical-synthesis
updated: 2026-07-17
---

# Legacy Training and Inference Design Lineage

This lineage explains how old names and design proposals relate to the current
coordexp-infras surfaces. It is a map for interpreting historical artifacts, not
permission to resurrect removed paths. Planning documents describe intent;
executed units and receipts are required before a claim becomes evidence.

## Lineage map

| ID | Legacy line | Planning vs executed evidence | What survives | What must not be inferred |
|---|---|---|---|---|
| LEGACY-001 | Historical Stage-1 JSON/coordinate-token SFT with softCE, W1, and gate. | Executed historical training and val200 notes; source `progress/pretrain/stage1_foundation.md` and Stage-1 benchmark notes. | Data/order/label-mask/loss vocabulary and the teacher-forcing vs free-rollout distinction. | Old config paths, model sizes, and scalar values are not current defaults. |
| LEGACY-002 | Old compact recursive-detection and prefix-roll-in families. | Both planning/contract docs and executed benchmark rows exist; current training router distinguishes production comparator, ablation, and legacy bridge. | Comparable-group discipline, EOS/continuation failure taxonomy, and explicit packing support status. | A proposal or ablation config is not production behavior; empirical EOS priors and removed routes must not be copied. |
| LEGACY-003 | Stage-2 `stage2_ab` / rollout-matching design line. | Historical `progress/benchmarks` and diagnostics are executed; current `docs/training/STAGE2_RUNBOOK.md` is the current owner. | Channel-A/Channel-B conceptual vocabulary, rollout-aligned diagnostics, and failure counters. | Old `n_softctx_iter`, soft-context, or decode knobs are not current entrypoints without revalidation. |
| LEGACY-004 | Stage-2 unlikelihood and pseudo-positive experiments. | Executed checkpoints and diagnostics; one UL capture bug was fixed between related runs. | Need to record capture-version, candidate validity, and rollout denominators. | The post-fix v2 result cannot be attributed to the objective alone when pre-fix capture and truncation differed. |
| LEGACY-005 | Oracle-K and temperature sweeps. | Executed additive analyses with fixed first-200 slices; planning notes propose follow-up controls. | Recoverable-vs-systematic FN split and diversity as a probe of latent support. | Best-of-K is not a single-decode detector, and a temperature from one checkpoint is not a global policy. |
| LEGACY-006 | Painted-GT transcription/steering proposal. | OpenSpec/design/tasks contain intended contract; output reports contain bounded executed route/debug evidence; several larger gates remain planning or not promoted. | Offline materialization, geometry audit, raw-prefix preservation, control identities, denominator-bearing debug metrics. | OpenSpec “SHALL” text is not proof that a GPU gate ran; route reports are not final mAP or architecture promotion. |
| LEGACY-007 | Model-card, smoke-checkpoint, and visualization-manifest surfaces. | Physically present output receipts, often generic or selection-only. | Provenance handles, checkpoint/base metadata, selected-image indices, and artifact paths. | A generic README or gallery does not establish training settings, population metrics, or causal mechanism. |

## Planning versus execution gate

For every historical source, classify the statement as one of:

1. **Plan** — proposed config, design, task list, or OpenSpec intent; useful for
   contract lineage, not execution evidence.
2. **Launch/preflight** — config resolution, parser/schema checks, or smoke
   setup; proves wiring only.
3. **Executed unit** — immutable run with effective kwargs, artifacts, and
   denominator-bearing metrics.
4. **Synthesis** — interpretation over linked executed units; bounded by the
   weakest source scope.
5. **Receipt** — path/hash/selection record with no scientific conclusion.

The handoff should link these classes rather than flatten them into a single
timeline. A planning artifact can motivate a probe; only a linked executed unit
can support a measured result.

## Current authority boundary

Current training, inference, evaluation, and artifact behavior remains in
`docs/` and stable `openspec/specs/`. Research interpretations remain in
`research/`; raw recovered Markdown remains under `docs/history/`. This file
contains no copied commands or current schema. Its job is to stop historical
names from being mistaken for executable current surfaces.
