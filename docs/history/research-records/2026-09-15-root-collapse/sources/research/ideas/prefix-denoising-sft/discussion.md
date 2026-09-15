---
type: idea
title: Prefix Denoising SFT Discussion
description: Debate, audit findings, repair decisions, and unresolved interpretation boundaries for prefix-denoising SFT V1.
tags: [stage1, prefix-denoising, discussion, audit]
updated: 2026-06-20
---

# Prefix Denoising SFT Discussion

## Approval Logic

The research design was accepted as coherent because the clean/noisy pair
directly expresses the intended recovery task: train under a drifted but valid
coordinate prefix while supervising the clean continuation. The design also
keeps the main research variable local by freezing ViT/aligner in the first
quality-bearing run and routing the idea through Stage-1 compact detection
teacher forcing.

The approval was not a claim of usefulness. It was approval to implement a
branch-local experiment with launch-health gates, strict hard-CE semantics,
packing/sidecar isolation, and later matched evaluation before any promotion
claim.

## Audit Findings

The read-only audit found no P0 flaw in the research design, but it did find
blocking implementation risks:

- prefix-denoising sidecars had to be registered at the fail-fast model-input
  boundary;
- packed branch-state detection could not silently default to unpacked
  behavior;
- packed multimodal materialization and boundary-map ownership needed explicit
  tests because a plain attention mask was not enough;
- fallback guidance for high skip rates and milder noise needed care;
- contract tests needed to match actual schema/parser behavior.

Those findings matter because a prefix-denoising result is uninterpretable if
the noisy branch can attend to the clean branch, if sidecars leak into
`model(**inputs)`, or if packed offsets silently break CE/KL site ownership.

## Repair Decisions

The post-audit repair changed the launch-health status. Historical 2026-06-14
smokes proved the paired shape could launch, but they forwarded clean and noisy
branches as one concatenated causal row. The 2026-06-15 repair replayed clean
and noisy branch segments as separate forwards before computing CE and optional
local-window KL.

After repair, the branch recorded CPU tests, focused provenance tests, compile
checks, cfg-only checks, and tiny GPU smokes. The repaired smoke artifacts
recorded prefix-denoising runtime payloads, branch-balanced CE metrics,
standard `llm_loss`/top1/top5 monitors, local-window KL diagnostics, and
dataset skip summaries.

## Interpretation Boundaries

The repaired launch-health evidence is tiny launch evidence only. It does not
prove rollout robustness, exposure-bias improvement, coordinate localization
gain, or superiority over the pure-CE or SoftCE references.

Later analysis found the tested V1 objective likely inert and the baseline
comparison confounded. Therefore the idea remains active but not promoted:
future claims need a matched denoising-OFF hard-CE LoRA control, coordinate
objective comparisons, and decode-side de-confounding before any conclusion.

## Sources

- `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md`
- `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`
- `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`
- `progress/directions/prefix_denoising_sft_v1.md`
- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`
