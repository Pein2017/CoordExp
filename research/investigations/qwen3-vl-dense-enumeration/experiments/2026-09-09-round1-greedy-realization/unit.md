---
title: Immediate greedy realization after retained Source256 RLOO round1
description: One cold train256 read closes the missing immediate outcome after the first historical RLOO update.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-09-round1-greedy-realization
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-09
---

# Authorized question

From original Source step2444, does its already executed first Source256 RLOO
update improve natural-greedy annotated-object coverage and F1 on the same
train256 panel, and which previously observed candidate gains are realized?

The user accepted the [retained-candidate result](../2026-09-09-natural-candidate-opportunity/results.md)
and authorized continuing with this exact missing checkpoint read. This is
not a new optimizer run, objective, random sample bank or architecture. Root
owns launch, source acceptance, interpretation and these records. The CPU
consumer owner owns only `probes/dora_owner_learning/round1_realization.py`,
its tests and its versioned reduction output. Prior accepted outputs remain
immutable; unrelated dirty work remains untouched.

# Frozen contrast and inputs

- Before: original Source train256 natural greedy retained under
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1/qwen3-vl-2b-sft256-source-train256-natural-v1`.
- After: exactly the saved first RLOO update adapter under
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1/update/adapter`.
  Receipt adapter fingerprint `5f55aa01188c67bc3d594cdf29178fb77db8b2a96a4b22d3b28c0db42642712d`.
- Same256 train images /1955 GT annotations, original selected-token embedding,
  prompt/template, geo_sorted_xy data, no resize, unmerged DoRA, FP32/SDPA,
  patch-embed linearization enabled, T0/top-p1/RP1/max_new_tokens3084, one
  completion per image and per-device batch2.
- New authored config:
  `configs/coordexp_swift/infer/source256-rloo-round1-train256-realization-v1.yaml`.
  Only adapter path and run name/root differ from the maintained Source profile.
- Output root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-round1-greedy-realization`.
  `prelaunch.json` binds source/config/adapter bytes and the absent final path.

# Outcomes and interpretation

Primary is category-consistent global one-to-one annotated-object matching
at IoU50. Report all256 images, TP/FP/FN/F1/recall, gained/lost/retained object
IDs, valid/invalid predictions, category-independent later pixel-IoU>.95
repeats, parser drops, complete action length and stop/caps. Carry IoU60/80.
Never filter capped or imperfect outputs. FP remains annotation-relative,
not a claim of hallucination or exhaustive-label precision.

The fixed previous K4 candidate scorecards define supplementary opportunity
strata; they do not select the evaluated population or replace the full
primary comparison. Report actual post-update coverage of candidate-gained
objects and paired losses, making unique-object counts versus per-candidate
incidences explicit. A candidate union is not a single trajectory. Signed
RLOO advantage is objective supply, not a measurement of likelihood change.

Strongest alternatives to a learning-failure claim are a small fixed dose,
likelihood movement insufficient to change greedy choices, interactions with
other action gradients, and localization/object exchanges. This read identifies
the immediate natural-greedy outcome, not these causal mechanisms. No new
before/after likelihood scoring or training follows automatically. Historical
round4 is a CPU fixture only and must never be relabeled this round1 result.

# Execution bound and stop

One full cold inference pass using current `python -m src.infer --config ...`:
eight currently idle A100-80GB devices, one worker/model load per device,
128 batch2 requests, at most789504 generated action tokens across all256
images, no materialized merged checkpoint, and existing selected-token scoring
from generation traces (no separate likelihood-replay job). Historical same-
shape cold reads suggest roughly20-40 minutes and about20GB peak reserved
CUDA memory per worker; these are estimates, not fresh measurements.

Hard inference wall ceiling:3600 seconds, then interrupt only this invocation
and stop for a user decision; allow at most60 seconds for process cleanup.
The corresponding allocation ceiling is8 GPU-hours plus bounded cleanup.
Do not automatically relaunch, enlarge the cap, evaluate development or read
confirmation512. Preserve any failure output. A launch/identity error is
technical failure, not a negative quality result.

Before launch, validate current YAML, input identity and adapter-file hashes
against the retained update receipt. The full run is the production-shaped
execution slice; require all ranks terminal, complete native raw/scored/trace/
image-plan/config/summary artifacts, exact row and prompt/media alignment, and
resource release. The CPU consumer must reproduce historical Source counts;
its declared historical round4 fixture must reproduce1278/1209/923 without
claiming round1 execution. Root validates exact fresh results and runs the
current detection evaluator as a separate artifact-consumer check.

Stop after this one accepted inference/evaluation and bounded interpretation,
whether positive, mixed or null. No best-checkpoint selection, new sampling,
optimizer step, architectural promotion, or implicit extension of the study.

# Closure

The one authorized inference pass completed2026-09-09 16:16:07UTC, exit0,
all eight ranks and256 merged images. The current CPU consumer and native
evaluator passed. The [results](results.md) record TP50+3 with F1 decrease and
zero realization of the65 distinct strong-witness gained objects, including
the61 appearing in positive-advantage candidates. The result identifies weak
immediate greedy realization at this one-update dose, not candidate likelihood
change or a general inability to learn. All GPUs are released; the stop is met.
