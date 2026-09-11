---
title: Full-Language DoRA SFT on Train256 with Disjoint Development128
description: Completed ordinary supervised baseline; training owner gains do not transfer to development at any registered IoU50 milestone.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-05-sft256-dev128-baseline
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: complete
updated: 2026-09-05
---

# First scale baseline, not another QP experiment

Closed at the registered stop: [results](results.md) and
[lead acceptance (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-sft256-dev128-baseline/lead-acceptance-v1.json). Technical execution is complete;
scientific disposition is `NO_DEV_IOU50_PROMOTION_AT_REGISTERED_MILESTONES`.
No checkpoint is promoted, and no subsequent RL or optimizer experiment was
launched by this unit. The protocol and failed qualification attempts below
remain provenance rather than current blockers.

## Decision and accepted surface

From original step-2444 Source, how do pure cross-entropy supervised fine-tuning
(SFT) training fit and disjoint natural owner recall evolve on a 256-image
training panel and a 128-image development panel?

The user requested scaling after independent CE replay fit all Human13 owners.
On 2026-09-05 the user explicitly chose full language DoRA (A/B plus magnitude),
with vision, aligner, input embedding delta, and readout frozen. This is a new
baseline parameter surface, not a pure dataset-size ablation of magnitude-only
N13. One shared unmerged adapter starts at Source, never the fitted N13 adapter.

## Available cohort and limits

The parent 256 unique training image IDs exist at:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-scalable-annotated-owner-shared-dora-pilot/g0/cross-image-v1/g0.4-source-signal-v3/execution-requests.jsonl`.

The 128 disjoint development images exist at:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/screen-dev.jsonl`.

Lead read-only checks found 256 versus 128 unique IDs with zero overlap.
The parent execution representation contains 1,955 annotated owners and the
development representation 891. Train object-count bands 1-3/4-7/8-15/16+ are
85/85/53/33 images; development is 43/43/32/10. Therefore aggregate development
recall alone is weak evidence about the densest images.

Reuse the image IDs, not raw execution geometry mixed with processed training
geometry. Rebuild both views from one canonical processed-data authority and
freeze its exact annotation IDs, geometry, ordering, and resulting denominators
before launch. Do not inherit the old Source-event eligibility filter that
reduced this parent set to train248. All valid training examples stay included
even if Source fails to decode them. Genuine data defects are surfaced, not
silently converted into exclusions or empty baselines.

These are official COCO labels, unlike the manually completed Human13 panel.
The existing 128-image screen has been used historically and is development,
not a fresh untouched final test. Valid unmatched detections remain unknown.

## Baseline and decision evidence

First run only ordinary complete-transcript CE, including the standard EOS
target, with a conventional full-DoRA AdamW schedule and no auxiliary loss.
The old `censored_transcript` C config masks EOS and cannot be relabeled pure CE.
Its embedding-freeze behavior is also tied to that profile, so the real entry
must prove the requested updated/frozen parameter surface independently.
Do not carry the magnitude-only learning rate 0.003 over to all A/B parameters.

At frozen milestones measure both teacher-forced diagnostics and unassisted
natural greedy behavior on train and development. Primary transfer evidence is
annotated unique-owner recall at IoU50, with IoU60/80, density strata, gains and
losses, geometry, duplicate/invalid/malformed/cap debt and prediction count
reported separately. Development does not need to reproduce one canonical
ordering. Keep decode budget and runtime fixed across checkpoints.

The strongest alternatives are ordinary training/data/representation limits,
new-image generalization failure, and sequential prefix/stop/grounding failure.
Do not diagnose one solely from low average CE or a teacher-forced success.

## Execution boundary

The user explicitly authorized autonomous implementation and execution on all
eight GPUs on 2026-09-05 and removed the suggested overall wall-time, GPU-hour,
and subagent-count caps. Those suggested caps are not active. This does not
remove the finite scientific stop below, authorize unrelated jobs to be killed,
or turn a wait expiry into experimental failure. Use event-driven long waits
or the wake-me-up plugin; no polling loops.

Existing entry is `python -m src.train`; no new scheduler or generic research
framework is authorized. The lead accepted the native standalone selected-token
embedding freeze option and the canonical inputs below after CPU checks.
Actual eight-rank update/save/eval and independent cold use remain qualification
gates before the full course. No seed sweep or post-outcome schedule extension.

## Frozen first-course profile

- Authoritative inputs are `inputs-v3/train.jsonl` and `inputs-v3/dev.jsonl`
  under `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/`.
  Train SHA-256 `05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5`;
  dev SHA-256 `b2ba42e6be18ca179cbdac387ed7af5b8400cf88a5a7d63f3b46643f2f61a46c`;
  manifest SHA-256 `4697ee57d8ea1ca04fdf2b0aafc7ff1ecbe3e064c372a6f1ec8e6af78a0fca86`.
  All 384 image headers and native-loader annotations were checked; counts and
  density strata above are unchanged, including all eight former exclusions.
- Config: `configs/coordexp_infras/research/qwen3_vl_2b_sft256_dev128_full_dora_ce_seed19.yaml`.
  Full language DoRA A/B/magnitude, rank 16/alpha 32/dropout 0, Source warm start;
  selected embedding/readout delta explicitly frozen, not merely learning-rate 0.
  Its installation/frozen-surface metadata remains present while optimizer
  ownership excludes it. Base/vision/aligner/readout weights stay frozen.
- Native segment-balanced full-transcript CE, including EOS; token-type gate
  weight 0; no annotated-owner, rollout-calibration, RL, or auxiliary objective.
  This is a new conventional full-DoRA baseline, not a strict data-size-only
  comparison to N13's FP32 magnitude-only, total-token-normalized optimizer.
- AdamW language lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0;
  cosine scheduler with zero warmup, gradient clip 1.0, seed 19.
- Eight ranks, one image segment per pack, effective batch 64, expected gradient
  accumulation 8, BF16 training with the existing qualified FA2 branch.
  Maximum 256 optimizer steps; with exactly 256 packs this is 64 image passes
  (16,384 pack presentations), not an inference from the `epochs` label alone.
  Verify the resolved schedule and actual consumption.
- Save steps 16/64/128 and final step 256. Forward development evaluation at
  16/64/128/256. Cold natural generation evaluates Source and all four saved
  checkpoints on both train256 and dev128 using the same native HF FP32/SDPA,
  batch 2, max_new_tokens 3084, temperature 0, top_p 1, n=1, RP1.0 config family.
  Report all milestones, not only a favorable checkpoint. Development is neither
  a fresh test nor guaranteed unseen in Source's earlier training history.

## Qualification, recovery, and ownership

The qualification uses the exact main profile, only reducing train examples to
the first 64 canonical rows and steps to two, with final checkpoint save and
forward evaluation at step two. Keep all 128 development rows so native cold
evaluation also exercises the full eight-worker/batch-two consumer shape.
The first 64 include 36-object images and the observed maximum raw image size;
the main set's maximum object count is 37. No dynamic re-selection occurs.
The full course always restarts Source, not the qualification adapter.

`inputs-v1` failed native absolute-image-reference loading. `inputs-v2` retained
the corrected relative data but had manually asserted/unreproducible manifest
fields. Both remain diagnostic provenance. A fresh Sol recovery owner produced
`inputs-v3` with computed header/count/annotation checks and atomic publication;
its train/dev bytes equal v2, so no scientific label change occurred. The direct
script bootstrap was qualified before v3 publication; no source data was edited.

The bounded routing pilot used Luna/max for the core CE/freeze implementation,
Luna/medium for data preparation, and Sol/medium for an evidenced data-receipt
recovery and a separate read-only execution plan. These are task-specific
observations, not a model ranking. Lead corrections included distinguishing
frozen-delta surface provenance from optimizer inclusion. The broad config suite
also exposed a pre-existing historical-profile inventory failure: 27 already
tracked smoke profiles are outside its old baseline fixture. The new research
config is not in that inventory's prod/smoke roots. Preserve that unrelated test
and fixture rather than rewriting their baseline to make this work appear green.

Freeze code/config/input hashes and the exact long-command owner in launch
packets. Native run completion, checkpoint payloads and frozen tensor equality,
independent inference completion, row coverage, and matcher metrics are required;
process exit, an armed monitor, or green unit tests alone never establish success.

The first real eight-rank qualification (`qualification-launch-v1.json`) stopped
at zero updates with `adapter.dora_source_gate_missing`: the current worktree's
default `outputs/probes/coordexp_swift/dora_roundtrip/receipt.json` was absent.
Its failed native run, config, and log remain under `qualification-v1/`; no cold
inference was attempted and no model-quality result is claimed. Recovery must
admit the existing real source-study/roundtrip gate through valid evidence,
never disable that gate or restore the censored-transcript profile as a bypass.
The production native entry, not the CPU mocks, exposed this missing runtime
asset. A fresh Sol owner is resolving this dependency before a new qualification.

Source-gate recovery used the existing real probe, not a new synthetic pass:
the exact absent worktree-local `outputs/probes/coordexp_swift/dora_roundtrip`
now links to `/data/CoordExp/.worktrees/research-probes/outputs/probes/coordexp_swift/dora_roundtrip`.
Receipt SHA-256 is `3b7c9f502e50e218c2ed2a0f882fec5d4710702e0a097dcbf9c85673c2e28205`.
Its torch/transformers/peft/safetensors versions match the current environment;
the unchanged native gate admits language/all_linear warm-start DoRA. This is
a small library-mechanics roundtrip (four targets), not proof of the full model's
196-target/eight-rank path. `qualification-v2` repeats the same two-step profile
under a fresh run identity with this dependency present. No guard was bypassed,
no source data was changed, and the first failed run was not overwritten.

Qualification v2 also stopped before any update, now at the separate selected
embedding roundtrip gate. A source-gate closure then checked both unconditional
startup gates together. The second exact absent leaf link points to the retained
`special_token_embeddings_roundtrip` artifact; receipt SHA-256 is
`7da3b22b11ef8a0957bedfe78cac16498313cf724e87cfc921caa0ab7c9f68e4`.
Runtime/Source/probe selections exactly agree on 1,004 tokens (four wrappers plus
1,000 coordinate tokens), shared additive-delta semantics, and library versions.
The replayable CPU preflight and receipt are preserved under the output root's
`source-gate-preflight-v1/`. No new synthetic gate or GPU probe was needed.

Qualification v3 completed the actual eight-rank training path: two finite,
applied steps, 16 microsteps per rank, forward evaluation at step two, and final
unmerged checkpoint `step-2`. It contains 588 adapter tensors, SHA-256
`0316d32ecb310248736d541cf83282562120c720e452e344bd59ce26ac6d71ef`.
The frozen selected-delta payload remains byte-identical to Source, SHA-256
`a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`.

Its first cold config was rejected before GPU use because native production
inference requires the canonical `configs/coordexp_infras/infer` namespace and
explicit leaf-authored generation/scoring settings. The corrected canonical
leaf declares the same numerical settings; `qualification-cold-launch-v1.json`
owns that one cold-only invocation. Do not repeat the successful training to
repair this consumer configuration. Qualification cold inference subsequently
completed with all 128 unique rows in canonical input order, zero parser/score/
image/cap failures, and 128 native `im_end` stops. The 17 dropped predictions
remain visible diagnostics, not silently removed debt. The native unmerged
adapter and frozen selected-delta identity checks passed. The lead replayed the
actual counters, ordered IDs/image paths, checkpoint keys and Source delta
bytes; `qualification-lead-acceptance-v1.json` freezes the evidence hashes.

`benchmark_eligible=false` is expected: native inference requires at least 200
rows for that flag (`src/inference/artifacts.py`); dev128 is fixed-cohort research,
not an official benchmark. `diagnostic_row_count=128` counts row diagnostics,
not failed rows. Do not alter these flags or expand the cohort to make them green.
Qualification proves execution mechanics only. It admits the original-Source,
eight-rank, 256-step main course frozen by `main-launch-v1.json`, with one durable
tmux invocation and a terminal success/failure log for callback monitoring.
The measured cold decode estimate is maximum-rank decode time (179.18 s), not
observed controller wall time; peak allocated/reserved is 10.94/12.21 GB per rank.

The main native run completed all 256 finite/applied updates across eight ranks,
2,048 microsteps per rank and 16,384 eligible segment presentations, with no
skipped segments. Native created-to-completed wall time was 1,522.54 seconds
(training plus four forward evaluations, excluding launcher overhead). Every
step-16/64/128/256 checkpoint has all 588 finite tensors changed from Source
(196 A, 196 B, 196 magnitude); its frozen selected-delta bytes equal Source.
`training-lead-acceptance-v1.json` binds the checkpoint and native artifact hashes.

Development forward CE at steps 16/64/128/256 is respectively
1.397071 / 1.435613 / 1.502298 / 1.547337. This rising teacher-forced loss is an
overfitting warning, not yet a natural-generation owner-recall result. Complete
the frozen Source-plus-four-milestones natural curve on both splits without
discarding late points. `launch-natural-eval-v1.sh` owns one serial ten-arm,
eight-worker native inference batch followed by the existing matcher reducer;
it stops on the first command failure and never retries or overwrites an arm.

Close after one baseline learning curve and its fresh natural validation.
Reinforcement learning (RL) is a subsequent explicit contrast: from the same
selected SFT anchor compare continued CE with refreshed RLOO at declared cost,
before testing an owner-credit innovation. QP, new margin losses, blind EOS
suppression, and static actual-prefix positive-only rescue are not this unit.

This exclusion is per-unit scope, not scientific abandonment. The user explicitly
retained QP/hidden-state/shared-DoRA optimization and self-prefix matcher credit;
their current evidence, existing code, and migration caveats are recorded in the
[compass](../../compass.md). Existing Human13 row/token return-to-go code should
be reused where its semantics apply, not advertised as a new implementation;
its unmatched-negative and nonpositive-EOS-advantage rules cannot silently become
the partial-COCO RL baseline.
