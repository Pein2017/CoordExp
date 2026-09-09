---
title: Image2299 layerwise Logit Lens checkpoint contrast
description: Matched self-prefix readouts of sorted-xy Source and Human13 CE overfit, with DeepStack boundary instrumentation.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-08-image2299-logit-lens
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-08
---

## Question, authority, and scope

From sorted-xy Source step-2444, does the existing Human13 cross-entropy
overfit adapter change layerwise next-token readouts on identical Image2299
self-generated prefixes, under exact native-logit and DeepStack-boundary checks?

The user authorized the fastest existing Hugging Face (HF) exploratory path,
Image2299 or Human13, and delegated pre/post injection placement. The subsequent
user refinement explicitly requested sorted-xy versus previously overfitted
checkpoint comparison to investigate dynamics. This is a new descriptive probe,
not a continuation or gate of the separate row-feedback training experiment.
No training, intervention, new framework, model architecture change, activation
patching, Human13 expansion, or generalization claim is authorized by this unit.

Primary scientific product: paired layerwise readout differences conditional
on the *same* image and prefix. There is no behavioral-treatment or causal
estimand. Apparent earlier decodability does not imply earlier causal use.
Strongest alternative: checkpoint-dependent representation alignment to the
final output head, or memorized route/prefix compatibility rather than a
general covered-set computation. Image2299 belongs to the overfit training panel.

## Anchors and identities

- Worktree: `/data/CoordExp/.worktrees/image2299-logit-lens`, branch
  `probe/image2299-logit-lens`, forked from research-probes HEAD
  `73b8b3cc2614db052055c7c49fc33d6207b0f80a`.
  Source worktree had only unrelated guidance deletions at admission; its
  tracked commit, not dirty changes, was forked. New worktree was clean.
- Base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
- Source checkpoint:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`.
  Adapter SHA256 `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`.
  Original training `resolved_config.json` declares `object_ordering: geo_sorted_xy`.
- Overfit adapter:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-human13-pure-ce-replay/run-v2/adapter`.
  Adapter SHA256 `8a5ebfcacfa92be4b873fea4439fc25570a9c415be2245e20fd1da94c9ff4070`.
  This is the 140-step Human13 magnitude-only cross-entropy replay, not output-QP.
- Both use Source `special_token_embeddings`; delta SHA256
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`.
- Reuse config `configs/coordexp_swift/infer/qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml`;
  override repetition penalty (RP) to 1.0 and generation cap to 768 for both.
  RP1.0 matches the overfit acceptance surface and removes repetition processing
  as an explanation of next-token preference. No RP sweep.
- Input: Image2299 from
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl`.
  One image, 46 annotated owners including 38 people; these are provenance,
  not a new owner-coverage evaluation denominator.
- Output root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-image2299-logit-lens`.
  Run receipts bind effective config, model/adapter/delta, image/media, prompt,
  tokenizer, exact prefix tokens, implementation hashes, and runtime versions.

## Smallest executed design

1. Generate one native greedy RP1.0 trajectory per checkpoint, cap 768 tokens.
   Distinguish EOS from cap. No exact historical generation or full-coverage claim.
2. Replay both trajectories on both checkpoints: two prefix origins by two
   adapters. These are self-generated-prefix teacher-forced replays, not GT
   supervision and not four independently free-running trajectories.
3. Select at most 24 text sites per trajectory deterministically: prompt end,
   first/middle/last completed-row boundaries, opener/description and coordinate
   decisions, terminal decision if present. Bind identical sites across adapters.
   State at position t predicts token t+1; no unique correct next owner is assumed.
4. Read each decoder residual using the existing final norm and output head.
   Report selected top-k tokens, actual-next-token probability/rank, and raw
   opener-minus-EOS margin. Keep both prefix-origin strata separate.
5. Capture pre- and post-DeepStack injection. The installed implementation adds
   only to visual positions, in place: clone pre-state before the addition.
   Immediate text-site equality is expected, not a failed probe. Record visual
   state delta counts/norms; do not interpret image-token LM-head words as objects.
   Downstream text changes are not isolated causal injection effects.

## Mechanical acceptance and resource stop

Lead owns research meaning and final acceptance. Worker `lens_runner` owns only
the experiment-local runner/tests and its single live invocation; shared HF code
is reused unchanged. No model weights are updated or saved.

- Hooked and unhooked selected native raw logits must agree within predeclared
  atol=2e-4, rtol=2e-4; final-layer norm/head readout must also agree. Never loosen
  tolerance after observing failure. Last normalized state is not normalized twice.
- Verify shifted next-token identity, finite summaries, identical paired prefix
  identities, expected injection sites, no hook-induced mutation, and durable
  write/read counts. Recompute decision-bearing summaries from saved artifacts.
- GPU0 initially, one process at a time; default `ms` environment, FP32/SDPA.
  Maximum two technical attempts and 30 minutes total GPU execution for the pair.
  Two native generations and four selected-site replays; bounded parity forwards
  are additional, recorded rather than hidden. No full hidden/logit cube storage.
- Record wall time, forwards, token/site counts, peak GPU/host memory and payload
  bytes. Save failed receipts; unsupported behavior leaves affected science unknown.
- Stop after this single-image paired result or the technical/resource bound.
  Stable patterns may nominate a future causal experiment; no automatic successor.

## Evidence and continuation

Closed after the paired Image2299 run; see [results](results.md) and
[lead acceptance](lead-acceptance.json). `run-v2/receipt.json` under the output
root owns execution counters/checks. Attempt 1 failed closed at the missing
embedding source-gate evidence binding, before GPU allocation. Attempt 2 staged
hash-verified existing gate evidence without relaxing shared validation; all
four replays passed exact hook parity and final readout reconstruction.
The diagnostic is mechanically accepted and scientifically descriptive only.
No successor, commit, publication, or records-only return to the protected
research authority was executed. The separate feedback experiment is unaffected.
