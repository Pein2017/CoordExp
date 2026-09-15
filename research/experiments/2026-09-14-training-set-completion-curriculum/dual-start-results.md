# Paired parameter starts: bounded batch result

Status: lead-accepted; bounded batch closed. All-known training-set completion remains incomplete.

## Decision and scope

At256 new updates, both starts reach218/218 on the versioned COCO-80 teacher,
FN=0, annotation-relative micro F1=1.0, with218 valid predictions and zero
malformed, geometry-invalid, non-COCO description, repeat, confirmed-FP,
physical-unknown, or EOS/cap debt. All218 matches also pass IoU>=0.8.
The frozen final-endpoint comparison is a tie.

A reaches this clean result at the first saved diagnostic readback, new step64,
and retains it at128/256. B still has8 FN and297 invalid rows at128, then reaches
the same result at256. This establishes earlier observed learning for A and
learnability of original B under this teacher/dose. It does not establish an
exact hitting time, convergence, transfer, or a persistent final-quality gap.

Retain **A final new-step256 as the operational mainline**, consistent with the
user's continuity preference; preserve B final new-step256 as an equally qualified
endpoint and a positive earlier-start learnability control. This operational
choice is not a claim that A has superior final FN/F1 or that later checkpoints
are inherently better. No next training job is automatically authorized.

## Frozen contrast

- A: historical fourth-fit step256 adapter; B: original geo_sorted_xy step2444
  adapter. Both use the bound step2444 additive special-token embedding delta.
- Same11 original images,218 trusted COCO-80 owners,2088 active tokens; retained
  sequence order and EOS unchanged across arms. No refresh or new teacher generation.
- Fresh seed42 AdamW, lr1e-5,256 updates, language DoRA only; base/vision/embedding/
  lm_head frozen. HF fp32 SDPA, model.eval, gradient clipping1.0.
- CE: per-image active-token mean, then global11-image mean. Four ranks per arm
  own3/3/3/2 images; loss/11 backward then gradient SUM preserves that objective.
- Shared expected-coordinate raw-axis hinge: weight0.01, margin1/999.
- Original image, empty assistant prefix, greedy temperature0/top_p1/top_k0/RP1,
  cap3084, EOS151645.88 cold readbacks cover both sources and64/128/256.

## Natural greedy trajectory

Primary matching is class-agnostic cardinality-first one-to-one IoU>=0.5.
F1=2TP/(218+all valid parsed predictions); annotation-unmatched is not physical FP.
Earlier points are diagnostics; the predeclared decision endpoint remains256.

| Arm | New updates | Covered | FN | F1 | Valid predictions | Invalid rows | Natural EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 209/218 | 9 | 0.9310 | 231 | 18 | 11/11 |
| A | 64 | 218/218 | 0 | 1.0000 | 218 | 0 | 11/11 |
| A | 128 | 218/218 | 0 | 1.0000 | 218 | 0 | 11/11 |
| A | 256 | 218/218 | 0 | 1.0000 | 218 | 0 | 11/11 |
| B | 0 | 89/218 | 129 | 0.1969 | 686 | 497 | 8/11 |
| B | 64 | 166/218 | 52 | 0.3144 | 838 | 863 | 6/11 |
| B | 128 | 210/218 | 8 | 0.5949 | 488 | 297 | 9/11 |
| B | 256 | 218/218 | 0 | 1.0000 | 218 | 0 | 11/11 |

Final218 matched descriptions are also correct in both arms. Source/intermediate
physical-unknown counts remain unresolved diagnostics; their annotation-unmatched
predictions are not converted to confirmed FP. The final endpoints have no
unmatched predictions, so no new visual adjudication or annotation delta is needed.

## Old-owner retention and the historical boundary

| Arm | Ledger | Retained from own source | Gained | Lost |
|---|---|---:|---:|---:|
| A | scoped218 | 209 | 9 | 0 |
| A | historical232 | 209 | 9 | 3 |
| A | current-known248 | 209 | 9 | 4 |
| B | scoped218 | 89 | 129 | 0 |
| B | historical232 | 89 | 129 | 6 |
| B | current-known248 | 89 | 129 | 7 |

Both final endpoints cover exactly the same218 owners in every ledger. Neither
loses an owner inside the scoped218 teacher. The historical232 losses are excluded
category-unresolved owners; they remain recorded. Current-known248 includes one
additional lost unknown owner for A and one additional verified person for B,
beyond each arm's historical232 losses. Thus scoped success is not universal
old-owner preservation. See the exact image-qualified comparison receipts.

Final coverage is218/232 and218/248, leaving14 and30 missing respectively. The30
current-known misses comprise **9 verified COCO-80 owners,19 class-unknown owners,
and2 verified non-COCO owners**. They were outside this round's218 teacher. Keep
all frozen ledgers and existing annotations; no denominator is silently replaced.

## Engineering and observed cost

Shared loss is enabled by default through
`losses.protected.raw_axis_validity_hinge.weight=0.01`; weight0 disables it.
LossContext/CoordinateLossTarget grouping uses example/segment/object/slot.
Complete boxes average within each eligible segment, then segments average equally;
eligible zero-box segments contribute0. Partial groups are counted/skipped, and
missing/inconsistent supervision mapping fails closed. Ordinary and streaming
paths share the formula used by the research probe.

Verification includes218 core/packing/pipeline/runtime tests,66 pack-cache tests,
98 scoped config tests, a real shared-entry update/checkpoint smoke, and real
four-rank-per-arm qualification. Root reproduced and rechecked an empty-only
context bug before acceptance. One full config-inventory assertion remains
incompatible with unrelated dirty profile additions; no such files were removed.

Both full arms completed256 updates/2816 logical image forwards. Training elapsed
1322.75s for A and1315.87s for B (about22min each, concurrently on GPUs0-3/4-7).
Training plus88 readbacks took2557.96s, about42.63min wall time. All10 subprocesses
exited0. Qualification consumed44 additional logical image forwards; its first
manifest-admission failure consumed0. Logical forwards exclude checkpointing
recomputation and autoregressive decode steps; GPU utilization is not an
efficiency proof. Full measured resource receipts are retained separately.

## Interpretation and next decision

Both initializations can realize this compliant teacher cleanly through natural
greedy. A's sampled trajectory is faster; B's early structural deterioration
does not imply inability to learn. Shared geometry's integration is accepted,
but this two-start experiment cannot attribute gains to geometry, the scope
filter, optimizer reset, or any single difference from historical fourth fit.

The proposed sample-equal versus token-equal CE ablation remains unlaunched.
Final256 on this218-owner cohort now has a ceiling in both starts, so repeating
that saturated endpoint alone has limited discriminating value. A later experiment
needs a declared learning-speed or changed-support question; normalization and
new-support changes should still have matched controls. The9 already verified
COCO-80 misses are concrete candidates for a separately authorized support
expansion. The19 unknown classes and2 non-COCO owners retain their unresolved or
out-of-scope status. No throughput, normalization, refresh, or expansion arm was
added to this completed batch.

## Evidence

- [Frozen release](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/release.json)
- [Teacher admission](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-lead-acceptance-v1/teacher.json)
- [Shared-loss acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-lead-acceptance-v1/shared-geometry.json)
- [Evaluation receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/evaluation-v1/evaluation-receipt-v1.json)
- [Endpoint and per-image table](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/evaluation-v1/aggregate-and-per-image-v1.json)
- [Own-source A retention](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/evaluation-v1/A-step-000-to-256.json)
- [Own-source B retention](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/evaluation-v1/B-step-000-to-256.json)
- [Final cross-arm comparison](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/evaluation-v1/A-versus-B-step-256.json)
- [Curves PNG](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/lead-closeout-v1/dual-start-curves.png), [PDF](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/lead-closeout-v1/dual-start-curves.pdf), [CSV](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/lead-closeout-v1/endpoint-metrics.csv)
- [A final adapter](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/A/training/checkpoints/step-00256/adapter)
- [B final adapter](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/B/training/checkpoints/step-00256/adapter)

The earlier misattributed B baseline remains superseded: it loaded an intervening
N16 adapter. This report uses only fresh original-B source0 evidence.

- [Lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/lead-closeout-v1/root-acceptance.json)
- [Lead full-dose/checkpoint replay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/lead-closeout-v1/lead-technical-replay.json)

- [Full finite-state and resource audit](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/technical-acceptance-v1.json)
- [Technical lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/dual-start-v3/lead-closeout-v1/technical-lead-acceptance.json)

Measured training-plus-readback GPU-worker-time projection is13978.80 seconds
(3.883 GPU-hours), excluding qualification and model-setup intervals absent from
worker timers; it is not a synchronized device-utilization measurement. Training
geometry hinge averaged1.8658e-5 in A and0.0027998 in B; clipping occurred6/256
and84/256 updates respectively. These are trajectory diagnostics, not isolated
causal evidence for normalization or geometry efficacy.
