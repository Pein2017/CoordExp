---
doc_id: progress.diagnostics.et_rmp_continuation_diagnostics
layer: progress
doc_type: diagnostic-study
status: canonical-cluster-entry
domain: stage1-et-rmp-ce
summary: Canonical progress-layer source for the pre-support-mass ET-RMP continuation diagnostics: objective contract, val200 behavior, RP sweeps, representative sample bank, FN latent probes, length/stop pressure, and stop-control ablations.
tags: [stage1, et-rmp-ce, repetition-penalty, continuation-bias, dense-detection, sample-bank, fn-diagnostics, stop-pressure]
updated: 2026-05-01
---

# ET-RMP Continuation Diagnostics Source

This is the canonical `progress/diagnostics/` entrypoint for the ET-RMP
continuation behavior investigation started around the pre-support-mass
enhancement Stage-1 set-continuation run.

Use this note to answer:

- what ET-RMP changed relative to old candidate-level MP;
- what the first ET-RMP run appeared to fix;
- where it under-generated;
- how repetition penalty changed rollout behavior;
- what the fixed representative sample bank contains;
- what the FN probability probes showed;
- whether stop/close pressure grows with length;
- whether hard stop-control recovered hidden missed objects.

Do not use this note as the implementation source of truth. Current objective
and config behavior belongs in:

- [docs/training/STAGE1_ET_RMP_CE.md](../../docs/training/STAGE1_ET_RMP_CE.md)
- [docs/training/STAGE1_OBJECTIVE.md](../../docs/training/STAGE1_OBJECTIVE.md)
- [configs/stage1/set_continuation/production.yaml](../../configs/stage1/set_continuation/production.yaml)

## Scope Guard

Most behavior conclusions here are about the old ET-RMP `v1` checkpoint before
support-mass enhancement.

Primary old ET-RMP checkpoint:

```text
output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_v1/setcont-coco1024-sota1332-et-rmp-ce-v1/v0-20260429-022918/checkpoint-300
```

Base comparison checkpoint used in later core-6 probes:

```text
output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full
```

Current support-mass-enhanced production profile is a later config surface:

```text
configs/stage1/set_continuation/production.yaml
```

It sets `branch_support_weight: 2.0` and `branch_balance_weight: 1.0`. Do not
generalize the old `v1` rollout behavior to that profile unless the same
representative bank and probes are rerun.

## Objective Contract

ET-RMP means **Entry-Trie Recursive Multi-Positive CE**.

The objective is still 100 percent teacher-forced over a complete continuation
sequence:

```text
prefix -> entry(o1) -> comma -> entry(o2) -> ... -> final_close -> EOS
```

The difference from random-order full-suffix SFT is inside object entries:

- Build a trie over the serialized object entries for all currently remaining
  objects.
- At every trie node with multiple valid next-token children, use a
  multi-positive branch loss over the valid child tokens.
- At every non-branch trie node, use ordinary hard-label CE.
- Boundary comma, final close, and EOS remain ordinary hard-label CE.
- Boundary/schema tokens are not part of the object-entry branch supervision.
- The main probability space is full vocabulary, aligned with autoregressive
  decode.

This is different from previous candidate-level MP:

| Aspect | Old candidate MP / candidate-balanced CE | ET-RMP |
|---|---|---|
| Unit | One candidate continuation chunk | Full remaining suffix |
| Supervision shape | Scores selected candidate chunks | Teacher-forced full sequence |
| Multi-positive location | Candidate score level | Entry-trie divergence tokens |
| Boundary tokens | Can contaminate chunk scores | Hard CE, outside branch MP |
| Decode alignment | Chunk score surrogate | Token-level autoregressive CE |
| Rollout closure | Not guaranteed after one object | Trains recursive object/comma/close/EOS closure |

Support-mass enhancement is a later extension of the ET-RMP branch loss:

```text
L_branch = branch_support_weight * L_valid_support
         + branch_balance_weight * L_object_uniform_balance
```

`branch_support_weight=1.0` and `branch_balance_weight=1.0` recover the earlier
object-uniform ET-RMP behavior. The current production experiment raises
support mass with `2.0/1.0`; this note mostly studies the earlier `1.0/1.0`
behavior.

## Artifact Map

Durable progress artifact copies:

```text
progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/
progress/diagnostics/artifacts/et_rmp_continuation_diagnostics_2026-05-01/
```

Important copied summaries:

| Evidence | Progress copy |
|---|---|
| Fixed representative visual bank | [artifacts/et_rmp_rp_sample_bank_2026-04-29/README.md](artifacts/et_rmp_rp_sample_bank_2026-04-29/README.md) |
| Core-6 deterministic RP/temp sweep | [artifacts/et_rmp_continuation_diagnostics_2026-05-01/core6_deterministic_sweep_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/core6_deterministic_sweep_summary.md) |
| Core-6 stochastic RP/temp/seed sweep | [artifacts/et_rmp_continuation_diagnostics_2026-05-01/core6_stochastic_sweep_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/core6_stochastic_sweep_summary.md) |
| Latent/FN continuation probe | [artifacts/et_rmp_continuation_diagnostics_2026-05-01/latent_probe_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/latent_probe_summary.md) |
| Boundary length-bias probe | [artifacts/et_rmp_continuation_diagnostics_2026-05-01/length_bias_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/length_bias_summary.md) |
| Strict stop-control rollout sweep | [artifacts/et_rmp_continuation_diagnostics_2026-05-01/stop_control_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/stop_control_summary.md) |
| Diagnostic stop-control salvage | [artifacts/et_rmp_continuation_diagnostics_2026-05-01/stop_control_salvage_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/stop_control_salvage_summary.md) |

Runtime artifact roots are still useful for raw predictions and traces:

```text
output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_v1/setcont-coco1024-sota1332-et-rmp-ce-v1/v0-20260429-022918/
output_remote/infer/coco1024_val200_et_rmp_ce_step300_rp100
output_remote/infer/coco1024_val200_et_rmp_ce_step300_rp105
output_remote/infer/coco1024_val200_et_rmp_ce_step300_rp112
output_remote/infer/coco1024_val200_et_rmp_ce_step300_rp115
output_remote/infer/coco1024_val200_et_rmp_ce_step300_rp118
output_remote/infer/core6_stopctrl_*
```

## First Training Read

The old ET-RMP run was evaluated on `val200` callback snapshots at steps 100,
200, and 300. These evals used strict JSON parsing, `max_new_tokens=3084`, and
`repetition_penalty=1.10`.

| step | bbox AP | AP50 | AR100 | TP@0.50 | FP@0.50 | FN@0.50 | pred total | invalid JSON | empty pred | hit max tokens | token mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.4102 | 0.5479 | 0.4590 | 791 | 164 | 653 | 999 | 0 | 0 | 0 | 238.8 |
| 200 | 0.4107 | 0.5444 | 0.4599 | 758 | 118 | 686 | 908 | 0 | 0 | 0 | 203.8 |
| 300 | 0.4205 | 0.5623 | 0.4690 | 795 | 164 | 649 | 992 | 0 | 0 | 0 | 239.5 |

Observed training-side behavior:

- JSON/format stability was restored compared with the old one-step
  candidate-balanced objective symptoms.
- Empty predictions and max-token truncation were gone in these callback evals.
- The run still under-generated relative to the starting reference in
  high-count scenes.
- High-count buckets remained the main failure surface.

The mechanism read after step 100 and step 300 was therefore:

```text
ET-RMP fixes rollout hygiene much better than old one-step MP,
but it does not automatically preserve starting-checkpoint coverage.
```

## Val200 Repetition-Penalty Sweep

The next question was whether decode-time repetition penalty could reveal
objects that the model otherwise failed to emit. These are `val200` results for
the old ET-RMP step-300 checkpoint.

| rp | AP | AR100 | TP@0.50 | FP@0.50 | FN@0.50 | pred total | read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.00 | 0.3839 | 0.4214 | 761 | 390 | 683 | 1163 | Too many FP/duplicates. |
| 1.05 | 0.4060 | 0.4450 | 770 | 174 | 674 | 959 | Better precision, still under-recall. |
| 1.10 | 0.4205 | 0.4690 | 795 | 164 | 649 | 992 | Strong default in callback eval. |
| 1.12 | 0.4200 | 0.4709 | 792 | 159 | 652 | 996 | Similar AP, slightly lower FP. |
| 1.15 | 0.4190 | 0.4744 | 803 | 166 | 641 | 1037 | Best FN count in this set. |
| 1.18 | 0.4155 | 0.4742 | 801 | 170 | 643 | 1077 | Emits more, AP softens slightly. |

Important observation:

```text
Higher rp was not monotonically more conservative.
```

From `rp=1.10` to `rp=1.18`, the model emitted more objects and had similar or
better FN counts, even though AP softened after `1.15`. Because COCO labels are
incomplete in dense scenes, mild FP increases were not treated as definitive
quality loss without visual review.

## Representative Sample Bank

The fixed bank avoids re-reviewing all 200 validation images. Use `core_6` from
`research_subset.json` as the default few-shot analysis set.

| Tag | Image idx | Source image | Primary reason |
|---|---:|---|---|
| `benefit_121` | 121 | `images/val2017/000000012639.jpg` | `rp=1.18` gains TP and reduces FN. |
| `benefit_010` | 10 | `images/val2017/000000001268.jpg` | `rp=1.18` recovers extra objects with mild FP. |
| `benefit_158` | 158 | `images/val2017/000000016010.jpg` | `rp=1.18` opens up a previously under-emitting sample. |
| `hurt_025` | 25 | `images/val2017/000000002157.jpg` | `rp=1.18` sharply under-predicts versus `rp=1.10/1.15`. |
| `hurt_061` | 61 | `images/val2017/000000006471.jpg` | `rp=1.18` loses high-quality matches. |
| `hurt_178` | 178 | `images/val2017/000000017959.jpg` | Dense kite/crowd case: `rp=1.18` shifts toward many tiny people while missing kite coverage. |

The kite/crowd review belongs to `hurt_178`. Many `rp=1.18` person boxes appear
to lie over genuinely dense crowd regions, where COCO annotations may not be
exhaustive. This is why later evaluation discussions separated confirmed FN
objects from possibly-unlabeled FP objects.

## Core-6 Base vs ET-RMP Sweeps

The `core_6` subset is not a benchmark replacement. It is a stress slice for
manual and mechanism analysis.

Deterministic sweep highlights:

| model | best shown cell | AP | TP@0.50 | FP@0.50 | FN@0.50 | pred |
|---|---|---:|---:|---:|---:|---:|
| base1332 | `rp=1.10,temp=0.2` | 0.4228 | 50 | 18 | 37 | 74 |
| ET-RMP step300 | `rp=1.15,temp=0.2` | 0.4068 | 47 | 16 | 40 | 68 |
| ET-RMP step300 | `rp=1.18,temp=0.5` | 0.3982 | 37 | 10 | 50 | 54 |

Stochastic sweep highlights:

| model | cell | AP | TP@0.50 | FP@0.50 | FN@0.50 | pred |
|---|---|---:|---:|---:|---:|---:|
| base1332 | `rp=1.05,temp=0.1,seed=29` | 0.4559 | 55 | 20 | 32 | 78 |
| base1332 | `rp=1.05,temp=0.2,seed=29` | 0.4590 | 55 | 19 | 32 | 77 |
| ET-RMP step300 | `rp=1.18,temp=0.35,seed=17` | 0.4399 | 42 | 12 | 45 | 59 |
| ET-RMP step300 | `rp=1.15,temp=0.35,seed=17` | 0.4309 | 42 | 18 | 45 | 66 |

Takeaway from the stress slice:

- Base checkpoint remained stronger on raw TP/FN count for this small set.
- ET-RMP step300 often had lower FP but also lower emission count.
- The ET-RMP high-`rp` sweet spot on core-6 was not identical to val200.
- This supports using core-6 for targeted mechanism checks, not global ranking.

## Duplicate-Cluster Read

The known Qwen3-VL detection symptom is same-desc duplicate burst or duplicate
collapse. The RP sweeps did not show a simple monotonic duplicate-burst story:

- `rp=1.00` on val200 had high FP and low AP.
- `rp=1.10` to `1.18` reduced the worst FP pressure but did not simply reduce
  emitted object count.
- In the strict stop-control core-6 sweep, same-desc IoU>=0.85 duplicate
  clusters were not catastrophic: max cluster size was 1 for base rows and 2 for
  ET-RMP rows in the salvage summary.

This does not eliminate duplicate collapse as a broader failure mode. It means
the main old-ET-RMP diagnosis on these slices was conservative coverage and
continuation, not uncontrolled duplicate burst.

## FN / Latent-Emission Probes

The fixed-bank FN probes asked whether missed GT objects have probability mass
under controlled prefixes.

Artifacts:

- [latent_probe_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/latent_probe_summary.md)
- [latent_probe_summary.json](artifacts/et_rmp_continuation_diagnostics_2026-05-01/latent_probe_summary.json)

Probe shape:

- Compare `close_now` against `continue_with_fn`.
- Score both base1332 and ET-RMP step300.
- Use prefixes including empty, oracle-before-FN, oracle-late-all-except-FN,
  oracle-matched-GT, and self-pred-all.
- Mask the FN bbox region versus another GT bbox region and measure likelihood
  drops for the FN continuation.

Compact read:

| scorer | representative mask effect | read |
|---|---|---|
| base1332 | FN-bbox masking usually lowers FN-continuation score; control masks are smaller or near zero. | Many FNs are visually conditioned, not pure hallucinated text. |
| ET-RMP step300 | FN-bbox masking is often stronger than base, with high positive-drop rates. | ET-RMP still carries visual-conditioned mass for many missed FNs. |
| both | Late-prefix close/continue margins become much smaller than empty-prefix margins. | Prefix state and boundary competition matter. |

Important limitation:

```text
This proves latent visual-conditioned continuation mass for many FN entries.
It does not prove that unconstrained autoregressive rollout will choose those
entries without changing other dynamics.
```

## Length / Stop-Pressure Probe

The boundary probe directly measured whether closing becomes cheaper as the
assistant prefix gets longer.

Definition:

```text
close_adv = log p(close first token) - log p(continue first token)
```

Positive `close_adv` means the model prefers close over the continuation first
token at that boundary.

Artifact:

- [length_bias_summary.md](artifacts/et_rmp_continuation_diagnostics_2026-05-01/length_bias_summary.md)

Linear trend excluding empty-prefix rows:

| scorer | n | Pearson(length, close_adv) | slope / 100 tokens | mean close_adv |
|---|---:|---:|---:|---:|
| base1332 | 375 | 0.385 | 0.391 | 1.391 |
| ET-RMP step300 | 375 | 0.379 | 0.221 | 1.009 |

Best single-step split:

| scorer | threshold token | left mean close_adv | right mean close_adv | jump |
|---|---:|---:|---:|---:|
| base1332 | 151 | 0.258 | 1.674 | 1.416 |
| ET-RMP step300 | 197 | 0.458 | 1.229 | 0.772 |

Object-count bins:

| scorer | low-count behavior | high-count behavior |
|---|---|---|
| base1332 | `1-3` objects: mean close_adv `0.494`, `frac_continue=0.594` | `13-15` and above: `frac_continue=0.000`; `20+` mean close_adv `2.529` |
| ET-RMP step300 | `1-3` objects: mean close_adv `0.688`, `frac_continue=0.219` | `4+` bins already have `frac_continue=0.000`; high-count close pressure remains positive |

This supports a real length/count-related close-pressure signal. It does not
tell us whether suppressing close at decode time is a good intervention.

## Hard Stop-Control Ablation

The next experiment controlled the stop signal during generation on core-6.

Variants:

- baseline;
- suppress first structural close after an object boundary;
- steer first array branch toward another object with bias `1.0`;
- suppress all terminators after an object boundary.

Strict eval result:

| model | variant | TP@0.50 | FP@0.50 | FN@0.50 | parsed preds | recovered baseline FN | issue |
|---|---|---:|---:|---:|---:|---:|---|
| base1332 | baseline | 55 | 19 | 32 | 77 | 0 | healthy |
| base1332 | stop-control variants | 0-9 | 0-11 | 78-87 | 0-20 | 0 | parse collapse or max tokens |
| ET-RMP step300 | baseline | 42 | 12 | 45 | 59 | 0 | healthy |
| ET-RMP step300 | stop-control variants | 0-21 | 0-7 | 66-87 | 0-31 | 0 | parse collapse or max tokens |

Diagnostic salvage result:

| model | variant | TP@0.50 | FP@0.50 | FN@0.50 | salvaged preds | recovered baseline FN | lost baseline TP |
|---|---|---:|---:|---:|---:|---:|---:|
| base1332 | suppress_first_close | 55 | 19 | 32 | 77 | 0 | 0 |
| base1332 | steer_array_b1 | 55 | 19 | 32 | 77 | 0 | 0 |
| base1332 | suppress_all_terminators | 55 | 22 | 32 | 80 | 0 | 0 |
| ET-RMP step300 | suppress_first_close | 42 | 12 | 45 | 59 | 0 | 0 |
| ET-RMP step300 | steer_array_b1 | 42 | 12 | 45 | 59 | 0 | 0 |
| ET-RMP step300 | suppress_all_terminators | 40 | 12 | 47 | 57 | 2 | 4 |

Only one cell recovered baseline FNs after salvage:

- `ET-RMP step300 + suppress_all_terminators`;
- recovered two `kite` GTs in `images/val2017/000000017959.jpg`;
- lost four baseline TPs;
- net TP/FN was worse.

Interpretation:

```text
Hard stop-control is too blunt. It mostly breaks terminal syntax or leaves the
same object set after repair. It is not a clean production fix for conservative
emission.
```

## Fair Evaluation Under Incomplete Annotation

For this line of work, official COCO-style metrics and manual review answer
different questions.

Protocol to keep comparisons fair:

1. Keep strict evaluator metrics as the primary reproducible benchmark.
2. Treat FN objects as confirmed missed objects because they are labeled GT.
3. Treat FP objects as unresolved until manually adjudicated.
4. For manually reviewed FP, assign one of:
   - plausible unlabeled object;
   - duplicate of a matched object;
   - localization miss;
   - wrong class;
   - background/no object.
5. Report both strict FP and human-adjudicated harmful FP.
6. Keep visual review on the fixed bank first, not all `val200`.
7. Never use salvage or manual relabeling to replace the official result; use it
   to explain failure modes and design the next experiment.

This is especially important for dense crowd scenes like `hurt_178`, where
many visually plausible person instances may not be exhaustively labeled.

## What Is Established

Established by current artifacts:

- ET-RMP trains full recursive suffixes and fixes the old malformed-output
  failure surface much better than one-step candidate MP.
- The old ET-RMP `v1` checkpoint still under-emits in high-count scenes.
- Repetition penalty in the `1.10-1.18` band changes emission/ranking behavior;
  it is not just a monotonic conservatism knob.
- The fixed sample bank contains both `rp=1.18` benefit cases and hurt cases.
- Many FN continuations have visual-conditioned likelihood: masking the FN
  region lowers their score more than control masks.
- Boundary close pressure increases with prefix length/object count.
- Decode-time hard stop suppression is not a clean solution; it damaged grammar
  or failed to recover useful FNs at scale.

## What Remains Unproven

Not established yet:

- The model reliably perceives every missed object.
- FNs are mostly emission failures rather than perception/localization failures.
- Higher `rp` is generally better.
- Support-mass-enhanced ET-RMP has the same behavior as old `v1`.
- Hard stop control is useful beyond diagnostic probing.

## Recommended Next Experiments

The next experiments should avoid hard decode-time stop suppression and instead
measure the model distribution under controlled prefixes.

1. Rerun the fixed bank on the support-mass-enhanced production checkpoint.
2. For each confirmed FN in the bank, score:
   - `close_now`;
   - `continue_with_fn`;
   - `continue_with_other_remaining_gt`;
   - `continue_with_plausible_unlabeled_object` after manual review.
3. Build count-prefix counterfactuals: same image and target FN, different
   emitted-count prefixes.
4. Record close rank, close probability, valid-continuation mass, and FN rank
   after every object boundary.
5. Use multi-sample union coverage only as a diagnostic, not as the official
   benchmark.
6. Keep strict parse/eval and diagnostic salvage separate in reports.

## One-Line Current Read

The old ET-RMP run restored SFT-like JSON closure but remained conservative in
dense/high-count scenes; the evidence points to a real length/count-related
boundary pressure plus latent visual-conditioned FN mass, while hard
stop-token suppression is an ineffective patch rather than a mechanism-level
solution.
