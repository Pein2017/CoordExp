# A2 EOS-Loosen Clean Ablation Design

Date: 2026-05-14

Status: Running

Owner surface: latest compact-full recursive detection CE training.

## Purpose

Run a clean A2+EOS-loosen ablation (`A2E`) that isolates missing-label-aware
EOS trust weighting on top of the stable A2 compact-full support2 baseline.

The research question is:

```text
Does reducing the final <|im_end|> target weight according to the empirical
missing-label prior improve valid-object emission / recall on A2 while
preserving A2's duplicate and parse stability?
```

## Corrected Comparison Framing

A2 remains the trusted compact-full anchor. A3 and A4 are prefix-rollin
mechanism probes, not fair replacements for A2.

The previous A2-vs-A3/A4 comparison is not a clean causal competition because
prefix-rollin changes several factors at once:

- state distribution: A3/A4 train from sampled GT prefixes;
- supervised-token density: with `K ~ Uniform[0, N]`, A3/A4 supervise about
  `E[N-K] = N/2` object entries per image exposure while still paying attention
  compute for prefix plus suffix;
- objective surface: A3/A4 use objectized target/boundary/type-gate sections
  instead of A2's flat support2/balance1 ET-RMP-CE weights;
- EOS behavior: A4 is A3+EOS loosen, not A2+EOS loosen;
- data ambiguity: COCO incomplete-label ambiguity affects stop/continue
  interpretation.

Therefore A4-vs-A3 can only probe EOS loosening inside the prefix-rollin
objective family. The clean EOS test is A2E: A2+EOS loosen with no prefix-rollin
and no reduced supervised-token density.

## Contract

A2E must preserve the A2 training contract except for `objective.eos`:

- same `compact_full` detection template;
- same `random_permutation_et_rmp_ce` variant;
- same `trie_support_weight=2.0` and `trie_balance_weight=1.0`;
- same `state_weighting=uniform_permutation`;
- same `normalization=semantic_image_bucket_balanced`;
- same coord-token rows and compact structure rows;
- same train/val JSONL and image root;
- same optimizer, schedule, batch, and epoch settings;
- no prefix-rollin, no prefix label masking, no suffix-only supervision;
- EOS trust applies only to the assistant `<|im_end|>` stop-token targets.

The A2E config is an ablation surface, not a production-calibrated EOS policy:

```text
experiment.surface = ablation
experiment.ablation_id = A2E-support2-eos-loosen
objective.eos.eos_trust_weight.source = empirical_unlabeled_poisson_v0
```

## Implementation Surfaces

- Schema: `src/config/schema.py`
  - allow `objective.eos` for `random_permutation_et_rmp_ce`;
  - keep `rollin`, `target`, `boundary`, and `type_gate` prefix-only;
  - require `experiment.surface` whenever `objective.eos` is present.
- Runtime: `src/detection/runtime.py`
  - pass `objective.eos.eos_trust_weight` into the latest detection dataset
    whenever `objective.eos` is configured.
- Dataset: `src/detection/dataset.py`
  - compute the per-image EOS trust weight outside the prefix-only branch;
  - pass it to the ordinary A2 target builder;
  - record `detection_metadata.eos_trust_weight` without adding roll-in fields.
- Objective: `src/detection/objective.py`
  - apply the trust weight only to token targets inside
    `tokenized.assistant_stop_token_span`;
  - preserve all non-EOS targets at loss weight `1.0`.
- Configs:
  - `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml`;
  - `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml`.

## Evaluation Gate

After training, evaluate the same A2 val200 surface first:

- greedy `max_new_tokens=1024`, `repetition_penalty=1.10`;
- greedy `max_new_tokens=3084`, `repetition_penalty=1.10` if cheap enough;
- raw and guarded metrics;
- duplicate guard report;
- parse/drop counters;
- prediction count;
- manual audit packet if valid emission or duplicate behavior changes.

Interpretation:

- If A2E increases valid-object emission while preserving A2 stability, the
  useful part of A4 may be EOS/missing-label calibration rather than
  prefix-rollin.
- If A2E increases duplicate/collapse, EOS loosening needs objectness or
  duplicate gating.
- If A2E does not help, A4 behavior is not explained by EOS loosening alone.

Future fair prefix-rollin ablations are separate: supervised-token-matched A3,
an A2/A3 mixture, or another design that controls active supervision density.
