# Recursive Detection Bucketing And Packing Optimization Design

Date: 2026-05-09

Status: design and planning only. Implementation is not authorized until the
user approves the audited plan.

Owner surface: latest compact recursive detection training stack under
`src/detection/*`, `src/data_collators/*`, `src/trainers/metrics/*`, and
`src/sft.py`.

Primary variant: `prefix_rollin_et_rmp_ce`.

## Purpose

Reduce wall-clock cost for compact-full recursive detection / prefix-rollin
ET-RMP-CE training without changing the Qwen3-VL architecture, adding a
detection head, changing ordinary autoregressive decode, or changing objective
semantics.

The current healthy run is slow for two reasons:

- full trainer eval is too frequent and expensive;
- train/eval batches are ordinary padded batches despite long variable-length
  multimodal detection sequences.

The safe optimization sequence is:

```text
config-only eval/runtime hygiene
  -> row-atomic length bucketing
  -> sidecar-safe static packing
  -> padding-free packed runtime only after static packing parity is proven
```

## Locked Decisions

| Area | Decision |
|---|---|
| Model | Do not edit Qwen3-VL/HF model files and do not add a detection head. |
| Decode | Preserve the main autoregressive generate/decode path. |
| Objective | Preserve `prefix_rollin_et_rmp_ce` target semantics, target weights, type-gate behavior, EOS trust weighting, and multi-positive trie distribution. |
| First runtime win | Treat eval cadence and eval batch size as config/runtime hygiene, not as a model change. |
| First code win | Implement row-atomic length bucketing before any packing. Bucketing may reorder samples but must not concatenate samples. |
| Packing safety | Keep latest recursive detection packing disabled until every offset-bearing sidecar is rewritten and tested. |
| Effective batch | Preserve artifact truth for requested `effective_batch_size`, derived accumulation, realized world size, and actual global effective batch. |
| Training quality | Efficiency improvements must be behavior-preserving or explicitly diagnostic. Production adoption requires matched-scope quality checks showing no training-signal, loss, or validation degradation beyond agreed stochastic tolerance. |
| Geometry | Preserve image path resolution, `do_resize=false`, bbox token alignment, and object order/coords. |
| Tokenizer/template | Preserve Qwen chat-template compatibility and `<|im_end|>` supervision contract. |
| Provenance | Every optimization mode must be visible in resolved config, effective runtime, and experiment artifacts. |

## Non-Goals

- Do not simply flip `training.packing=true`, `training.eval_packing=true`,
  `packing.static_packing=true`, or `packing.padding_free_packed=true`.
- Do not enable `use_logits_to_keep` for recursive detection. Current targets
  are sparse and non-contiguous; a global suffix crop is not a semantics-safe
  optimization.
- Do not introduce dynamic token-budget batches as the first patch. It changes
  per-step sample counts and gradient-accumulation semantics more than
  row-atomic bucketing.
- Do not cache encoded samples for latest recursive detection until the
  sidecar, RNG, epoch, tokenizer, and artifact fingerprint contract is designed
  separately.
- Do not make full validation disappear. Full validation should move to less
  frequent checkpoint gates or out-of-band runs, while in-loop health eval uses
  explicit limited scope.

## Current Runtime Diagnosis

The active A3/A4 run artifacts showed:

- `per_device_train_batch_size=8`;
- `effective_batch_size=128`;
- derived `gradient_accumulation_steps=4` on four DDP ranks;
- `per_device_eval_batch_size=1`;
- `eval_steps=100` in the launched artifact, even though the current worktree
  YAMLs were later changed to `eval_steps=600`;
- `training.packing=false`, `training.eval_packing=false`,
  `packing.static_packing=false`, and `packing.padding_free_packed=false`;
- full eval runtime around 55-56 minutes per eval pass.

The immediate relaunch profile should use the user-authored
`eval_steps=600` plus explicit `per_device_eval_batch_size=8`, with a dedicated
eval-bsz8 memory smoke first because recursive CE eval uses full logits and may
increase memory. A tiny train-only smoke is insufficient because it does not
exercise the local eval batch of 8.

## Phase 1: Eval And Runtime Hygiene

This phase is config-first and should be safe to apply before deeper training
infrastructure work.

Requirements:

- A3/A4 bsz8 configs explicitly set `per_device_eval_batch_size: 8`.
- A3/A4 bsz8 configs retain `eval_steps: 600`.
- A health-eval overlay may cap only validation rows with an explicit scope such
  as `health-val64`; it must not cap train rows for real training. Latest
  detection configs reject `custom`, so the health overlay uses
  `debug.enabled: true` with `debug.val_sample_limit` set and
  `debug.train_sample_limit` absent or null.
- Full validation remains available as a separate full-val command or checkpoint
  gate.
- Effective runtime must not report inert or contradictory packing state such
  as disabled packing with nested eval-packing truthy metadata.
- Effective runtime and experiment manifest summaries must record
  `eval_strategy`, `eval_steps`, `per_device_eval_batch_size`, and any active
  dataloader optimization mode.
- Any smoke or relaunch must pass a non-disruptive launch gate: unique tmux
  session names, unique master ports, explicit GPU allocation, artifact-root
  collision checks, and no broad process or artifact cleanup.

Acceptance checks:

- resolved config records eval batch size and eval cadence;
- effective runtime and experiment manifest summaries record eval cadence and
  eval batch size;
- eval provenance records any health-eval sample limit;
- health-eval provenance records uncapped train data and capped eval data;
- recursive CE metrics, type-gate metrics, EOS-trust metrics, and multi-positive
  trie metrics still appear;
- no train-side objective field changes.

## Phase 2: Row-Atomic Length Bucketing

Row-atomic bucketing groups examples with similar encoded length into the same
ordinary padded batch. It does not concatenate sequences and therefore does not
require sidecar offset rewriting.

Design contract:

- `DetectionTrainingDataset` exposes deterministic encoded sequence length per
  base row for the active template/objective/config.
- The length source must match the exact model input sequence length that will
  be padded in training.
- Prefix-rollin length must be proven invariant across sampled `K` values for a
  fixed normalized object order. `K` may change labels and sidecars, but should
  not change the full compact input sequence length.
- Random object order may change token order but not total encoded length for a
  fixed object set; this must be tested.
- K/order invariance must be proven through template-equivalent tokenization,
  not only through the simple character-level fake tokenizer used by smoke
  tests.
- Bucketing must not mutate `TokenTarget.position`, `LossAtom.token_positions`,
  type-gate targets, EOS trust weights, trie positives, labels, attention masks,
  image fields, or metadata.
- DDP sampling must preserve normal epoch shuffling and deterministic worker
  behavior.
- DDP bucketing must be tested without training for `world_size > 1`, including
  rank sample counts, repeat/drop behavior, epoch reshuffle, and non-empty rank
  shards.
- Length precompute is run-local memory only for v1. It must not be persisted
  or reused across process/config boundaries unless a stricter fingerprinted
  cache design is added later.

Preferred implementation shape:

- Add a small object-oriented length provider for latest detection datasets
  rather than adding ad hoc sampler code throughout `src/sft.py`.
- Use a dedicated latest-detection length-grouped sampler path unless the
  implementation proves that stock HuggingFace/Transformers receives an
  explicit `list[int]` from the length provider without calling
  `DetectionTrainingDataset.__getitem__` during sampler construction.
- Treat `training.group_by_length` and `training.length_column_name` as existing
  TrainArguments pass-through keys, not new CoordExp-internal schema knobs.

Acceptance checks:

- sidecar alignment tests pass before and after bucketed collation;
- two samples with very different lengths are batched with less padding under
  bucketing than under plain random batching;
- matched-seed non-bucketed and bucketed tiny runs produce comparable recursive
  CE, type-gate, EOS-trust, trie, and eval-health metrics; bucketing is not
  considered successful on throughput alone;
- sampler construction does not call dataset `__getitem__` or encode all rows
  through the normal sample path;
- same seed yields stable bucket order;
- different epochs reshuffle within the intended deterministic contract;
- effective batch artifacts remain truthful;
- effective runtime records the length source, `run_local_only` cache policy,
  sampler class, seed, epoch policy, rank/world-size behavior, and `drop_last`
  behavior.

## Phase 3: Sidecar-Safe Static Packing

Static packing concatenates multiple original samples into one longer packed
row. This is high-leverage but correctness-sensitive because latest recursive
detection sidecars currently use absolute row-local positions.

Packing is allowed only after the implementation introduces an explicit
sidecar-packing contract. The first production-safe static packing contract must
preserve sample isolation for forward semantics. Plain concatenation with
cross-sample causal attention may be explored only as a diagnostic throughput
prototype and must not be represented as objective-equivalent training.

Required sidecar rewrite:

- `TokenTarget.position`;
- `LossAtom.token_positions`;
- debug span token positions used by diagnostics;
- target grouping needed to preserve per-original-image normalization;
- detection metadata needed to report original sample count and pack membership.
- packed-row grouping: `recursive_detection_targets` length equals packed row
  count, while original sample groups are nested metadata inside each packed row.

Required semantic preservation:

- packed and unpacked examples produce equal recursive CE loss on synthetic
  logits that assign the same probabilities to the same original targets;
- unequal original samples are normalized independently first, then averaged by
  original-sample count;
- EOS trust weight is preserved per original sample;
- type-gate targets are preserved per token role;
- multi-positive trie target distribution is preserved;
- padding positions never contribute to loss;
- malformed object-entry spans fail fast.
- segment-aware attention and position semantics prevent later packed samples
  from attending to earlier packed samples, unless the run is explicitly marked
  as a non-equivalent diagnostic prototype.

Preferred implementation shape:

- Add a typed adapter such as `RecursiveDetectionSidecarPackingAdapter`.
- Add an explicit packed-sidecar schema version, for example
  `recursive_detection_sidecar_packing_v1`.
- Keep the current fail-fast guard for packed recursive sidecars unless the
  adapter has produced and validated the expected contract marker.
- Extend recursive CE loss to normalize by original samples, not merely packed
  rows.
- Record both packed-row counts and original-sample counts in metrics and
  artifacts.
- Keep existing `effective_batch_size` and `actual_global_effective_batch_size`
  in packed-row units when static packing is enabled, and add explicit original
  sample count estimates/observations so pack units and image units cannot be
  confused.

## Phase 4: Padding-Free Packed Runtime

Padding-free runtime is not part of the first implementation. It is a future
phase after sidecar-safe static packing proves target parity.

Additional requirements for that phase:

- correct Qwen varlen metadata such as `cu_seq_lens_q`, `cu_seq_lens_k`,
  `position_ids`, and `text_position_ids`;
- no global `logits_to_keep` suffix crop;
- indexed target-logit gathering or full logits with proven parity;
- trainer metric plumbing for the packed runtime mode;
- artifact coding for runtime mode values.

## Audit Gates

Before implementation approval, independent audit agents must review:

1. dataset/length-bucketing correctness and deterministic length assumptions;
2. collator/sidecar/loss offset safety for static packing;
3. config/runtime/artifact provenance and effective-batch semantics;
4. eval strategy and smoke-test gates.

All audit P0/P1 issues must be resolved in this spec and the companion plan
before implementation begins.

## 2026-05-09 Audit Resolution Addendum

The first parallel audit round found no P0 issues. The P1/P2 findings are
resolved in this spec and the companion plan as follows:

- Eval-bsz8 now has a mandatory memory smoke and non-disruptive launch gate.
- Disabled-packing eval metadata is a mandatory runtime-provenance fix.
- Health eval is explicitly scoped as not full eval, with latest-detection
  `debug.val_sample_limit` only and uncapped real training rows.
- Eval cadence, eval batch size, dataloader mode, sampler provenance, and length
  source must appear in effective runtime and experiment manifest summaries.
- Row-atomic bucketing now requires a dedicated latest-detection sampler unless
  an implementation proves explicit length-list use without dataset re-encoding.
- DDP sampler behavior and run-local length-cache provenance are first-class
  acceptance checks.
- Static packing now uses a two-level packed schema: one sidecar per packed row,
  plus original sample groups for normalization and diagnostics.
- Static packing production use requires forward isolation; plain cross-context
  concatenation is diagnostic-only.
- Effective batch units remain packed-row units for static packing, with
  explicit original-sample count fields added separately.
- All efficiency phases now include a quality-preservation gate: speedups must
  not degrade matched-scope training or validation metrics before production
  relaunch.
