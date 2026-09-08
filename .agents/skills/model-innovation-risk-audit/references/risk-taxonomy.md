# Risk Taxonomy

Use this reference when the audit needs a systematic checklist of silent model-innovation failure modes.

## 1. Config Truthfulness Risk

Look for cases where the config or resolved config says one thing but runtime hard-codes another.

Examples:

- config says `normalization=token_mean`, runtime uses semantic bucket balancing,
- authored `gradient_accumulation_steps` conflicts with derived effective batch,
- stale `custom.*` aliases are still accepted,
- production and ablation surfaces share unsafe defaults,
- materialized config omits the objective identity that actually trained.

Diagnostics:

- inspect authored YAML,
- inspect materialized config,
- inspect runtime object,
- compare against sidecars or emitted artifacts.

## 2. Tokenizer And Template Stop Risk

Check whether training, inference, parser, and diagnostics agree on:

- EOS token,
- pad token,
- text-level terminators,
- assistant stop marker,
- chat-template closure,
- image marker rendering,
- added tokens and expanded vocab.

Common hidden failure:

- training supervises `<|im_end|>`,
- generation also treats `<|endoftext|>` as EOS,
- decode can stop early with a token never used as semantic training EOS.

Diagnostics:

- real tokenizer id probe,
- real `apply_chat_template` text probe,
- generation config probe,
- compact/generated output scan for leaked terminators.

## 3. Data And Geometry Risk

Check that labels match pixels and geometry.

Audit:

- image paths resolve strictly,
- images exist,
- row dimensions match actual PIL dimensions,
- coords are valid and canonical,
- bbox order is valid for the declared format,
- source object order matches the prompt/object-order contract,
- object count caps fail fast rather than silently truncate,
- temp/subset JSONLs do not infer wrong image roots.

Diagnostics:

- JSONL schema scan,
- bbox decode and validity scan,
- image existence/dimension preflight,
- max object and max token-length risk scan.

## 4. Sidecar Alignment Risk

For objectives using sidecars, sparse targets, token weights, masks, or object metadata, verify alignment after every transformation.

Audit:

- prepared token positions,
- encoded positions,
- shifted positions,
- collated positions,
- padding offsets,
- packed offsets,
- labels,
- attention masks,
- causal logits row.

Late invariant:

```text
For each target:
1 <= target.position < seq_len
input_ids[b, target.position] == target.teacher_token_id
labels[b, target.position] == target.teacher_token_id
attention_mask[b, target.position] == 1
attention_mask[b, target.position - 1] == 1
loss consumes logits[b, target.position - 1]
```

This check belongs as late as possible before loss computation.

## 5. Logits, Padding, And Packing Risk

Audit all paths that can change time dimensions or target offsets:

- `logits_to_keep`,
- left padding,
- right padding,
- static packing,
- padding-free packing,
- packed attention kwargs,
- `position_ids`,
- `cu_seq_lens`,
- template truncation,
- cache reuse.

Danger sign:

- loss indexes absolute target positions, but runtime enables any feature that changes token offsets or slices logits.

Diagnostics:

- fail fast on unsupported offset-changing modes,
- full logits shape check,
- synthetic tests for target at boundary positions,
- separate packed-runtime design if packing is allowed.

## 6. Loss Composition Risk

Inspect whether the actual differentiable scalar matches the intended formula.

Audit:

- hard CE vs soft CE,
- support and balance decomposition,
- per-position weights,
- EOS trust weights,
- type-gate or schema auxiliary terms,
- semantic normalizer,
- zero-weight target behavior,
- duplicate candidate multiplicity,
- teacher token in positive set.

Common hidden failure:

```text
intended: eos_weight * CE + type_gate
actual:   eos_weight * (CE + type_gate)
```

Diagnostics:

- deterministic tiny-logit tests,
- formula tests with expected scalar,
- metric tests that distinguish raw and effective weighted terms.

## 7. Precision Risk

For probability math, check dtype and autocast behavior.

Audit:

- `log_softmax`,
- `logsumexp`,
- sparse soft CE,
- support mass,
- balance conditional distribution,
- KL/entropy-like terms,
- valid mass metrics.

Rule:

```text
Use fp32 for probability/log-probability math even when model forward uses bf16.
```

Diagnostics:

- bf16 logits test,
- autocast test,
- finite gradient test,
- metrics detach/no-grad test.

## 8. Metrics Blind-Spot Risk

Metrics can be correct but misleading if they log raw terms instead of effective contributions.

Audit:

- raw loss vs weighted loss,
- per-rank mean vs all-reduced numerator/denominator,
- absent denominator keys,
- zero-weight target counts,
- parse/drop counters,
- EOS/continue margins,
- valid-mass metrics,
- type-violation metrics.

Ask:

```text
Could the model be training the wrong thing while all logged metrics look normal?
```

If yes, add a diagnostic or explicit artifact.

## 9. Train/Decode/Eval Parity Risk

Compare training and inference:

- prompt construction,
- chat template,
- image preprocessing,
- `do_resize`,
- EOS/pad ids,
- grammar/logits processors,
- generation config,
- parser mode,
- coordinate surface,
- bbox format,
- confidence/scoring postprocess.

Diagnostics:

- exact train prompt vs infer prompt comparison,
- exact processor kwargs comparison,
- generation config dump,
- parse/drop summary by failure mode.

## 10. Distributed And Effective Batch Risk

Audit optimizer-step semantics.

Check:

- `effective_batch_size` source of truth,
- world size,
- per-device train batch,
- gradient accumulation,
- packing units,
- padding units,
- final partial accumulation windows,
- DDP metric reducers.

Preferred contract:

- Under packing or padding-free packing, `per_device_train_batch_size=1`; effective batch is in packs/global sequences per optimizer step.
- Under non-packing padded training, `per_device_train_batch_size` may be greater than 1; effective batch remains the source of truth and gradient accumulation is derived from world size.
- Do not let authored `gradient_accumulation_steps` silently conflict with derived effective batch.

## 11. Artifact And Portability Risk

Audit whether a future agent can reproduce the run.

Check artifacts include:

- resolved config,
- source config chain,
- git commit,
- command,
- checkpoint path,
- adapter path,
- base model path,
- tokenizer path,
- base model fingerprints,
- generation config,
- processor config,
- dataset JSONL,
- image root,
- prompt/template identity,
- metric scope,
- eval parser and coordinate surface.

Danger sign:

- adapter checkpoint is cited without its expanded-vocab base model bundle.

## 12. Execution Topology Risk

Audit behavior that can be locally correct while the real process graph hangs,
double-reduces, skips mutation, or diverges by rank.

Check:

- wrapped versus unwrapped model calls;
- model forward counts per sample, pack, segment, and rank;
- collective sequence when ranks have intentionally different local work;
- autograd-hook order, activation recomputation, and no-sync boundaries;
- optimizer mutation relative to finite/error consensus;
- pre- and post-process-group work, including rank-zero-only materialization;
- parent, worker, and child-process ownership on exit or interruption.

Diagnostics:

- a real two-process asymmetric-work smoke through the production wrapper;
- per-rank forward and collective receipts;
- an injected rank-local failure before optimizer mutation;
- process-tree and exit-status verification.

A single-rank or leaf-helper test cannot close this risk.

## 13. Scale And Resource Risk

Audit paths whose correctness survives small tests but whose complexity makes
production execution impractical or changes the effective behavior.

Declare and measure:

- full model forwards per sample, pack, segment, and rank;
- complete cache or materialization passes and representative bytes;
- I/O amplification and repeated deserialization;
- wall time, peak RSS, workers, queues, and concurrency caps;
- artifact bytes before and after rank aggregation;
- representative-scale extrapolation and the bound that rejects promotion.

Danger signs:

- a full-data pass is repeated only to reprove unchanged content;
- validation materializes full objects when a content-bound attestation exists;
- a branch grows with segments, ranks, or samples without a declared ceiling;
- timeout increases substitute for a complexity diagnosis.

## 14. Artifact And Activation Lifecycle Risk

Treat artifact production and activation as runtime protocols, not logging or
operator ceremony.

Exercise:

- non-empty canonical serialization and finite-value validation;
- hashing, byte-size bounds, rank transport, and atomic publication;
- fresh reload through the production-owned downstream consumer;
- target-bound attestations and invalidation after source or config drift;
- intent publication, process/run identity binding, and terminal receipts;
- at-most-once claim consumption, partial activation, and uncertain outcome;
- append-only parent-linked recovery with a fresh attempt identity and ceiling.

Run the complete canonicalize-to-consume path before expensive broad execution.
Never retry an uncertain activation merely because no final artifact appeared.
