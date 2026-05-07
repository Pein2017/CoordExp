# Subagent Prompts

Use these prompts when the audit has independent surfaces that can run in parallel. Keep each subagent read-only unless the user explicitly requested implementation.

## Objective/Loss Auditor

Audit the innovation objective and loss implementation for silent training-signal mismatches.

Focus on:

- target distribution vs intended formula,
- teacher token in positive set,
- support/balance decomposition,
- EOS weighting,
- auxiliary/type-gate composition,
- semantic normalization,
- zero-weight target behavior,
- duplicate candidate multiplicity,
- fp32 probability math,
- metric raw-vs-effective accounting.

Return P0/P1/P2/P3 findings with evidence, impact, patch direction, and tests. Also list confirmed OK checks and commands run.

## Dataset/Collator Auditor

Audit data, template, encoded labels, collator sidecars, padding, and target positions.

Focus on:

- JSONL schema,
- image-root resolution,
- image existence/dimensions,
- bbox validity,
- object ordering,
- max object/max length,
- prepared vs encoded target positions,
- collated labels/input_ids/attention_mask,
- causal shift `logits[position - 1]`,
- packing/padding incompatibilities.

Return P0/P1/P2/P3 findings with evidence, impact, patch direction, and tests. Also list confirmed OK checks and commands run.

## Tokenizer/Template/Decode Auditor

Audit tokenizer/template/inference/eval parity.

Focus on:

- special token ids,
- chat-template stop marker,
- EOS vs pad vs text-level terminators,
- generation_config.eos_token_id,
- processor kwargs such as `do_resize`,
- train prompt vs infer prompt,
- compact grammar or logits processors,
- parser/drop counters,
- generated special-token leakage.

Return P0/P1/P2/P3 findings with evidence, impact, patch direction, and tests. Also list confirmed OK checks and commands run.

## Config/Runtime/Artifact Auditor

Audit schema, materialized config, runtime wiring, distributed semantics, and artifacts.

Focus on:

- hidden overrides,
- stale aliases,
- production vs ablation gates,
- `effective_batch_size` source of truth,
- world-size dependent gradient accumulation,
- packing units,
- resolved config completeness,
- manifest/provenance,
- adapter/base model portability.

Return P0/P1/P2/P3 findings with evidence, impact, patch direction, and tests. Also list confirmed OK checks and commands run.

## Test Designer

Design unit tests, diagnostics, and smoke runs for the suspected silent-risk surfaces.

Focus on:

- tests that fail before the fix,
- deterministic tiny-logit loss formula tests,
- synthetic collated-batch alignment tests,
- real tokenizer/config probes,
- data preflight probes,
- production-like smoke commands with exact scope labels.

Return a prioritized test plan with filenames, test names, expected failure before patch, and commands to run.
