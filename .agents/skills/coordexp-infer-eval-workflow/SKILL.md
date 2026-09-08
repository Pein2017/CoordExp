---
name: coordexp-infer-eval-workflow
description: Launch, repair, validate, or summarize current CoordExp-Swift inference, scoring, and detection evaluation through HF or vLLM.
---

# CoordExp-Swift Inference And Evaluation

Use the current YAML-owned pipeline. Discover the live launcher, config family,
backend contract, artifact writer, and evaluator from current authority and CLI
help before launch. Load a historical inference or evaluation branch only when
the user names it explicitly.

## Run

1. **Resolve the contract.**
   - Load the authored and effective config; verify input JSONL, image roots,
     checkpoint/adapter, prompt/template, coordinate surface, decode settings,
     output root, and distributed launch shape.
   - Resolve every decode and batching value from the current authored config
     family and its effective-config receipt. This skill owns no numeric
     defaults: do not copy a value from an older run, this text, or launcher
     behavior into the command line. Stop when authored and effective values
     disagree without an owning override.
   - Complete when the intended run and benchmark scope are explicit.

2. **Execute the owned path.**
   - Use the discovered current launcher and its YAML interface; do not select a
     runner from memory.
   - Keep HF dynamic composition first-class. Treat materialized HF as a
     composition oracle and label vLLM precision or parity claims by their actual
     qualification.
   - Use the authored sampling and completion-count contract. Keep completion
     count, per-device batch size, and distributed worker count distinct; do not
     infer one from another.
   - Complete when every worker exits, rank coverage and merge order are
     complete, and resources return.

3. **Validate artifacts before metrics.**
   - Require terminal success plus the current run manifest and complete raw,
     scored, provenance, trace, diagnostic, image-plan, config, and summary
     surfaces owned by the pipeline.
   - Distinguish a plumbing smoke from a metric-bearing smoke. Terminal success
     with truncation or parser failures proves the runtime path, not inference
     quality.
   - Bind every row to image, dimensions, GT, parser status, and coordinate
     meaning. Predictions are pixel `xyxy`; GT coordinate bins require the
     canonical conversion path.
   - Failed or diagnostic-only runs may retain failure artifacts but cannot
     publish benchmark-looking outputs.
   - Complete when artifacts are metric-bearing or explicitly labeled
     diagnostic.

4. **Evaluate the same identity.**
   - Run the current detection evaluator against the completed artifact root.
   - Verify benchmark eligibility, metric-bearing status, row counts, parser and
     drop counters, and matched evaluation scope.
   - Complete when metrics, the evaluation receipt, and evaluator sidecars point
     back to the authoritative run manifest, config, and predictions. Use the
     visualization skill when per-row visual evidence is required.

## Repair

When a stage fails, preserve the exact failure identity, classify whether the
owner is config, model composition, worker/runtime, parsing/scoring, merge, or
evaluation, and rerun only the invalid stage when the contract permits it. Do
not rerun inference merely because a downstream view changed.

Use `model-diagnosis` when valid artifacts show abnormal model behavior and
`model-innovation-risk-audit` when silent contract drift threatens the claim.

## Report

State config and checkpoint identity, backend and precision, data/decode scope,
artifact root and terminal status, evaluation scope, counters, metric paths,
verification, and limitations.
