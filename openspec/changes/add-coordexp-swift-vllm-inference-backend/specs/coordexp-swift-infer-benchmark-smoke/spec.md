## ADDED Requirements

### Requirement: Real vLLM qualification smoke
Backend support MUST include a real single-GPU Qwen3-VL smoke with the qualified
vLLM version. The smoke MUST verify one image, no resize, exact prompt token
ids, special-token-preserving generated ids/text, `<|im_end|>` retention,
policy logprobs, parser output, selected-token score replay, artifacts, and the
unchanged evaluator. Mock-only tests MUST NOT satisfy this gate.

#### Scenario: Base-only tracer bullet
- **WHEN** the fixed real base-only smoke runs through vLLM
- **THEN** it completes inference and evaluation with all required receipts

### Requirement: Fresh-worker repeatability
The fixed base-only tracer fixture MUST run at least twice in separate fresh
workers. Prompt ids, generated ids, semantic stop reason, parser output, scored
rows, row ordering, and non-timing artifact content MUST match exactly. Policy
likelihood values MUST satisfy the dual-likelihood numeric tolerances rather
than requiring bitwise floating-point identity. The initially supported maximum
`max_num_seqs` MUST also have a real concurrent-request receipt before use in a
production config.

#### Scenario: Repeat run changes one generated token
- **WHEN** two fresh-worker runs differ in any generated token id
- **THEN** repeatability qualification fails even if evaluator metrics match

### Requirement: Materialized checkpoint qualification
The exact base-plus-DoRA-plus-selected-token-delta composition MUST pass dynamic
HF versus materialized-HF parity before vLLM evaluation. Tied weights, prompt
ids, greedy token ids, selected-token logits, and source fingerprints MUST be
verified.

#### Scenario: Delta omitted during materialization
- **WHEN** selected-token logits reveal a missing or duplicated delta
- **THEN** adapter-enabled vLLM qualification fails

### Requirement: Dual likelihood numeric gate
A fixed repeated-token fixture with repetition penalty 1.10 MUST compare HF and
vLLM policy and raw likelihood channels. Prompt and generated token ids MUST
match exactly. Across aligned tokens, median absolute likelihood difference
MUST be at most `0.002`, P99 at most `0.02`, maximum at most `0.05`, and
per-object absolute log-score difference at most `0.01`.

#### Scenario: Numerically close but token shifted
- **WHEN** likelihood values meet numeric tolerances but any token id or
  conditioning position differs
- **THEN** the gate fails

### Requirement: Real outer data-parallel smokes
The backend MUST pass two-GPU and eight-GPU production-like smokes using the
existing outer sharding controller. Smokes MUST verify one logical GPU per
engine, exact row coverage/order, identical execution-model fingerprints,
strict trace merge, terminal cleanup, and evaluator consumption.

The eight-GPU gate MUST have exactly eight visible CUDA tokens, exactly eight
active nonempty ranks, at least eight decode blocks, one validated engine
session receipt per rank, and a successful merged evaluator receipt. Merely
making eight GPUs visible while activating fewer ranks does not satisfy it.

#### Scenario: Worker failure
- **WHEN** any engine worker fails or leaves invalid shard evidence
- **THEN** the controller preserves diagnostics and does not publish canonical
  top-level scored artifacts

#### Scenario: Required failure-injection matrix
- **WHEN** Wave 5 acceptance is evaluated
- **THEN** engine-startup failure, CUDA OOM, worker timeout, forced process-tree
  termination, orphan process, malformed shard, backend-identity mismatch, and
  likelihood mismatch each have an executed terminal receipt proving cleanup
  and no canonical top-level publication

#### Scenario: Eight visible GPUs but seven active ranks
- **WHEN** an alleged eight-GPU smoke activates fewer than eight nonempty ranks
- **THEN** the eight-GPU acceptance gate fails

### Requirement: Matched val200 acceptance
The same checkpoint, dataset, prompt, generation policy, and evaluator MUST be
run through HF and vLLM on the fixed val200 scope. Input and GT identity MUST
match exactly. Absolute bbox mAP and mRecall differences MUST each be no greater
than `0.005`. Parse, drop, invalid, stop, truncation, throughput, and peak-memory
counters MUST be reported even when the metric gate passes.

#### Scenario: Metric drift
- **WHEN** either absolute mAP or mRecall difference exceeds `0.005`
- **THEN** vLLM remains experimental and MUST NOT replace HF benchmark evidence
