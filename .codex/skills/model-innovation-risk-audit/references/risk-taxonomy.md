# Risk Taxonomy

Select only failure classes that could change the current decision. This is a
menu of counterexamples, not fourteen required gates. Reuse closed evidence;
apply artifact, topology, scale, or activation checks only when the declared
execution or claim depends on that surface. Routine model implementation uses
`qwen3-vl-execution`; this reference supports a read-only fidelity audit.

## 1. Config Truthfulness Risk

Compare intended behavior with authored/resolved config, the executable owner,
and emitted evidence. A matching key name does not prove matching semantics.

Decision-changing examples:

- `normalization=token_mean` resolves to semantic bucket balancing;
- authored accumulation conflicts with the effective-batch derivation;
- a stale alias or shared ablation default changes the declared factor;
- materialized config omits the objective identity that actually trained.

Close the relevant mismatch through the current config-to-runtime path, not an
additional config representation.

## 2. Tokenizer And Template Stop Risk

Bind training, generation, parser, and diagnostic meanings for EOS, pad,
assistant closure, image markers, and expanded vocabulary. For example,
training may supervise `<|im_end|>` while generation also stops on
`<|endoftext|>`; that extra stop changes the observed completion population.

Use [chat templates and real encoded inputs](../../qwen3-vl-execution/references/execution-checks.md#chat-templates-and-real-encoded-inputs)
for the actual processor/token-history checks. Inspect stop configuration and
leaked terminators only where they distinguish the suspected mismatch.

## 3. Data And Geometry Risk

Verify the declared sample still denotes the same pixels, objects, and order:

- image roots resolve and recorded dimensions match the actual image;
- coordinates and bbox ordering match the declared representation;
- object order agrees with the prompt/target contract;
- object or token caps do not silently remove evidence-bearing targets;
- temporary subsets preserve image-root and sample identity.

A bounded raw-image/JSONL check should expose the claimed failure before a
larger run. Do not infer geometric correctness from parser success.

## 4. Sidecar Alignment Risk

Sidecars, sparse targets, weights, and object metadata must follow their tokens
through encoding, collation, padding, packing, and logits selection.

The late loss-boundary invariant is semantic: each supervised target retains
its literal token identity, sample/segment identity, weight, and declared
support; its mapped logit row predicts that target from its preceding causal
context without crossing a segment boundary.

Use the actual logical-to-physical and target-to-logit maps. Native execution
may use `labels=None`, compact logits, or packed tensors; do not require a
universal `labels[b, position]` or two-dimensional attention-mask layout. When
labels or masks are present, verify their meaning against the same target map.

The execution owner supplies [causal positions, padding and packing](../../qwen3-vl-execution/references/execution-checks.md#causal-positions-padding-and-packing).
A wrong causal row with otherwise valid token IDs is a useful counterexample.

## 5. Logits, Padding, And Packing Risk

Select offset-changing paths actually enabled: `logits_to_keep`, left/right
padding, packed attention, segment/position metadata, truncation, or cache reuse.
Absolute target indexing into compact logits can silently score another token;
a shape check alone will not detect that mistake.

Follow the [causal mapping checks](../../qwen3-vl-execution/references/execution-checks.md#causal-positions-padding-and-packing)
for unequal lengths and segment boundaries. Unsupported mappings should fail
at the owning boundary; a supported packed path needs evidence for its actual
mapping, not a second generic packed-runtime design.

## 6. Loss Composition Risk

Verify the differentiable scalar, including hard/soft CE, support/balance,
per-target and EOS weights, auxiliary terms, empty support, zero weights,
duplicate candidates, and teacher-token membership where relevant.

A concrete composition error is:

```text
intended: eos_weight * CE + type_gate
actual:   eos_weight * (CE + type_gate)
```

Use [objective and denominator checks](../../qwen3-vl-execution/references/execution-checks.md#objective-denominator-and-distributed-reduction)
for reduction semantics and [autograd checks](../../qwen3-vl-execution/references/execution-checks.md#autograd-tensor-meaning-and-capture)
for differentiability. A tiny formula counterexample should distinguish the
intended scalar and gradient from the suspected implementation.

## 7. Precision Risk

A finite loss can conceal inaccurate support mass, soft CE, KL/entropy,
conditional balance, or log-probability reductions. Solver feasibility can also
fail to survive materialization or replay at the accepted precision.

Use [precision and numerical acceptance](../../qwen3-vl-execution/references/execution-checks.md#precision-and-numerical-acceptance)
for arithmetic ownership, autocast, gradient preservation, and tolerance rules.
Test the claimed arithmetic boundary; do not treat an output upcast or a change
to all parameter dtypes as proof that the required precision was preserved.

## 8. Metrics Blind-Spot Risk

Ask whether the model could train the wrong objective while displayed metrics
remain normal. Inspect only the counters that could expose that mismatch:

- raw terms versus effective weighted contributions;
- absent or wrong denominators and zero-weight/empty-support counts;
- parse/drop counts and eligible/executed/analyzable population differences;
- EOS/continue, valid-mass, or type-violation diagnostics when claim-bearing.

For matching-based outcomes, bind geometry/category policy, threshold and tie
rules, duplicate handling, and unmatched denominator to the canonical evaluator.
Two predictions competing for one owner can distinguish per-prediction matching
from the declared owner-set outcome. Do not substitute a visual/proxy matcher
for benchmark semantics.

## 9. Train/Decode/Eval Parity Risk

Compare the surfaces required by the claim: prompt and literal history, image
preprocessing/resize, coordinate serialization, stop policy, grammar/logits
processors, parser, and confidence/scoring postprocess. Natural generation,
forced-prefix continuation, and teacher-forced replay support different claims.

Use the [encoded-input checks](../../qwen3-vl-execution/references/execution-checks.md#chat-templates-and-real-encoded-inputs)
and the actual generation/evaluator owners. Preserve intentional differences;
call them confounds or transfer assumptions rather than silently normalizing
them into parity. Label parser failures separately from model-quality outcomes
according to the frozen protocol.

## 10. Distributed And Effective Batch Risk

Resolve supported batch/packing shape from the current config and trainer;
there is no universal packed batch size. Bind effective-batch units separately
from the objective's loss denominator and reject conflicting authored versus
derived accumulation settings.

The [distributed reduction checks](../../qwen3-vl-execution/references/execution-checks.md#objective-denominator-and-distributed-reduction)
own weighting across ranks, accumulation, and unequal local counts. Include a
final partial accumulation window when it can change the claimed optimizer-step
semantics. Equal-sized batches or equal displayed losses do not prove gradient
normalization equivalence.

## 11. Artifact And Portability Risk

For a retained result or reproducibility claim, follow the current artifact
contract instead of prescribing a second filename list. Required evidence must
bind the effective config and source/code version, model/adapter/tokenizer,
input/template, runtime, and evaluation scope relevant to that result.

An adapter cited without its required expanded-vocabulary base bundle is not a
portable checkpoint. Resolve required payloads through the owning writer/loader
and verify the consumer can reconstruct the claimed execution. Historical
artifacts retain their version-bound contract.

## 12. Execution Topology Risk

A locally correct path may hang, double-reduce, skip mutation, or diverge by rank.
For topology-dependent behavior, inspect:

- wrapped versus unwrapped calls and forward counts per sample/pack/rank;
- collective order with intentionally unequal local work;
- autograd hooks, recomputation, and no-sync boundaries;
- optimizer mutation relative to finite/error consensus;
- rank-zero materialization before/after process-group work;
- parent/worker/child ownership on exit or interruption.

Close the implicated seam with real multi-process asymmetric work through the
production wrapper. A rank-local failure before mutation is discriminating
when consensus/recovery is the risk. Keep per-rank call/collective and exit
evidence needed for that conclusion; a single-rank or leaf test cannot close it.

## 13. Scale And Resource Risk

Before promoting a scale-dependent path, declare and measure relevant bounds:
forwards per sample/pack/segment/rank; full cache/materialization passes and I/O;
wall time, peak RSS, workers/queues/concurrency; and merged artifact bytes.
Record the representative case and extrapolation that would reject promotion.

Reject unbounded growth or a measured bound violation. Repeated full-data passes
to reprove unchanged content, full-object validation despite a sufficient
content-bound attestation, or escalating timeouts without complexity diagnosis
are reasons to test the cost assumption before scaling.

## 14. Artifact And Activation Lifecycle Risk

Use this branch only when publication, downstream reload, or activation is part
of the selected contract. Exercise required canonical serialization, finite
values, identity/hash and size bounds, atomic publication, and fresh consumption
on a non-empty artifact. In-memory success does not establish durable success.

For at-most-once activation, bind intent, target/run identity, and terminal
outcome. Source/config drift invalidates the relevant attestation. Follow the
owning protocol for partial or uncertain activation and bounded recovery; retain
attempt linkage where that protocol requires it. Never retry an uncertain
activation merely because no final artifact appeared.

Before expensive broad execution, close the lifecycle seam actually relied on
by that run. Do not add activation or append-only recovery machinery to an
ordinary probe that has no such contract.
