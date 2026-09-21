# Training-set completion probes

This package owns the research recipes, not a universal trainer. Fixed-teacher
fitting, Source256 A/B completion, normalized completion, output ranking and
readout/corner diagnostics retain their own cohorts, teacher/admission policy,
loss denominators, geometry, ordering, release gates and stop conditions.
The current scientific frontier is in [research/index.md](../../research/index.md).

## Reuse by concept, not by experiment ancestry

| Operation | Maintained owner | Caller responsibility |
| --- | --- | --- |
| Bound route materialization and compact causal replay | `replay.prepare_microbatches`, `replay.batched_aligned_logits` | Literal routes, native batch composition, masks and gradient policy |
| Original per-route active-token CE and geometry terms | `replay.route_terms` | Alternative CE denominators and final objective aggregation |
| Globally normalized gradient SUM and deterministic state consensus | `distributed` | Global denominator, rank partition and scientific failure interpretation |
| Source256 A/B and normalized paired execution | `source256_training.run_paired_training` | Explicit variant validator, route resolver, CE terms, producer identity and update enrichment |
| Source256 shared structural recipe validation | `source256_training.validate_training_recipe` | Explicit schema and producer identity; normalized projection retains its own checks |
| Singleton native tensor combination | `src.qwen.native.combine_singleton_native_inputs` | Ordered prompt IDs; no pixel slicing as though pixels were one row per image |
| Owned child spawn, wait and termination | `src.runtime.owned_process` | Command, working directory, environment/GPU assignment, deadline and recovery policy |
| Completion-family JSON encoding and file hashing | `artifacts` | Publication policy and the meaning of the artifact |

`coco227_training` and `dual_start_distributed` are still scientific recipe
owners. They are no longer the source of shared private replay or distributed
functions. Source256 ranking has a different objective and reference-cache
lifecycle, so it reuses the narrow operations rather than inheriting the paired
training loop. The fixed 11/4 and 22/8 partitions remain with their recipes.

Normalized completion now calls the Source256 loop explicitly. It does not
copy a function's globals, reconstruct a `FunctionType`, substitute `__file__`,
or intercept every publication through a module proxy. Only update evidence is
enriched; checkpoint, rank and terminal publication use their normal paths.
The release check remains before entry into model execution.

These are two actual consumers of one Source256 recipe, not a generic context,
plugin registry or extensible trainer framework. A new research problem should
not acquire these interfaces unless its contract really matches.

## Lifecycle and artifact invariants

An owned process must be a child started in its own process session. A failed
spawn closes its opened log. A timeout or interruption while waiting stops and
reaps the exact owned child. An already completed child remains complete when
observed after its deadline; this does not prove when it historically finished.
The primitive does not allocate GPUs, discover or kill arbitrary processes,
recover orphaned descendants after their leader has been reaped, or authorize a
retry. Those are separate lifecycle decisions.

Canonical completion JSON remains UTF-8, sorted compact keys, unescaped Unicode
and one terminal newline. Historical Source acquisition digests without the
newline remain explicitly different. `training.publish` is exclusive;
`source256_data.publish` accepts an existing file only when bytes are identical.
Neither was replaced with a configurable universal artifact store.

## Historical evidence and current execution

The pre-refactor recovery commit is
`b34e8a721129b6b7df3b4c0f244265f83de898f9`. Existing experiment outputs,
producer hashes, checkpoints, results and scientific states were not rewritten.
The sibling research worktree was not modified by this refactor.

An old receipt describes its original effective sources, not whatever happens
to exist at the same path today. Recover old producers from their recorded Git
commit or effective-source snapshot when reproducing or validating that result.
Do not replace recorded hashes with current hashes merely to make a historical
gate pass. The same caution applies after moving to a different checkout.

Newly authorized work must bind the current producer and shared dependencies.
Closed compute grants remain closed. A code refactor, passing CPU tests or an
old launch command is not permission to rerun training or regenerate outputs.
Do not retain obsolete forwarding aliases solely to keep old scripts executable
from the latest tree.

## Verification boundaries

Tests are colocated in `tests/`. Replay tests cover mixed sequence lengths,
causal alignment, gradients and literal masks. Distributed tests use actual
CPU/Gloo processes and preserve SUM without an extra world-size divisor.
`test_paired_execution.py` executes the complete shared loop on four CPU ranks
with a tiny substituted model and compares its two AdamW updates to serial
execution, including checkpoint and variant receipt checks. It is not a Qwen
loader, real-image, GPU-memory or NCCL parity claim.

`tests/runtime/test_owned_process.py` and this package's
`tests/test_process_lifecycle.py` exercise actual low-cost child processes,
late deadlines, interruption, log closure and exact process ownership.
`tests/test_artifact_primitives.py` checks byte identity and distinct collision
policies. `tests/qwen/test_singleton_native_batch.py` checks the shared native
batching seam, alongside the owner-successor replay tests.

Run these with CUDA hidden and bounded CPU threads. Tests explicitly requiring
retained production preparations, a bound tokenizer or model artifacts are a
separate integration tier; do not replace their inputs with fabricated receipts
or reinterpret a CPU-only pass as their acceptance.

## Shared identity and scoring

`artifacts.literal_binding` preserves authored paths, while `artifacts.binding` resolves them; their JSON/hash contracts are intentionally distinct. Native tensor/input identity is owned by `src.qwen.input_identity`, not `readout_norm_fresh`. `row_scoring.score` owns the shared frozen saved-row accounting, with eleven real/edge characterization cases. `row_branch` is a distinct maintained row-boundary runtime. `src.qwen.untied_embeddings` retains the exact untied payload implementation; it is not interchangeable with the general embedding owner.

New producer receipts bind current maintained sources and shared dependencies. Historical source recovery does not permit rewriting old hashes or running old grants. See the [storage policy](../../docs/OUTPUT_STORAGE_POLICY.md).
