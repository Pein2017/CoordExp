# Inference, scoring, evaluation, and visualization

Run inference through the package entrypoint:

```bash
python -m src.infer --config <infer-config.yaml>
```

HF and vLLM are first-class backends. A vLLM configuration must pass its
qualification path before it can be used as a production backend; a successful
import or a small probe is not a qualification witness. When vLLM needs a
materialized composition, set `COORDEXP_EXECUTION_MODEL_CACHE_ROOT` to an
absolute, durable external path. It is required unless the embedding caller
passes an explicit absolute `cache_root=`; a worktree-relative cache is not
accepted.

Dynamic HF retains its adapter-plus-embedding-delta execution semantics.
Composed BF16 inference executes an authenticated derived model. Its fixed
composition fixture must retain exact prompt IDs, generated sequence length,
non-coordinate tokens (including EOS and structural tokens), tied rows and
merged target weights. Coordinate tokens may differ by at most one grid unit
under the validated canonical coordinate mapping. Larger coordinate errors,
lexical/structural changes and invalid policy evidence reject qualification.

Receipts retain exact-greedy and logit-comparison diagnostics even when they
fail the former exact-parity thresholds; the bounded coordinate contract is
explicitly identified and validated. vLLM probabilities belong to the merged
execution model and are not claimed to equal dynamic-HF probabilities.

Run normal vLLM inference with that root explicitly supplied:

```bash
COORDEXP_EXECUTION_MODEL_CACHE_ROOT=/durable/external/coordexp-execution-model-cache \
CUDA_VISIBLE_DEVICES=<one-physical-index-or-GPU-UUID> python -m src.infer \
  --config configs/infer/vllm.yaml
```

vLLM qualification is BF16-only and reuses the current config, execution-model
materialization, inference shard, backend session, forced replay, and
composition paths. Produce into an absent external root, then admit the whole
validated set:

```bash
export COORDEXP_EXECUTION_MODEL_CACHE_ROOT=/durable/external/coordexp-execution-model-cache

CUDA_VISIBLE_DEVICES=<one-physical-index-or-GPU-UUID> python -m src.qualify_vllm run \
  --config configs/infer/vllm.yaml \
  --output-root /absent/external/vllm-qualification

python -m src.qualify_vllm admit \
  --config configs/infer/vllm.yaml \
  --receipts-root /absent/external/vllm-qualification
```

Each qualification child has a 1800-second execution deadline by default.
For a slower host, pass `--child-timeout-seconds 3600` to `run`; the value must
be finite and positive. This is a per-child operational ceiling, not a model
performance guarantee or a deadline for the entire four-mode run. On timeout,
the supervisor terminates its owned process group with bounded cleanup and
reports failure. GPU census commands also have bounded waits; missing evidence
cannot establish successful cleanup.

Qualification source identity covers project-local semantic dependencies,
including template rendering. A change to those source bytes invalidates old
receipts. Keep historical receipts as evidence and produce a fresh qualification
set for the current source before admission; do not reseal old receipts to reuse
them. The expanded coverage in the source-attestation repair intentionally
requires requalification. Config/model path binding and numerical acceptance
rules remain in force.

`CUDA_VISIBLE_DEVICES` must name exactly one physical GPU by index or `GPU-...`
UUID. Qualification measures only that GPU before execution and after a bounded
settle window. Missing measurements fail closed; memory return is accepted only
when the measured after value is at most 64 MiB above the measured baseline,
in addition to the owned-child and process-group cleanup checks.

Separate children execute runtime `max_num_seqs=1`, concurrency
`max_num_seqs=4`, forced replay at `max_num_seqs=1`, and composition fidelity.
Runtime/concurrency checks establish operational completion and finite policy
logprobs, not HF-versus-vLLM token/logit parity. Forced replay remains exact
against vLLM's own generated request, prompt, continuation and stop evidence;
the one-grid composition allowance never applies to replay alignment.
Receipts contain hashes, counters, numeric summaries, and cleanup/resource
evidence; detailed artifacts stay under the external root and are hash-checked
at admission. Admission installs exactly these files into
`src/inference/qualification_receipts`:

- `vllm-bf16-composition.json`
- `vllm-bf16-runtime-seq1.json`
- `vllm-bf16-concurrency-seq4.json`
- `vllm-bf16-forced-replay-seq1.json`

The destination must contain none of those files or a byte-identical complete
set. Partial or different state is rejected, and production remains fail-closed
until complete admission succeeds.

Evaluate the scored inference artifact directory directly:

```bash
python -m scripts.evaluate_detection \
  --artifact-dir <inference-artifact-dir> \
  --out-dir <eval-dir>
```

The direct evaluator consumes the scored artifact family and emits
`metrics.json` plus COCO conversion outputs in the requested output directory.

Render inspection images from run artifacts:

```bash
python -m scripts.visualize_detection gt-vs-pred \
  --run-dir <run-dir> \
  --out-dir <image-dir>
```

For a side-by-side comparison, use the `compare` subcommand with explicit
left and right run directories. Evaluation and visualization are downstream
consumers: neither changes a checkpoint or proves training resume.
