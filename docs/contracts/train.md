# Training and packing

Training is configuration-first:

```bash
python -m src.train --config <train-config.yaml>
```

The package contract is Qwen with Transformers and PEFT. General configurations
retain support for one through four local GPU processes. The explicit
`configs/smoke/eight_gpu_qwen3_vl_2b_coco.yaml` overlay adds an eight-process
smoke without requiring eight ranks for general training. Its effective batch
size remains 48, giving six accumulation micro-steps per rank per update.

## Pack cache

Prepare and validate the training cache through its public entrypoint:

```bash
python -m src.prepare_train_cache --config <train-config.yaml>
```

The cache identity covers every semantic determinant of the packed
micro-steps. A stale, corrupt, incomplete, or mismatched cache is rejected;
it is never silently reused.

Parallel preparation uses 16 materialization processes by default and keeps
at most twice that number of submitted unfinished tasks. Results are restored
to source order; raw examples and completed encoded results remain
materialized. This is a submission bound, not a streaming-data guarantee.
`packing.worker_count` is a separate planner identity field and does not
select the number of preprocessing processes.

For a read-only cache admission check, use `--require-all-hit` on the same
entrypoint. A missing or invalid cache fails before any rebuild. The optional
`coordexp_infras_PACK_CACHE_ROOT` environment variable selects a cache root.

## Exact resume

Exact resume is opt-in and stricter than loading an inference payload. It is
admitted only at an optimizer boundary, requires the same world size, and
fails closed when its runtime state or identities do not match. It is not a
promise of topology migration or an inference mechanism.

An uninterrupted parent must itself set `resume.mode: exact_same_world_size`
with `resume.checkpoint_dir: null` to publish exact training state. A fresh
continuation selects its nonterminal checkpoint and keeps the same total
schedule. The four-update smoke saves step 2 for this purpose.

Strict replay fixes the native DDP bucket layout with
`find_unused_parameters=True, static_graph=False`; fresh and uninterrupted
processes must use the same gradient reduction policy. This policy enters
checkpoint admission identity. Checkpoints produced before this policy was
bound remain inference payloads, but cannot supply compatible training state.
Existing step rows expose compact rank-zero native bucket observations.

Strict CUDA replay requires these variables before Python starts:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
```

Supported pre-change DoRA and special-token embedding components retain
loading/inference compatibility, including independently supplied components
without a root publication manifest. Old optimizer/RNG/scheduler/cursor state
migration is not promised. New checkpoints retain strict same-world-size exact
resume; available inference authentication evidence remains mandatory.

See [artifacts](artifacts.md) for the separate checkpoint payload contract and
[operations](operations.md) for the acceptance boundary.
