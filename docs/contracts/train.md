# Training and packing

Training is configuration-first:

```bash
python -m src.train --config <train-config.yaml>
```

The package contract is Qwen with Transformers and PEFT. A launch may use one,
two, three, or four local GPU processes; neither a configuration nor this
contract may hardcode an eight-rank topology.

## Pack cache

Prepare and validate the training cache through its public entrypoint:

```bash
python -m src.prepare_train_cache --config <train-config.yaml>
```

The cache identity covers every semantic determinant of the packed
micro-steps. A stale, corrupt, incomplete, or mismatched cache is rejected;
it is never silently reused.

## Exact resume

Exact resume is opt-in and stricter than loading an inference payload. It is
admitted only at an optimizer boundary, requires the same world size, and
fails closed when its runtime state or identities do not match. It is not a
promise of topology migration or an inference mechanism.

See [artifacts](artifacts.md) for the separate checkpoint payload contract and
[operations](operations.md) for the acceptance boundary.
