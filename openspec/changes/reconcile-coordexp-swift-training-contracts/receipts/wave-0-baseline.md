# Wave 0 baseline — current-contract reconciliation

- UTC recorded: `2026-08-12T15:23:03Z`; CWD:
  `/data/CoordExp/.worktrees/CoordExp-swift`; branch: `coordexp-swift`.
- Pinned commit: `71dab9772983a4680dfcea742aee949c79960560`.
- `git status --short` and `git diff --check` each produced no output.
- Runtime: Python `3.12.11`; Torch `2.9.1+cu128`; Accelerate `1.10.1` from
  Conda environment `ms`.
- Serena Light activation with `python_environment=ms` returned `UNCERTAIN`.
  The task used `rg`, source inspection, OpenSpec validation, and pytest; no
  semantic conclusion is based on that failed activation.

## Executed command manifest

The frozen all-file collection command was:

```bash
conda run -n ms python -m pytest --collect-only -q tests/config/test_train_config.py tests/artifacts/test_checkpoint_payload_identity.py tests/artifacts/test_checkpoint_writer.py tests/artifacts/test_training_state.py tests/artifacts/test_run_artifacts.py tests/artifacts/test_provenance.py tests/runtime/test_train_runtime.py tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py tests/training/test_pipeline_cache_preflight.py tests/training/test_pack_cache.py tests/training/test_pack_cache_determinant_registry.py tests/training/test_prepare_train_cache_cli.py tests/inference/test_pipeline.py tests/inference/test_artifacts.py tests/adapters/test_inference_reload_status.py
```

The command's RTK synopsis said `Pytest: No tests collected`, but raw stdout
captured through `bash -lc` showed exit code `0`, `776` output lines, and
`774 tests collected in 5.92s` on the post-audit rerun. Therefore the synopsis was not a pytest
selection result and is not a test gap.

| Command | Result |
|---|---:|
| `conda run --no-capture-output -n ms python -m pytest -q tests/config/test_train_config.py` | `96 passed` |
| `conda run --no-capture-output -n ms python -m pytest -q tests/artifacts/test_checkpoint_payload_identity.py` | `10 passed` |
| `conda run --no-capture-output -n ms python -m pytest -q tests/artifacts/test_checkpoint_writer.py` | `21 passed` |
| `conda run --no-capture-output -n ms python -m pytest -q tests/artifacts` | `239 passed` |
| `conda run --no-capture-output -n ms python -m pytest -q tests/runtime/test_train_runtime.py` | `68 passed` |
| `conda run --no-capture-output -n ms python -m pytest -q tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py` | `49 passed` |
| `conda run --no-capture-output -n ms python -m pytest -q tests/training/test_pipeline_assembly.py` | `54 passed in 7.61s` |
| the frozen 16-file manifest above, run through raw `bash -lc` stdout/stderr capture | `774 passed, 6 warnings in 84.36s` on post-audit rerun |

These are unit/control-plane receipts only: no GPU, production-shaped
distributed, performance, or exact-continuation claim is made.

The six warnings are Python `multiprocessing.popen_fork` deprecation warnings
from the named Gloo failure-control tests in `tests/training/test_exact_resume.py`.
They do not fail the suite, but Wave 3's fresh launcher packet must continue to
bound process behavior rather than treating this CPU control-plane receipt as a
CUDA execution proof.

The pipeline-assembly result above was isolated. A concurrent broader-suite
attempt observed a CUDA-initialization-order failure in this file after earlier
tests had contaminated process state; it is not substituted for, or combined
with, the isolated result and does not establish a CUDA launch claim.

## Authority scan

`openspec/specs/coordexp-swift-packing-forward/spec.md` supports only
`synchronous` and `overlapped`. Live source still admits `legacy_fused` and
`COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` in
`src/training/forward_input_provider.py` and `src/training/pipeline.py`; they
are unsupported residue for the later decomposition change and do not enter
these deltas. The complete legacy-owner inventory is
`src/config/models.py:ForwardInputProviderMode`,
`src/training/forward_input_provider.py:resolve_forward_input_provider_mode`,
`src/training/pipeline.py:_FORWARD_INPUT_PROVIDER_MODE_ENV`, and
`src/artifacts/run_writer.py:RunWriter.bind_forward_input_provider_mode`.
`docs/ARTIFACTS.md:47-57`, `docs/SYSTEM_OVERVIEW.md:123-127`,
`docs/COORDEXP_SWIFT.md:101-102`, and
`openspec/specs/coordexp-swift-vertical-smoke/spec.md:130-133` still deny an
accepted exact-resume feature. That canonical-doc/stable-smoke position remains
correct until the later qualification gates complete.
