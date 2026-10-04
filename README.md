# CoordExp

CoordExp studies coordinate-token visual detection and enumeration on pretrained
vision-language models. Source, research evidence and retained artifacts have
separate owners.

| Surface | Purpose |
|---|---|
| `src/`, `configs/`, `tests/` | Current implementation and executable contracts |
| `scripts/` | Maintained command entries |
| `public_data/`, `manifests/` | Data tooling, identity and recovery evidence |
| `docs/` | [Long-lived design boundaries](docs/README.md) |
| `openspec/specs/` | Stable compatibility-sensitive requirements |
| `research/` | [Route to canonical research and retained source material](research/index.md) |

Main owns stable integration, `coordexp-infras` infrastructure development/runs,
and `research-probes` canonical research. Resolve the exact checkout rather than
assuming the branches have identical capabilities. See
[checkout ownership](docs/BRANCH_AND_WORKTREE_POLICY.md).

Use local source, exact-checkout CodeGraph, CLI help and typed configs for current
commands and options. Do not copy a historical launch recipe as an active default.
Root `outputs/` is only for selected shared assets; branch runs use their physical
worktree. [Storage policy](docs/OUTPUT_STORAGE_POLICY.md) owns that boundary.
Historical docs are [sealed and recoverable](docs/RETENTION.md), not another
active source tree. No model launch or resource change follows from reading this
page.

## Validation boundaries

Tests live with their source owner under `tests/`; small size alone does not
justify removing a contract test. Default discovery includes both annotation
systems. Frontend contracts use Node 22. Tests marked `live_environment` compare
installed applications or external data with frozen receipts and remain enabled
by default. Use `pytest -m 'not live_environment'` for the retained CPU contracts
without that data/application gate, and `pytest -m live_environment` for the gate
itself. Local model-component canaries can still depend on installed libraries
and assets; passing one gate does not imply the other passed. No routine check
authorizes a model run.

## License
Pending project decision; preserve applicable upstream licenses.
