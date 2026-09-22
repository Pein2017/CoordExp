# Coordinate-pair score-support evidence

- `execution-plan.json`: original11 failure prefixes, six supports,66 cells/22 controls, source/sign/model-input bindings and8GPU schedule. No altered histories or forced tokens; every row is free.
- `sources/`: executed minimal producer snapshot and dependency bindings; accepted predecessors remain immutable.
- `runtime/<shard>/<cell>/release.json`: all emitted tokens/stop, same-state six-policy competitors/margins/EOS/family probabilities, row/role positions and source identity.
- Same cell `trajectory.pt`: actual head inputs, raw/applied/six-policy coordinate scores and emitted tokens. Effective output rows and factors reside in each shard `effective-readout.pt`.
- `verification.json`: independent CPU formula and support reconstruction, bitwise untouched-coordinate checks, greedy emissions,22 exact original/sign23 controls and wrong-support corruption sensitivity.
- `reduction.json`: complete per-boundary/policy numerical metrics, including exact/near all-pairs counts (invalid literal rows retained), original-state first-fork shadows, debt and stopping separately.
- Runtime command/exit/failure receipts preserve attempts and measured cost. Parent `result.json` / `integrated-terminal.json` close the integrated candidate.

Reproduce without model calls:

```sh
python -m probes.training_set_completion.coordinate_pair.reduce --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout --out /tmp/coordinate-pair-reduction.json
python -m probes.training_set_completion.coordinate_pair.verify --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout --out /tmp/coordinate-pair-verification.json
```

Magnitudes intentionally differ: these are static score-support ablations, not magnitude-matched directions. All generated rows count. Near/all-pairs boxes are numerical proxies, not physical owners; lower repetition or EOS is not recovery. No all-layer/KV capture or visual review.
