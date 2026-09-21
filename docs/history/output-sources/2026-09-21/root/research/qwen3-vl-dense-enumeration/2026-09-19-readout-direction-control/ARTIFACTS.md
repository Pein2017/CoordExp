# Direction-versus-magnitude artifact map

- `execution-plan.json`, `selection.json`: frozen11 original failure boundaries,121 cells, qualification reuse, source/control bindings and8GPU allocation.
- `sign-vectors.json`: exact1000-bin balanced vectors for seeds19–26;500 plus/500 minus, hashes and generator convention. Frozen before model calls; no selected seed.
- `sources/`: executed producer and dependency snapshots/bindings. Accepted predecessor files remain unchanged.
- `runtime/<shard>/<cell>/release.json`: complete target tokens/text/stopping, per-step all11 shadow winners/top2/margins/EOS/family probabilities, source identities. Companion outputs are not evaluated.
- Same cell `trajectory.pt`: actual final head inputs, raw/full/applied/all11 shadow coordinate scores, emitted tokens. No all-layer/KV collection.
- Shard `effective-readout.pt`: actual effective coordinate rows, IDs and frozen FP64 norm factors. Full remains FP64 z*alpha cast once; reflected/sign formulas preserve coordinatewise absolute change in FP64, not necessarily bitwise after rounding.
- `qualification-verification.json`, `verification.json`: independent CPU formulas, magnitude matching/rounding residuals, corruption sensitivity, emission/readback and all22 original/full unchanged controls.
- `reduction.json`: every policy and boundary; separate recurrence/debt/stops and first forks with all11 operators on the SAME saved original state. Later diverged states are not controlled contrasts.
- `result.json`, `integrated-terminal.json`: parent-owned integrated candidate and closure after all work ends.

Reproduce without model calls:

```sh
python -m probes.training_set_completion.readout_direction.reduce --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-readout-direction-control --out /tmp/readout-direction-reduction.json
python -m probes.training_set_completion.readout_direction.verify --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-readout-direction-control --out /tmp/readout-direction-verification.json
```

Numerical recurrence is not physical-owner identity. Earlier EOS, another loop or changed token is not recovery. Eight sign vectors are fixed correlated-panel interventions, not independent images. No weighted success scalar, strength search or training-origin claim.
