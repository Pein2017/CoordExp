# Readout common-component artifact map

- `execution-plan.json`: frozen25 boundaries ×4 policies, source bindings, qualification reuse, eight-GPU assignment and ceilings.
- `sources/runtime-readout-component-v1.py`: exact executed producer; predecessor loader/history/metric helpers remain unchanged.
- `runtime/qualification/`: four counted qualification continuations; `qualification-verification.json` independently reconstructs formulas and checks original288-token source parity.
- `runtime/<shard>/<boundary>--<policy>/release.json`: complete target token IDs/text/stop, source prefix identity, all-step four-policy shadow winners, margins, EOS and coordinate-family probabilities. Companion outputs are not interpretable or evaluated.
- Same cell `trajectory.pt`: actual final head inputs, raw coordinate logits, allfour shadow coordinate-logit vectors and emitted tokens, every targetstep.
- Same cell `receipt.json`: source/producer/plan/tensor bindings and measured cell costs.
- Each shard `effective-readout.pt`: effective1000 output rows, coordinate IDs, FP64 lower-median norm factors and mean row.
- Each shard `shard-receipt.json`: loaded model identity, aggregate cost, artifact hashes, completion/error status and PID.
- Runtime launch/log/exit receipts retain command and failure evidence. No KV, attention, intermediate layer archive or exhaustive physical review is captured.
- Parent `reduction.json`, `verification.json`, `result.json`, `integrated-terminal.json`: saved-output numerical outcomes, CPU tensor reconstruction, bounded scientific synthesis and candidate closure (created after completion).

CPU reproduction (no model calls):

```sh
python -m probes.training_set_completion.readout_component.reduce --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component --out /tmp/readout-component-reduction.json
python -m probes.training_set_completion.readout_component.verify --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component --out /tmp/readout-component-verification.json
```

All physical recovery/precision claims remain unestablished by numerical recurrence metrics. The14 proxy boundaries are not a healthy independent-image cohort. Later policy histories differ; only shadows at one retained state share conditioning.
