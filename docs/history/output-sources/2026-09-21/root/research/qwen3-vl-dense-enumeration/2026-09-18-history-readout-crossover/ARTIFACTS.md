# History/readout crossover evidence

- `execution-plan.json`:44 exact actual histories (11 boundaries × cuts4/8 × original/full),88 cells, source hashes, supplied-row identities and existing output debt. All cuts supported; no replacement cases.
- `sources/`: immutable executed producer versions. Accepted component runtime is reused unchanged; source aliasing and receipt-binding corrections are preserved separately.
- `runtime/<shard>/<cell>/release.json`: full target tokens/text/stop; exact supplied history and source identities; all-step raw/full shadow winners, competitors, margins, EOS/family probabilities. Inherited shared/centered shadows are descriptive only, not extra executed arms.
- Same cell `trajectory.pt`: every target-step actual final head input, raw/allfour shadow coordinate logits and emitted tokens. No attention/KV archive. Companion outputs are outside the estimand.
- `effective-readout.pt` and shard/cell receipts: model/operator/source identities, effective rows, coefficients, commands/costs/exit witnesses.
- `qualification-verification.json`, `verification.json`: independent CPU formula/token/history verification; all supported unchanged-policy overlaps are checked.
- `reduction.json`: full per-cell suffix metrics, crossing-cut recurrence with supplied/free row indices, and44 same-history future-policy contrasts. Supplied rows receive no free-output credit.
- `result.json`, `integrated-terminal.json`: parent synthesis and candidate closure, only after complete integration.

Reproduce without model calls:

```sh
python -m probes.training_set_completion.history_readout.reduce --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover --out /tmp/history-readout-reduction.json
python -m probes.training_set_completion.history_readout.verify --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-history-readout-crossover --out /tmp/history-readout-verification.json
```

Near recurrence means the accepted description-matched all-pairs8-bin numerical criterion, not physical identity. Across supplied histories positions and content differ. Same-history raw/full first forks share conditioning; later diverged states do not. EOS/shorter output/another loop alone is not physical recovery.
