# Physical-admission join

This task-local adapter binds the frozen 502 candidate jobs to their real
source JSONL ordinals and converts the heterogeneous review outputs into the
exact review-index schema consumed by `materialize_physical_bank`. It emits
proposal evidence only; it never emits root decisions or a training bank.

Run from the repository root:

```bash
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput/physical-admission-join/build_join.py \
  --inputs research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput/physical-admission-join/inputs-v1.json \
  --output-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/physical-admission-join
```

Rerun after the eight recovery `decisions.jsonl` files advance. The adapter
validates each saved group against the root-sealed resume manifest and reports
remaining exact group/job gaps. The emitted consumer index can be passed to
`python -m probes.owner_successor_scale.training physical-source-preflight`.

`repair_final_aliases.py` is the fail-closed final-v1 to final-v2 repair for
consumer-required alias provenance. It derives execution aliases from exact
frozen visual-group membership and derives each row's same-image aliases from
the other jobs in that exact group; co-image membership alone never creates an
alias.
