# Root-granted residual admission launch

Root grants physical GPU0 after history released0,1. GPU1 remains unassigned.
Root independently replayed15 tests and verified the exact packet before grant.
This launch contains exactly TWO new P+A+residualB continuations and reuses
the accepted P+A baselines. No training, diagonal reruns, alternate pair/order,
backfill, or further acquisition is authorized.

Input SHA256:
`a16a26d28b8221ccbe1d9e4d84a16eed691229d55bd5fa9181425b9e0dc92ead`.
Immediate prelaunch checks found no existing composition acquisition process;
input bytes match the root-granted SHA and all three exact output targets are
absent. Expected unrelated GPU stress occupancy is not an idle-wait gate.

```bash
CUDA_VISIBLE_DEVICES=0 python -m probes.parallel_owner_research.composition acquire \
  --physical-gpu 0 \
  --input /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/composition/residual-preparation-v1/input.json \
  --output-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/composition/residual-acquisition-v1
```

Complete stdout/stderr: lane-root `residual-acquisition-v1.log`.
Shell exit code: lane-root `residual-acquisition-v1.exit-code`.
One live invocation owns the producer's launch, raw rows, loaded-model identity,
native reduction and terminal receipt. Stop after cold CPU replay and the scoped
admission result. Technical failure preserves evidence and is not a scientific
negative or authorization to change the contrast.
