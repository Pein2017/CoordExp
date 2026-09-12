# Eight-branch acquisition launch

Root grant: two physically admitted pairs on351017/417044; retained first-person
prefix P; natural/A/B/AB each; GPU0 only; no target substitution or extra arms.
Training remains a separate ungranted phase.

Input SHA256:
`727b61c62087e866f5a65527aac1ec9e4c0913bf4e0df0dcc3903c8b7780114d`.
Five focused tests passed. No matching composition acquisition process existed
in the immediate prelaunch process inventory.

```bash
CUDA_VISIBLE_DEVICES=0 python -m probes.parallel_owner_research.composition acquire \
  --input /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/composition/acquisition-preparation-v1/input.json \
  --output-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/composition/acquisition-v1
```

Complete stdout/stderr is redirected to lane-root `acquisition-v1.log`; the
shell exit code is preserved in `acquisition-v1.exit-code`. The producer owns
`acquisition-v1/launch.json`, raw `rows.jsonl`, native `reduction.json`, loaded
identity, and terminal resource/status receipt. A technical failure does not
authorize a changed target, prefix, condition, dose, or scientific denominator.
