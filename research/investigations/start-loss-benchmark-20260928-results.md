# Fourth-loss benchmark: partial results

Status: current round completed; all subsequent arms await explicit user authorization.

All four runs completed 256 updates with finite losses; eight val200 evaluations completed. Frozen config hashes and identical COCO ground truth verified.

| Run | mAP (%) | FN50 | Exact repeats | Invalid geometry | Length caps |
|---|---:|---:|---:|---:|---:|
| local_mass-order29-step256 | 46.6924 | 613 | 371 | 6 | 1 |
| ce-order17-step256 | 46.6304 | 611 | 406 | 5 | 1 |
| instance_margin-order29-step256 | 46.3842 | 614 | 8 | 4 | 0 |
| control-order17-step256 | 45.9902 | 614 | 382 | 8 | 1 |
| source | 45.5651 | 614 | 460 | 220 | 2 |

Matched order17: extra CE vs control = +0.6402 AP points, FN50 -3; repeats +24.
Matched order29: local mass vs instance margin = +0.3082 AP points, FN50 -1; repeats +363.

No replicated winner: order29 has no three-loss control and order17 has no local-mass or margin arm. Cross-order ranking is descriptive, not a causal loss comparison. Step64 results remain diagnostic; the frozen primary endpoint is step256.

Artifacts: /data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928/comparison.json; /data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928/partial-acceptance.json.
