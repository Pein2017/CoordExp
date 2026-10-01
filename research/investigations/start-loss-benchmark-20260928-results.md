# Completed fourth-loss benchmark

All 8 runs and 16 evaluations completed. Primary endpoint is step256; step64 remains diagnostic. Config/data/source hashes, finite updates, identical COCO GT and norm=median verified. No norm-off experiment has been launched.

| Arm | AP order17 | AP order29 | Mean AP | Mean FN50 | Mean exact repeats | Mean invalid boxes |
|---|---:|---:|---:|---:|---:|---:|
| ce | 46.6304 | 46.7339 | 46.6821 | 616 | 412 | 8 |
| control | 45.9902 | 47.2036 | 46.5969 | 618 | 381.5 | 6.5 |
| local_mass | 46.2989 | 46.6924 | 46.4956 | 613 | 391 | 9 |
| instance_margin | 46.4420 | 46.3842 | 46.4131 | 620 | 16 | 5 |

No fourth loss improves final AP over the matched control in both orders. Extra CE leads mean AP by only 0.0853 points; the signs reverse across orders. Margin reduces exact repeats from 382 to 24 and 381 to 8, with zero length caps in both orders; FN changes by +12 and -8. Invalid boxes change only 8 to 6 and 5 to 4. Local mass has the lowest mean FN (613), but changes are -1 and -9 with no demonstrated consistent AP gain.

Decision: retain margin as a replicated exact-repeat/length-cap mitigation candidate under median norm, not an overall detection winner. A same-checkpoint control/margin norm-on/off inference comparison can test dependence on the decoder normalization; it is proposed, not launched. Two orders on one fixed cohort are exploratory replication, not a significance claim.

Authoritative final evidence: `/data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928/final-acceptance.json` and `comparison.json`.

## Historical progress records

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

## Authorized deadline reached: 2026-09-28 19:21 UTC

Six of eight arms and all their step64/step256 evaluations completed. Both remaining controllers stopped their training processes at the original deadline; no extension answer was received. All completed config hashes, finite updates and identical eval ground truth verified.

| Run | mAP (%) | FN50 | Exact repeats | Invalid geometry |
|---|---:|---:|---:|---:|
| ce-order29-step256 | 46.7339 | 621 | 418 | 11 |
| local_mass-order29-step256 | 46.6924 | 613 | 371 | 6 |
| ce-order17-step256 | 46.6304 | 611 | 406 | 5 |
| instance_margin-order29-step256 | 46.3842 | 614 | 8 | 4 |
| local_mass-order17-step256 | 46.2989 | 613 | 411 | 12 |
| control-order17-step256 | 45.9902 | 614 | 382 | 8 |
| source | 45.5651 | 614 | 460 | 220 |

Instance margin order17 stopped after 24 finite updates; control order29 after 20. Neither reached the first saved checkpoint (64), so completion requires restarting those two arms from the unchanged source checkpoint with a newly authorized deadline. These partial updates are not comparison endpoints.

Extra CE exceeds local mass in final mAP in both orders, but FN is better for CE in order17 and worse in order29. Margin deduplication still has only order29 evidence; no cross-order replication or complete control comparison is available.

## Restart authorized 2026-09-29

User requested inspection and restart. Stop logs confirm original deadline-triggered SIGTERM, with finite last updates and no saved checkpoint. The two interrupted attempts were archived under `attempts/deadline-stop-20260928`; exactly those two arms restart from the original source with unchanged config hashes. New deadline is recorded in `restart-20260929.json`. All 13 completed main inference artifacts (source plus six arms at two steps) resolve coordinate_output_norm=median. No norm-off runs have been launched; a same-order control/margin norm-off comparison is a proposed follow-up.
