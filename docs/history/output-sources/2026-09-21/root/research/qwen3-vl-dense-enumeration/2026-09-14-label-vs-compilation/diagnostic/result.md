# Free-h conditional continuation result

Status: **accepted complete execution / CPU analyzed**. All eight granted cells completed with exact bindings and zero free-c parity stops.

| Image | N16/A free TP50 | Literal c | Immediate literal w | Immediate geometric w | Frozen obligations N16 retained/lost; A retained/lost | Common / N16-only / A-only free owners |
|---:|---:|---|---|---|---|---|
| 477415 | 12/12 | exact/exact | yes/yes | yes 1.000/yes 1.000 | 9/1; 9/1 | 1266786,1277811,1278899,1309807,1598977,1599562,1603150,1999680,2156246,2205728,293197,433117 / none / none |
| 351017 | 5/3 | exact/exact | no/no | yes 0.826/yes 0.826 | 2/0; 2/0 | 499060,667769,667794 / 2094819,96050 / none |
| 417044 | 10/9 | exact/exact | no/no | yes 0.860/yes 0.860 | 10/1; 9/2 | 1079494,1080038,1082111,1082918,1083042,1083135,1083599,1572342 / 1079910,1083295 / 1083564 |
| 388795 | 2/2 | exact/exact | no/no | yes 0.988/yes 0.988 | 1/0; 1/0 | 1986212,375136 / none / none |

## Decision-bearing observations

- Exact `c` is the free prefix in all 8/8 cells. The entry-realization parity gate passes.
- Image 477415 produces exact immediate `w` at both endpoints. Images 351017, 417044, and 388795 never produce the registered literal `w`, but both endpoints produce an immediate same-class geometric match (IoU 0.826, 0.860, and 0.988 respectively). Literal and geometric success are therefore not interchangeable.
- Free owner sets are identical for N16/A on 477415 and 388795. Under common `h`, A loses N16 owners 2094819 and 96050 on 351017; on 417044 it loses 1079910 and 1083295 while adding 1083564.
- No supplied-`h` GT50 owner is re-emitted in any cell. The observed free owners are newly covered relative to supplied `h`; the reviewed unlabeled donut remains separate from GT-owner accounting.
- Every cell EOS-stops. Across all cells: zero geometry-invalid rows, zero other malformed rows, zero cap stops, and one strict repeat—A/388795 repeats exact `c` later.

## Resources and scope

Generated 1122 free tokens with 1122 model forwards and 8 image forwards. Sum of per-cell elapsed time was 460.108s (0.127808 GPU-hours by rank-time); outer elapsed time was 75.963s. Utilization was not measured.

Inference: on this four-case localization panel, A preserves `c` entry and immediate geometric successor behavior, but downstream annotation-owner compatibility is equal on two cases and lower/different on two. This supports a post-entry compatibility difference, not a c-realization failure.

Limits: existing empty-root outputs are contextual only; Stable50 suffixes are not target trajectories; geometric matches do not establish whole-suffix physical quality; GT-unmatched predictions are not treated as unlabeled owners. No fallback or further call is authorized.
