# N4 low-norm mechanism decomposition: terminal receipt

## Disposition

**The N4 norm advantage is not a format/EOS trick. Coordinate rows are the
dominant carrier, category rows make a smaller but real contribution, and the
five format/EOS rows are negligible.** The frozen verdict is
`NON_FORMAT_ROWS_NECESSARY`.

The residual is low effective-rank by energy (`8.515`), but it is not safely
rank-16 compressible under the exact margin contract. Rank 16 retains `88.09%`
of energy yet preserves only `163/273 = 59.71%` of positions; among the frozen
tested ranks, exact position retention first returns at rank 128. Thus “low
norm” does not mean that a tiny rank is enough to reproduce every constraint.

This is a CPU-only decomposition of the already sealed N4 semantic train
payload. It neither changes the failed N2 held-out transfer result nor proves
N256 scaling, learning, or architecture suitability.

## Row-family evidence

The tokenizer-derived partition exactly reproduces all 1,136 frozen rows:

| family | rows | residual energy | energy share |
| --- | ---: | ---: | ---: |
| coordinate | 1,000 | 0.101846 | **93.5948%** |
| category | 131 | 0.006967 | **6.4022%** |
| format/EOS | 5 | 0.00000325 | **0.0030%** |

The full FP64 and production-shaped FP32 replays both retain all `310,128`
registered constraints and all 273 positions at threshold `0.00998`.

Removing coordinate rows reduces coordinate-target success from `116/116` to
`12/116` (`-89.66` percentage points) and overall success to `169/273`.
Removing category rows reduces category-target success from `37/37` to
`28/37` (`-24.32` points) and overall success to `264/273`. In contrast,
removing all five format/EOS rows retains `272/273` positions; format/EOS alone
retains only `160/273`, essentially the no-residual baseline of `159/273`.

The causal replay and energy accounting agree: the low-norm semantic solution
primarily reprograms coordinate-token readout, with a secondary category
readout component. It is not mostly a stop-token or wrapper calibration.

## Singular-direction evidence

| rank | energy retained | positions retained |
| ---: | ---: | ---: |
| 1 | 24.13% | 159/273 (58.24%) |
| 4 | 59.88% | 163/273 (59.71%) |
| 8 | 76.08% | 161/273 (58.97%) |
| 16 | 88.09% | 163/273 (59.71%) |
| 32 | 96.19% | 165/273 (60.44%) |
| 64 | 99.57% | 166/273 (60.81%) |
| 128 | 100.00% | **273/273 (100%)** |

The apparent discontinuity is expected for a minimum-norm solution with many
near-active margins: small-energy tail directions can be indispensable to a
few exact inequalities. At the predeclared resolution, the constraint-complete
rank lies in `(64, 128]`; this study did not add post-hoc ranks to narrow it.

## Interpretation and next discriminator

The strongest remaining alternative is no longer “format calibration.” It is
that four images share a coordinate/category readout geometry that does not
continue to scale. The shortest discriminator is therefore a **train-only N8
exact semantic-versus-K4-null scaling point**, reusing the sealed N256 semantic
capture and the same fixed row surface. Its decision should compare the N8 norm
and difficulty-normalized cost against N4 and all four N8 nulls; it still must
not reopen screen-dev or claim transfer.

## Identity and stop boundary

- Contract: [n4-mechanism-amendment-v1.md](n4-mechanism-amendment-v1.md),
  SHA-256 `995d5bb5b30f79df1d83e6b42150ba2894738fba19c4dc88fa970e38714a6a62`.
- Analyzer: `scripts/research/analyze_n4_output_qp_mechanism.py`, SHA-256
  `ec23b854bda8bb5ac2df1c1015ab6a74328d288d6d99bef8ef4bc0b845cdb818`.
- Terminal receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-n256-shared-output-qp-norm-scaling/n4-mechanism-v1/mechanism.json`,
  SHA-256 `86a2a8f6892f2117733c9b13febb5567773dd6db4ec80b68862a5e449513352b`.
- Bound capture SHA-256:
  `1e0e724cd7824110caa6a10ae3ba71fb5f06489540d39e00cb4e301e535d48bf`.
- Bound semantic N4 payload SHA-256:
  `457ca97205cf333280ee76cdb6d00de3cfa596f3798c454a38519dfd857cce4e`.

The mechanism amendment stops here. No rank-truncated payload, N8 solve,
screen access, decode, or model/GPU execution was performed.
