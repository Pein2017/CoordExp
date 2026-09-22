# Shared execution result

Technical status: **candidate passed; root acceptance pending**.
Scientific status: not applicable; plumbing/update compatibility only.
Stop reached: one authorized two-rank C2 run, one saved-adapter cold load,
and one comparison to the accepted old eight-rank C2. No retry, extra eight-rank
run, tolerance relaxation, scientific training, endpoint claim or promotion.

## Evidence

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/shared-execution/qualification-C2-two-rank-v1`.

| Artifact | SHA256 |
|---|---|
| `receipt.json` | `231bc6e0663f0824720224453c80e8f67b564ef1e3a9b4a3c4b034e743be67cf` |
| `cold-check.json` | `59bbdf7118a1059588e79205ae3eff6c0eb5d274fce61aea665e075f605c0c73` |
| `qualification.json` | `752457486530c1e44579d53ac3f8ae663cc06775827b18ebdc15f3d584b49e88` |

Input: adjacent `preparation-v1/inputs.json`, SHA256
`739a9323fc5ac8a86992183a8b1a190e2d1803484fc92aaa077f79a57b92d709`.
Fresh `training.verify_receipt` passed after execution. All 30 frozen comparison
checks passed. Launch/cold/comparison exit codes were all zero.

## Numerical observations

- Exact 588 fp32 adapter tensors / 18,006,016 scalars.
- Maximum absolute adapter difference versus old C2: `7.32135958969593e-7`.
- Relative L2 difference between adapter-change vectors: `6.508237609807499e-5`.
- Adapter-change vector cosine: `0.9999999978821753`.
- Step 1 global objective components and gradient norm match exactly.
- Step 2 component deltas: positive `1.9073486328125e-6`, conditional KL
  `8.60891304910183e-8`, normal KL `7.86756572779268e-7`, margin
  `4.461774369701743e-5`. Raw gradient norm delta `-0.006744384765625`.
- Final positive score maximum absolute difference: `5.245208740234375e-5`;
  discrete target-argmax/token counts match.
- Every cold/live score difference is exactly zero; discrete counts match.

Inference: the literal-row two-rank route preserves the frozen objective and
has a numerically compatible optimizer update and saved/cold consumer. The old
oracle has no archived raw-gradient vectors, so no vector-gradient-parity claim
is made. This says nothing about scientific benefit of new positive records.

## Measured execution bounds

| Observation | Value |
|---|---:|
| Rank 0 / rank 1 model+image forwards | 133 / 127 |
| Training replays/backwards per rank | 68 |
| Synchronized backwards / updates per rank | 2 / 2 |
| Maximum whole-rank elapsed seconds | 188.979 |
| Maximum CUDA allocated bytes | 11,388,423,168 |
| Maximum CUDA reserved bytes | 13,656,653,824 |
| Maximum rank RSS bytes | 12,369,846,272 |
| Rank 0 / rank 1 reference cache bytes | 2,075,090,640 / 1,653,110,760 |
| Cold forwards / elapsed seconds | 3 / 9.109 |

All declared runtime bounds passed. Full raw counters, CUDA observations,
terminal states and reference hashes are retained. Physical GPUs 2,3 have no
remaining shared-execution invocation; only root may reallocate the reservation.

## CPU contract evidence and consumers

Root independently reran the focused+legacy suite: **44 passed**. Tests cover
complete-row sums, explicit denominators, independent conditional histories,
literal unknown-mask exclusion, current full-vocabulary margin competitors,
unequal token lengths and two/eight-rank uneven partition gradient+AdamW parity.
An intentionally missing world factor fails the same numerical oracle. A real
two-process CPU Gloo DDP test covers uneven local replay counts with one final
synchronization. Scoped worker logs are adjacent `cpu-tests-v2.log` and
`cpu-tests-ddp-v1.log`.

Code: `probes/parallel_owner_research/training.py`; tests: `tests/test_training.py`.
All old receipt-bound producers remain unchanged. History has successfully
consumed `prepare_packet` and CLI `verify`, with its own explicit 32-step packet.
Its fixed records and endpoint receipt/cold schema are compatible. Composition
must still freeze scientific history/conditional choices with root before fitting;
that is not an unresolved shared-engine normalization choice.
