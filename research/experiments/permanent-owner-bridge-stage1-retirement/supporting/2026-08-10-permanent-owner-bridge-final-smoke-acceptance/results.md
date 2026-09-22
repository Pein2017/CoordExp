# Permanent owner bridge: final-tree smoke acceptance (2026-08-10)

## Status and claim boundary

This record binds the final Stage 1 mechanics evidence executed from repository
commit `80ded9ad21302b061bf21330f45ee14a8912c549` (`80ded9a`). It is a
technical-validity and infrastructure record. It does **not** establish better
recall, duplication, atom AP, stopping, dense-enumeration quality, or any other
model-quality outcome. Smoke loss and accuracy values are deliberately not used
as launch gates.

The accepted claim is narrower: the final-tree single-rank and eight-rank paths
completed their bounded schedules with finite applied updates; published bridge,
adapter, selected-embedding, event, and checkpoint artifacts were readable; the
fresh dynamic-HF compositions completed bounded lifecycle decode; and the W8
first-update collective receipt was rank-identical. This record does **not** say
that production training was launched.

The scientific question remains unresolved by these smokes. The production
checkpoint and its later natural greedy evaluation, rather than this record,
own any model-behavior conclusion.

## Source boundary

- Worktree: `/data/CoordExp/.worktrees/permanent-owner-bridge`
- Branch: `codex/permanent-owner-bridge`
- Executed code identity: `80ded9ad21302b061bf21330f45ee14a8912c549`
- Parent checkpoint lineage: four-coordinate `geo_sorted_xy` step 2444, source
  resolved-config fingerprint
  `89e5af1269c42bdcc28b7175c8c701a4b57673e4b2fa8255c8912ee8800283ae`
- Stage 1 smoke semantics: COCO, no-resize 1024, global length 12000,
  `ordinary_partial`, four presentations (`geo_sorted`, `random-1`,
  `geo_sorted`, `random-2`), effective batch size 24, and teacher-forced eval
  using the training forward.
- This note synthesizes immutable files below. OpenSpec remains the formal task
  authority; this research record does not authorize or attest a production
  launch.

## Accepted W1 exact-module smoke

| Field | Bound value |
|---|---|
| Config | `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_single_gpu_ebs24_train1_val1_4epoch_warmup0p1.yaml` |
| Config fingerprint | `a641b422790ce4bec2716e8b78914d219bdc80616b09203fa6616947731e734d` |
| Run root | `outputs/smoke/coordexp_swift_owner_bridge_single/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_single_gpu_ebs24_train1_val1-20260810T135809Z` |
| Runtime | W1, gradient accumulation 24, effective batch size 24 |
| Terminal state | `completed`; 4/4 planned steps; `consumed_packs=96` counts 96 physical scheduled slots (4 real packs plus 92 shadows), not 96 data packs; final finite status `finite`; final optimizer update `applied`; no terminal error |
| Logging/events | 8 logging rows (4 train, 4 eval); 12 logical owner-bridge events (4 train, 8 eval) |
| Checkpoints | Four checkpoint events and payload directories at steps 1, 2, 3, and 4; `final.json` and `best.json` both resolve to step 4 |
| Final bridge payload fingerprint | `0b3cd889e0b3a95cb41f0a719aa90ccabb1863ce4b92e7041c30277120031eac` |
| Final lineage fingerprint | `9c5830529036d391d729ff7cdeabd74a6ab9bddafd81fd1e4e46cb2b5fc39dce` |

The eight logical eval landmarks are preserved even though midpoint and end
coincide physically at each of the four one-step presentations. Their event IDs
remain distinct (`tf_eval:p{0..3}:{midpoint,end}`), while only four physical
eval forwards/log rows are required.

All four updates are finite and reported as safely applied. Because this
four-step smoke resolves one warmup step, the first applied update has zero
learning rate; three updates move parameters. This short-schedule degeneracy is
not used to interpret the production ramp or optimization quality.

## Accepted W8 bounded smoke

| Field | Bound value |
|---|---|
| Config | `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1.yaml` |
| Config fingerprint | `ca563a0a5fc36328b7767c45707631a340357f1bb58459d509865843d555e629` |
| Run root | `outputs/smoke/coordexp_swift_owner_bridge/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T133717Z` |
| Runtime | W8, gradient accumulation 3, effective batch size 24 |
| Terminal state | `completed`; 8/8 planned steps; `consumed_packs=24` is the runtime physical-slot counter; final finite status `finite`; final optimizer update `applied`; no terminal error |
| Logging/events | 16 logging rows (8 train, 8 eval); 12 logical owner-bridge events (4 train, 8 eval) |
| Checkpoints | Four checkpoint events and payload directories at steps 2, 4, 6, and 8; final alias step 8; best alias step 2 |
| Final bridge payload fingerprint | `dbc4d1968486e88e228a0cecc1605f3d93a1ba4186c9962d9e9b9d7b9de5de06` |
| Final lineage fingerprint | `5c3b72844f73517c82691c653cc5b58e7b2410de558387c30c1ec373655918df` |

This run reached the full bounded budget after the final runtime and artifact
aggregation repairs. It supplies W8 mechanics evidence only; its short-horizon
metrics are not a scientific result.

## W8 first-update collective choreography

Receipt:
`outputs/smoke/coordexp_swift_owner_bridge/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T133717Z/runtime_choreography/step-000001.json`

- Status `pass`, W8, planned step 1, accumulation 3, finite/applied update.
- All ranks reported the same canonical digest:
  `f0fed8b904401fe4731d39a9499feddf0ff8e2d1db50c328e99a7994d6702c9e`.
- Each rank's post-baseline NCCL trace contains 2 broadcasts and 2 all-reduces;
  the generic Gloo trace contains 6 all-gathers.
- Receipt file SHA-256:
  `db51188a110a3f65d13df3a74bd9cdf63f6cb68d806e3fdfa34fc0e52ceb9c05`.
- The recorder attests per-recorder operation order, ranks, sequence IDs,
  profiling names, shapes, dtypes, and timeouts. It does not attest tensor
  values, broadcast root, reduction operator, device/stride identity, or a
  total order across the NCCL and generic recorders.

This closes the observed rank-divergence mechanism at the tested first-update
surface. It is not a proof that every future conditional execution path is
collective-safe.

## Fresh dynamic-HF reload and lifecycle smokes

### W1 fresh-HF

- Materialized config:
  `configs/coordexp_swift/infer/materialized_owner_bridge/owner-bridge-stage1-w1-fresh-80ded9a.yaml`
- Materialization receipt: sibling
  `owner-bridge-stage1-w1-fresh-80ded9a.materialization.json`
- Resolved inference config fingerprint:
  `8dc772a16d0b4c6ad0a37b36a2a0eff3a6be554f8b62bbcf1e72cbd339028c2a`
- Composition fingerprint:
  `a9dd86f16a82bb610fb945937cfe198295773c35e0af49dfb88d7b294bc4bcdc`
- Materialization receipt SHA-256:
  `770d4725a8a642721e465b4dda96eba0d64fdd1f5145484341a063384d8127cd`
- Output root:
  `outputs/smoke/coordexp_swift_owner_bridge_w1_hf/owner-bridge-stage1-w1-fresh-80ded9a`

### W8 fresh-HF

- Materialized config:
  `configs/coordexp_swift/infer/materialized_owner_bridge/owner-bridge-stage1-w8-fresh-80ded9a.yaml`
- Materialization receipt: sibling
  `owner-bridge-stage1-w8-fresh-80ded9a.materialization.json`
- Resolved inference config fingerprint:
  `a37a484b425796ef9a3ec9d326dd8b5557752ebad55578c3d280ff5126ee2fdf`
- Composition fingerprint:
  `58fdac7f1efae0e8fee96e377c279895bfae8ae5c755e0e7b4b4ff598309c352`
- Materialization receipt SHA-256:
  `822c7dd3df6b2df5a8b107ce11492d8c6866482a03a998fabf17a1ca7c2bea6f`
- Output root:
  `outputs/smoke/coordexp_swift_owner_bridge_w8_hf/owner-bridge-stage1-w8-fresh-80ded9a`

Both summaries are `completed` and report 2 raw, 2 scored, 2 diagnostic, and
2 lifecycle rows; 2/2 decode successes stopped at `im_end`; lifecycle-invalid,
parser-failure, image-validation-failure, score-failure, dropped-prediction,
and truncated-decode counts are all zero. Each run records 43 token-trace rows.
This proves fresh load, bound composition, artifact production, and the tested
batched lifecycle surface. It does not make either two-row smoke benchmark
eligible.

Image-binding caveat: the runtime lifecycle rows expose a batch-scoped image
binding structure that is identical across these two rows. Row identities and
artifacts remain isolated, and the lifecycle unit suite exercises per-sequence
isolation, but these runtime receipts alone do not independently prove
per-request image isolation. No stronger image-binding claim is made here.

## Algorithm and execution-shape batteries

The frozen-tree CPU algorithm receipt is:
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-10-permanent-owner-bridge-wave5-algorithm-accuracy/receipt.json`.
It records battery
`permanent-owner-bridge-stage1-wave5-algorithm-accuracy-v2`, source-set SHA-256
`a35e1c016a4d44de8b581b955da543d23738d7432cc0e50806293d6d737fee9c`,
and 23/23 passing cases on clean commit
`5ecbae86e7a68e89f11ac68985e1a672226e9db4`. The following commit only
publishes that refreshed receipt and changes no executable source or config.

Its packed-lane evidence
`wave5-production-owner-use-lane-packed-vs-full-v1` has evidence SHA-256
`62e83db542e08156bb50831e9752d2570858ee1b740e496b7537a14ba29f6be1`.
It compares one packed physical forward against four isolated/full-pack
forwards, reports exact `cu_seq_lens_q/k = [0, 18, 36, 54, 72]`, equal
`L_use`, logits and likelihoods within `atol=1e-6, rtol=1e-5`, all 143
parameter gradients within tolerance, and typed fail-closed outcomes when an
interior boundary is removed. This is deterministic CPU production-seam
evidence; it is not a live GPU FA2-kernel proof.

The canonical suite also exercises a real, unmocked two-rank Gloo acceptance in
`tests/runtime/test_owner_bridge_ddp_anchor_choreography.py`: uneven local
owner-use branches (4 versus 0), mixed physical slots, two wrapped anchor
forwards per rank, one optimizer step, identical trace digests, complete
gradient reach, and gradient/parameter parity to the single-process reference.
The final-tree canonical pytest aggregation reports 2000 passing tests and one
explicit CUDA-hidden skip out of 2001 collected tests. Its exact command,
versions, canonical `pytest.ini` scope, skip identity, import/residue commands,
and JUnit SHA are retained in
`../2026-08-10-permanent-owner-bridge-final-readiness/receipt.json`; the
algorithm battery separately reports 23 passing cases. These counts are
mechanical acceptance summaries, not scientific denominators or quality
measurements.

## Failed W8 attempts and what their repairs mean

Failed attempts are retained as negative infrastructure evidence. A repair
enables a fresh run; it does not retroactively convert a failed attempt into
scientific evidence.

| Run timestamp | Observed terminal boundary | Interpretation and narrow repair meaning |
|---|---|---|
| `20260810T025027Z` | Failed at 0/8 with Gloo `Connection closed by peer`; rank logs showed ranks 0/4 at NCCL work 77 while the other six ranks reached 79 and hung on broadcast sequence 78 until the watchdog | Confirmed divergent collective choreography. The execution-shape repair made one wrapped anchor forward per physical slot and moved rank-varying auxiliary branch forwards to the unwrapped module. This has no model-quality meaning. |
| `20260810T094510Z` | Failed at 0/8 with the same peer-close surface during the repair-validation sequence | Additional runtime failure evidence only; it is neither algorithm evidence nor a quality result. |
| `20260810T102608Z` | Reached one finite/applied step, then failed closed with `runtime.accuracy_stats_missing` | Exact distributed accuracy reduction required integer sufficient statistics rather than a silent mean-of-rank-means fallback. This repaired observability/metric aggregation, not the objective. |
| `20260810T115106Z` | Reached one finite/applied step, then failed closed because all rank-report payloads exceeded the 64 KiB control-plane limit | Bounded rank-artifact transport was compacted while preserving typed identity and failure behavior. This is control-plane/artifact observability evidence only. |
| `20260810T130112Z` | Reached two finite/applied steps, then artifact publication rejected `RMS caps disagree` at train step 2 | Same-step cross-rank RMS agreement remains strict; presentation-level aggregation now admits the configured nondecreasing row-cap ramp and merges count/sum/max/clipped fields. This repairs temporal artifact aggregation only and does not alter the configured RMS computation or establish training quality. |

The first two failures localize rank-collective execution shape; the next three
localize exact metric reduction, bounded artifact transport, and temporal
presentation aggregation. None is evidence for or against atom matching,
routing, owner use, autoregressive learning, recall, duplication, or stopping.

## Disposition and continuation

- **Infrastructure disposition:** bounded W1, bounded W8, first-update W8
  choreography, checkpoint publication/reload, fresh-HF lifecycle, real
  two-rank execution shape, and packed-versus-isolated algorithm surfaces are
  accepted at their stated scopes on the `80ded9a` tree.
- **Scientific disposition:** no model-quality conclusion; natural greedy
  production-checkpoint evaluation remains the decision-bearing surface.
- **Authority boundary:** this record does not attest live GPU headroom, an
  at-most-once production guard, a production process, or a first production
  heartbeat. Those remain separate launch and operations evidence.
