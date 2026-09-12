# Frozen global-L2 amplitude and direction control

Status: **complete, lead-accepted and closed**; see`results.md`. The root grant
and exact six-worker placement amendment below are consumed, not a new launch
authorization.
Owner: `astra_high_instance_state`. Parent first panel remains lead-accepted,
closed and immutable. New raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/instance-state/amplitude-control/`.

## Question and root ruling

Does the positive417044coordinate-cache effect depend on A-donor direction,
or can the much smaller B/background deltas produce similar continuation when
given A's global amplitude? Root froze global joint float64 L2, not per-layer
or per-K/V renormalization. The original first-panel amplitude disparity is
max|delta|168.77(A),0.438(B),0.025(background).

Same image, Stable50, emitted history, common natural opener, native MRoPE and
four coordinate positions across all28layers. The shared opener keeps this
post-admission conditional; no autonomous admission/EOS mechanism claim.

## Eight fixed branches

1. Native/self cache, exact prior full-suffix reproduction.
2. Original A donor cache bytes, exact prior positive full-suffix reproduction.
3. B delta scaled to global L2(A).
4. Background delta scaled to global L2(A).
5. A delta downscaled to global L2(B).
6. One CPU float64 Gaussian direction, seed20260912, scaled to global L2(A).
7. A K-only direct donor keys with original native values.
8. A V-only direct donor values with original native keys.

For1/2/7/8, direct slices preserve exact donor/native bytes. For3..6, the
joint flattened tensor order is layer ascending, K then V, native tensor
flattening order, over precisely the four coordinate positions. Compute
delta and scale in CPU float64 and cast each final native-plus-delta slice
back to native float32. Zero donor/target norms fail closed; no fallback or
epsilon scaling. Observed realized delta must match the frozen global target
within relative5e-5 after float32 rounding. The Gaussian covers the same
joint coordinate-carrier shape, including layer0; it is not block-matched.
K/V releases are separately labeled component diagnostics, not norm-matched
comparisons or unique-component necessity claims.

Save all native/A/B/background raw slices in safetensors. Save global and
per-layer K/V L2,RMS,max diagnostics, scaling coefficients and random seed
before any continuation. No new seed, alpha, layer, head, image or owner
selection follows outputs. A null random direction cannot prove owner state.

## Single driver and native route

`probes/parallel_owner_research/instance_state_amplitude.py` owns this unit.
It imports the unchanged first-panel mask/cache/generation helpers.

Driver first captures four prefills onGPU5 and exits that child, preserving
raw donor slices and exact native MRoPE position/delta hashes. A GPU5gate
worker then independently prefills the same native history, requires exact
native selected-cache bytes versus the capture, and runs branches1/2. Both
full suffixes must exactly equal first-panel artifacts before any controls.

After that gate, launch six independent, disjoint workers concurrently:
GPU0branch3; GPU1branch4; GPU4branch5; GPU5branch6; GPU6branch7;
GPU7branch8. Root approved this placement-only amendment after GPUs4/6/7
were released, in response to the user's maximal-parallelism request. The
original unlaunched three-worker packet is preserved as
`packet-three-workers-unlaunched.json` (SHA256
`4b6ccf96b3ebebf98f15aa915dbd9d675381e9c363c044475b87ca14422acae6`).
No intervention row, norm rule, seed or output horizon changes. The existing
launcher and consumer iterate generic worker/stage lists; no scheduler rewrite.
No DDP or new architecture.
Each prefills once, checks the captured native cache bytes and native MRoPE
hashes, and consumes only the common opener through native cached generation.
No image forward is allowed during decoding. Every stage snapshots its exact
runner, dependency and packet. Full logs, exit codes and resource receipts
persist; one driver owns all waits and records a failed stage without retries.

Root-launch candidate:
`python -m probes.parallel_owner_research.instance_state_amplitude launch --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/instance-state/amplitude-control/run-v1`.

Packet: new raw root `packet.json`; binds prior panel and both exact output
reference hashes. Root granted capture/gate/full/cold execution on the amended
physical GPU placement; no repeated approval is needed for this correction.

## Resources, consumer and stopping rule

Eight full continuations, each at most3064free tokens; original total action
limit3084. Eight model loads total, eleven total vision/model prefills, at most
24,512decode forwards plus11prefills. Estimate10..13GBGPU and12..16GBRSS per
worker, capture payload approximately4MB, outcomes a fewMB, and parallel
completion roughly10..30minutes from prior measured throughput. These are
estimates, not a portfolio spend ceiling.

Cold consumer reopens stage/output hashes, validates all8branch identities,
literal token/text/prefix consistency, and uses the existing native parser and
global matcher atIoU50/60/80. Preserve all malformed/geometry-invalid drops,
raw starts, strict native-pixel later-row IoU>.95 duplicates, caps/EOS and
physicalA/B review. Useful continuation must not be called a perfect ledger:
the original positive repeats physicalA below.95 and includes one giant donut
extent. Report whether controls reproduce that whole output or another route.

Stop after these8fixed outcomes and the bounded interpretation. Stronger
architecture, owner-state, generalization and promotion claims are not granted.
Any additional contrast needs a new root ruling; no adaptive seed/dose search.

## CPU acceptance

`python -m pytest probes/parallel_owner_research/tests/test_instance_state_amplitude.py -q`:
3passed. Tests distinguish one global norm from per-layer matching, exact
self/A/K-only/V-only bytes, seeded random determinism, realized norm targets,
and zero-norm fail-closed behavior. Existing first-panel3tests also retained.
