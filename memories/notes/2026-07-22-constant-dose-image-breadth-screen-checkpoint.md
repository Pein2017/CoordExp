# Constant-Dose Image-Breadth Screen Checkpoint

Source session: active July 22 2026 CoordExp research loop.

Source handles:

* `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/results.md`
* `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md`
* `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/`

Captured: 2026-07-22 while trajectory generation was still running.

## Why this unit exists

The completed 118-image treatment learned its selected physical owners and
improved box quality and standard detection metrics, but it did not reliably
expand the final greedy physical-owner set. Source-missed selected owners were
recovered approximately four to six times as often as non-selected missed
owners, while images that supplied no gradient lost owners at every milestone.
The stable symptom was therefore narrow, learnable owner redistribution rather
than an absent gradient.

The next discriminator holds the training dose fixed and changes physical
image breadth. It compares a 118-image concentrated arm with a 496-image broad
arm. Each arm contains 496 sampled-route treatment events and 496 exact
Source-route preservation events, for 992 total events and 31 optimizer
updates at effective event batch size 32. This is not a larger epoch and not a
commitment to the current treatment family.

## Frozen data and comparison

The label-only candidate pool contains 2,432 images, with 608 images in each of
four object-count bands. It was split before route inspection into 2,048
training-candidate images, 256 development images, and 128 held-out images,
with 512, 64, and 32 images per band respectively.

Key frozen artifacts:

* candidate pool:
  `candidate-pool-v1/candidate-pool-2432.coord.jsonl`, SHA-256
  `133afcf6659b78d71893e9f79e568dca676515a05c7fe23f3c237de01350b7e2`;
* training-candidate split: SHA-256
  `c0efce5806e0ee487298f7b3b7697f4ebe663d1fa210b2e0e10e08de08ab5746`;
* development split: SHA-256
  `e4600b1998ad1269e351373e1c03caa39ff0a97b0474eb6852268b3c1736d0e4`;
* held-out split: SHA-256
  `8378af4429cc3cf2084da50a34fdf9b163e8d210291b3a787cdc05c1bdbd266e`;
* missing 2,176-image complement: SHA-256
  `f3001fdb046ad93ae2e881a8544d5418e3e4fc37ddf070b4fbefcdf299c527af`.

The old 256-image panel is a semantic subset of the new pool. Its membership is
215 training-candidate, 27 development, and 14 held-out images. The twelve
human-refined validation images remain safety evidence only and never supply
gradients.

## Event-difficulty control and claim boundary

An independent audit found that the first design would have compared broad
rank-one events against deeper concentrated events. The selector was repaired.
It now allocates exactly 124 pairs per object-count band in both arms and uses
this ordered fallback:

1. exact object-count-band by within-image selection-rank matching;
2. object-count-band by rank-one, rank-two, rank-three, and rank-four-or-deeper
   matching;
3. an explicitly policy-only broad-rank-one comparison.

The final receipt's `matching_mode` and `interpretation_scope` control what can
be claimed. Exact matching permits a breadth effect claim at fixed event dose,
band allocation, and exact ordinal-rank histogram, while still disclosing row
content differences. Coarse matching permits only a breadth comparison under
coarse rank control. Policy-only fallback does not permit a causal physical
image-breadth claim.

The two training seeds, 19 and 23, are matched optimizer seeds, not independent
cohort allocations. A second cohort allocation is a possible promotion check,
not a pilot launch blocker.

## Live trajectory generation

The trajectory root is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-2432-v1/`

The missing 2,176 images are being processed under request-scoped physical
batch size one. Eight GPUs run independent two-seed sampled shards, covering
seeds 31,001 through 31,016. Generation uses temperature 0.4, nucleus
probability 0.95, repetition penalty 1.0, and a 512-token generation limit.
Random state is reset per image and seed. The frozen producer is
`scripts/research/run_current_seeded_sampled_rollouts.py`, SHA-256
`9413a0d891daa59040b99043f719d5d7ee8b1007c260c00df030a98c589861c3`.

At capture time all eight sampled workers were healthy and no sampled JSON
shard had completed. The outer launcher was intentionally paused after its
children started so it cannot fall through to its original single-GPU greedy
stage. This did not pause the child sampled workers. Leave those workers
running.

After all sampled shards exist and their workers have exited, prevent the
paused outer launcher from starting its one-GPU greedy command. Generate seed
31,000 greedy trajectories as eight image shards under the same physical
batch-one semantics. Then validate the union of old and new greedy and sampled
artifacts, run the full 2,432-image trajectory analysis, and assemble both
StateBanks. Read the emitted `matching_mode` before interpreting or launching
training.

The rollout JSON format does not itself cryptographically prove which producer
created it. The accepted local evidence is the live command and process,
producer hash, decode parameters, request-major row ordering, seed replay, and
artifact hashes. Preserve this as a provenance limitation rather than adding a
post-hoc receipt that would not prove causation.

## Training and decision path after assembly

Four thin training configurations already bind broad versus concentrated arms
to matched seeds 19 and 23. Do not launch them until the real StateBank
manifests exist and one mixed-step loader and gradient smoke passes. If the
smoke succeeds, train all four arms for 31 updates and save steps 10, 20, 30,
and 31.

Use development to choose one shared step for both arms, then read held-out
once. The image-breadth hypothesis is supported only if both optimizer seeds
agree that the broad arm produces net held-out owner growth over Source and
the newly matched concentrated arm without unacceptable geometry, duplicate,
semantic, or output-health regression. Otherwise report the result as
inconclusive or stop this treatment family according to the unit's rules.
