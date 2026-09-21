# COCO owner-focus training qualification

Date: 2026-09-07

Status: **qualified candidate mechanics; lead acceptance and all 64-update
production launches remain pending.** Scientific status is
`MECHANICS_ONLY_NO_MODEL_QUALITY_CLAIM`.

## Frozen implementation

`scripts/research/train_coco_gt_correction.py` retains the old portfolio as its
default identity and requires the successor to opt into experiment
`2026-09-07-coco-owner-focus-ablation`. The successor then fails closed unless
the arm is R/M/Rweak on the DoRA surface, bank ID is
`37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`,
Source identity is
`200c8fee10d20ee29b93a13c117613d27660686d3a07b9ebf202050484b58528`,
seed is `20260908`, and production is two ranks with global image batch 32 and
actual microbatch at least two.

Each actual microbatch left-pads complete multimodal trajectories, concatenates
their native pixel tensors and image grids, derives Qwen MRoPE positions with
the padded attention mask, and asserts that every attended position equals its
unpadded native position. It keeps only the aligned action-logit suffix but uses
the full 152,670-token vocabulary. Anchor and correction trajectories are
summed per image before the global image mean. Rweak uses R's whole mask and
full R denominator, multiplied per event by
`sum(M_mask) / sum(R_mask)`; its canonical anchor coefficient remains 1.
Frozen tensors, selected input embeddings, and old checkpoint/payload fields
remain guarded. The successor's experiment, objective variant, seed, global
batch and microbatch are included in the v1 checkpoint ID.

## Machine receipt and cold checkpoints

Authoritative receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/qualification/training-qualification.json`
(file SHA256
`13635e27661347c961c12e44b0b09190c097247d8ee7ec225169ad47adf05230`).
The verifier cold-loaded all payloads and rechecked their manifest, bank,
Source, recipe and producer identities.

| Arm | Effective-batch32, two-rank, actual-mb2 checkpoint | Checkpoint ID |
|---|---|---|
| R | `qualification/r-effective32-two-rank-mb2-v1/checkpoint-000001` | `1e7a2cc35153bab00de08ed70016e11a3ee9068bfa413e3e41b5ffb3d38ccc39` |
| M | `qualification/m-effective32-two-rank-mb2-v1/checkpoint-000001` | `d04d3bbaa5767f5e76349f4574b8c3a22bc97eb582b0d9ef8eb3e2af1fe3fe12` |
| Rweak | `qualification/rweak-effective32-two-rank-mb2-v1/checkpoint-000001` | `8b35005dc692a049d2a05915ee250ccbb9b37dbcdaa6cf6ba1ba996f6d384161` |

The common 32-image slice has 22 correction events and 10 anchor-only images;
it includes the longest 4,677-token trajectory. It is mechanics evidence only,
not a training outcome.

## Parity and resource evidence

The Rweak parity slice uses the same four images `368,7116,5156,9813`: two
unequal correction events and two images without corrections.

| Comparison | Global loss abs | Component max abs | Effective-gradient exp_avg max abs / relative L2 | Update max abs / relative L2 / cosine | Native positions |
|---|---:|---:|---:|---:|---|
| actual mb1 vs mb2, one rank | `7.153e-7` | `1.431e-6` | `1.876e-7` / `2.895e-5` | `1.322e-5` / `1.691e-3` / `0.99999857` | exact hash multiset |
| one rank vs two ranks, actual mb2 | `7.749e-7` | `3.100e-6` | `2.013e-7` / `4.299e-5` | `1.639e-5` / `2.954e-3` / `0.99999564` | exact hash multiset |
| effective32 mb2 vs mb4, two ranks | `5.215e-8` | `5.245e-6` | `1.141e-7` / `2.580e-5` | `1.620e-5` / `1.688e-3` / `0.99999858` | exact hash multiset |

Declared acceptance bounds are respectively `3e-6`, `4e-6`, `5e-7`, `1e-4`,
`2e-5`, `5e-3`, cosine at least `0.98`, and exact position-hash equality.
AdamW's first-step sign sensitivity makes max parameter difference less useful
than update relative L2 and cosine; all are reported rather than hidden.

The one-rank four-image update improved from 8.868s / 0.451 images/s at mb1 to
8.327s / 0.480 images/s at mb2 (+6.5% throughput), while allocated memory rose
from 10.70GB to 12.41GB. The bounded effective32 mb4 rung was then measured on
the exact same Rweak slice: it slowed from 47.569s at mb2 to 57.179s at mb4
(`-16.8%` speed gain; 0.673 to 0.560 images/s), while peak allocated/reserved
memory rose from 33.87/45.48GB to 54.97/81.28GB. Its maximum component-loss
difference, `5.245e-6`, also exceeded the declared `4e-6` parity bound. Per the
frozen `>5%` usefulness rule, **mb2 remains selected for all three arms** and
the batch ladder stops at four.
The longest-case actual batch contained three padded trajectories of lengths
1,372, 1,608 and 4,677, completed in 13.382s, and peaked at 26.32GB allocated,
29.63GB reserved and 12.27GB host RSS.

The production-shaped effective-batch32 updates measured:

| Arm | Update | Images/s | Peak allocated / reserved | Host RSS | Save |
|---|---:|---:|---:|---:|---:|
| R | 46.855s | 0.683 | 33.87GB / 45.48GB | 12.38GB | 0.828s |
| M | 46.845s | 0.683 | 33.87GB / 45.48GB | 12.55GB | 0.821s |
| Rweak | 47.569s | 0.673 | 33.87GB / 45.48GB | 12.56GB | 0.876s |
| Rweak mb4 rejected rung | 57.179s | 0.560 | 54.97GB / 81.28GB | 12.17GB | 0.698s |

Each checkpoint is about 216.8MB. Retaining 64 per arm extrapolates to about
13.88GB per arm / 41.6GB for all three; `/data` had 1.90TB free at qualification.
One-update timing plus save extrapolates to about 52 minutes per 64-update arm,
well below the inherited 12-hour ceiling, but this remains an extrapolation.

## Falsification and verification

The single load-bearing check covers both left-padded native positions/action
alignment and Rweak image weighting. Sensitivity was demonstrated twice before
final GREEN: reversing `M/R` to `R/M` failed with `2.0 != 0.5`; changing left
padding to right padding failed at `padding changed native trajectory positions`.
Both mutations were reverted.

```text
conda run -n ms python -m py_compile scripts/research/train_coco_gt_correction.py scripts/research/qualify_coco_owner_focus_training.py
conda run -n ms pytest -q tests/research/test_train_coco_gt_correction.py
# 5 passed
conda run -n ms python scripts/research/qualify_coco_owner_focus_training.py
# mechanical_status=QUALIFIED_CANDIDATE
git diff --check
# PASS
```

Source hashes at final verification:

| Artifact | SHA256 |
|---|---|
| bank manifest file | `4486f173a98e3ba253798f8aa868a0712056b0480b9a4ee8fc081dfa90412ed2` |
| bank manifest content identity | `b984331ccaff369edaa178c50dfad61dfec69898a2313a018c373f54f38e02cd` |
| trainer | `f59a457c90046eba693eb7d1f5c91982bc131e3d28069be58f8913a1f1626f97` |
| qualifier | `aa3897a1a7848aa2c2276a7c8e682425ff93be607db1b3126c76b6ea7e6ca194` |
| trainer test | `f3bf40421f0025e0551cddaef6a3605447d2f89135c0ca53e79fa6ba20fcb447` |
| checkout commit | `73b8b3cc2614db052055c7c49fc33d6207b0f80a` |

## Production commands — not executed

These roots must not already exist. One durable invocation owns each root.

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n ms torchrun --standalone --nproc_per_node=2 scripts/research/train_coco_gt_correction.py --experiment-id 2026-09-07-coco-owner-focus-ablation --bank-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json --arm R --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/R --max-updates 64 --global-batch-size 32 --microbatch-images 2 --seed 20260908
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ms torchrun --standalone --nproc_per_node=2 scripts/research/train_coco_gt_correction.py --experiment-id 2026-09-07-coco-owner-focus-ablation --bank-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json --arm M --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/M --max-updates 64 --global-batch-size 32 --microbatch-images 2 --seed 20260908
CUDA_VISIBLE_DEVICES=4,5 conda run --no-capture-output -n ms torchrun --standalone --nproc_per_node=2 scripts/research/train_coco_gt_correction.py --experiment-id 2026-09-07-coco-owner-focus-ablation --bank-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json --arm Rweak --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/Rweak --max-updates 64 --global-batch-size 32 --microbatch-images 2 --seed 20260908
```

## Remaining risks and stop

- Parity is within declared FP32 numerical tolerances, not bitwise update
  identity; near-zero AdamW coordinates can differ by about one LR-sized step.
- The timing is a one-update production-shaped extrapolation, not a measured
  64-update wall time. Peak memory did exercise the sealed longest trajectory.
- This package establishes mechanics, serialization and cold consumption only;
  natural holdout quality and the scientific ablation remain pending.

STOP at qualified candidate. No production arm, batch rung beyond four, seed,
sweep, model promotion, commit or push was started here.
