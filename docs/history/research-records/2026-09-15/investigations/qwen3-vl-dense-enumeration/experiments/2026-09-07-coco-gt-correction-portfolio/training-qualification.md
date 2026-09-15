# COCO GT-correction bank and training qualification

Date: 2026-09-07

Status: **candidate mechanics QUALIFIED; scientific status is
`MECHANICS_ONLY_NO_MODEL_QUALITY_CLAIM`; lead acceptance and production launch
remain pending.** No 64-update arm was launched by this qualification.

## Sealed bank

Durable bank:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/`

- Bank ID: `37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`.
- Manifest content SHA256:
  `b984331ccaff369edaa178c50dfad61dfec69898a2313a018c373f54f38e02cd`.
- `records.jsonl` SHA256:
  `00bc83da2516a2f56b6115e41dd16d4c6f1cb58705e4be620bc647806053b015`.
- Source identity SHA256:
  `200c8fee10d20ee29b93a13c117613d27660686d3a07b9ebf202050484b58528`.
- Input identities retained in the manifest include train JSONL
  `05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5`,
  dev JSONL `b2ba42e6be18ca179cbdac387ed7af5b8400cf88a5a7d63f3b46643f2f61a46c`,
  canonical plan `76024b2fa4ccd2c98f9df794963090985698f80e3b224c6af9dd0d2af49bfa5a`,
  and Source capture
  `7bff43fea168ea637411c1c064e045933fe760b08ae34c4fe4ed58b585ec2099`.
- 256 records and canonical anchors; 1,955 annotated owners: 1,259 covered
  and 696 missed at the frozen category-compatible global IoU50 assignment.
  There are 160 earliest-miss correction events, including 16 identical-policy
  events. Nineteen owners are tagged ambiguous and 316 tagged near-match; these
  tags do not filter the fixed population.
- Exact terminal parsing gives 252 native-EOS tails and four incomplete capped
  tails, with no empty tails. Corrections stop at the last complete object
  closure without invented EOS; every canonical anchor retains native EOS.
- Maximum token lengths are prompt 1,372, canonical action 344, correction
  action 3,315, and 4,677 total teacher-forced tokens.
- W's fixed train-independent output universe is 1,136 rows by hidden size
  2,048: COCO category pieces, the four row-grammar tokens, all 1,000
  coordinate tokens, and native termination.

The bank freezes raw Source token offsets, exact prompt/image/media identities,
canonical owner rows and ordering, global assignments, h_pre/h_post cuts,
uncovered owner sets, teacher strings, masks, denominators, and selected-owner
spans. R and M share the exact teacher string and fixed full-R denominator; M
zeros only nonselected correction-row numerator positions. Prefix/prompt direct
loss remains zero and every arm retains the same canonical anchor.

## Trainer and checkpoint contract

The single probe trainer is
`scripts/research/train_coco_gt_correction.py`. R/B/M activate only the 588
sealed Source language-DoRA tensors. W freezes Source and installs one FP32
`[1136, 2048]` output-row residual. Both surfaces recompute the complete current
prefix for every teacher-forced token; there is no detached hidden-state cache
or selected-row softmax. Image-normalized full-vocabulary CE, optimizer recipe,
global batch, seed and 64-update stop remain those in `unit.md`.

Each completed update is published atomically as a checkpoint directory with a
manifest and payload/optimizer hashes. Cold loading verifies checkpoint ID,
arm/surface, bank ID, Source identity, the live bank-manifest file and content
hashes, payload hash/schema/layout, and excludes optimizer loading. Resume adds
the optimizer/RNG payload, requires its arm, surface, completed update,
bank ID, Source identity and parameter layout to equal the checkpoint manifest,
and requires the same output root. Fault injection
changed a W payload from expected
`9fc013f2eb643de39407091c600728f6e61dfc22c13bfc6be8d18124321e6194`
to `3ea85186e8e70afc3542efdfb3d87d151c6bace9ba3870d3bba81beb72d95622`;
the cold loader rejected it as `output residual payload changed`.

Preferred fresh one-update checkpoints for downstream cold qualification:

| Surface | Checkpoint | Checkpoint ID |
|---|---|---|
| W | `qualification/w-fresh-frozen-check/checkpoint-000001` | `f35245868170f233a7be3209e61183f1460bf7dfe381be47ce499fedc6f1d10c` |
| DoRA (R) | `qualification/dora-fresh-frozen-check/checkpoint-000001` | `cc9b26c3d440f63ab0fec6e8e6f0349c997d7c9110d5a70efc9b177685e6a49a` |

These paths are below the durable pilot root.

## Qualification evidence

Machine-readable receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/qualification/training-qualification.json`.
Qualification uses fixed images `351017,477415`, at most two updates per
lineage, and GPUs 0/1 only.

- W and DoRA direct two-update runs equal the corresponding interrupted
  one-update plus resume-to-two runs exactly at both completed boundaries:
  loss difference and maximum payload-parameter difference are zero.
- One-rank versus two-rank global loss is exact for both surfaces. Maximum
  payload-parameter differences are `1.9148842511640396e-09` for W and
  `1.0244548320770264e-08` for DoRA, below the sealed `2e-8` agreement bound.
- The fresh post-invariant W and DoRA runs reproduce their original step-one
  loss and payload exactly. Their manifests report finite trainable gradients,
  zero frozen gradients, unchanged frozen parameter versions, unchanged
  selected input embeddings, and full-current-prefix recomputation. DoRA uses
  non-reentrant activation checkpointing after asserting every dropout module
  has `p=0`.
- Independent cold consumption passed for both preferred checkpoints. The W
  payload receipt is
  `qualification/cold-consumer-v1/w-payload-qualification.json`, SHA256
  `57c44202941f010cb9f4dc1bb5b332950b1b7e0d3155a8eaf186052be435d863`:
  actual FP32 `[1136,2048]` payload, 2,326,528 nonzero entries, exact selected
  logit delta, exact unselected logits, same-prefix hidden state unchanged, and
  exact Source restoration for zero/removal. Native cold HF one-image decode
  and scoring completed for W at
  `qualification/cold-consumer-v1/w/w-fresh-step1-image368-v5/` and DoRA at
  `qualification/cold-consumer-v1/dora/r-fresh-step1-image368-v1/`, each bound
  to the expected checkpoint ID.

Two prelaunch-review counterexamples were observed RED before correction:

1. Changing selected owner category token `8987` to `8988` consistently in the
   owner and R/B/M strings, then recomputing the action, record, bank,
   annotation-identity, records-content, file and manifest hashes, was accepted.
   Loading now recomputes the manifest annotation identity and proves every
   owner token row against its independently retained canonical-anchor row. It
   also reconstructs each correction from the named canonical owners and
   verifies exact prefix, suffix, span, selected span, mask and denominator.
   The fully resealed corruption now fails as `canonical owner row changed`.
2. Changing optimizer `bank_id` and `source_identity_sha256` to foreign values,
   zeroing every `exp_avg`, and updating the optimizer file hash was accepted.
   Resume now compares both optimizer lineage fields with the checkpoint
   manifest; the resealed payload now fails as `optimizer identity changed`.

Direct counterexample replay after correction:

```text
conda run -n ms pytest -q tests/research/test_coco_gt_correction_bank.py::test_bank_rejects_altered_token_or_owner_identity tests/research/test_train_coco_gt_correction.py::test_resume_rejects_resealed_foreign_optimizer_lineage
# Pytest: 2 passed
```

Fresh deterministic acceptance after final loader hardening:

```text
conda run -n ms python -m py_compile scripts/research/coco_gt_correction_bank.py scripts/research/train_coco_gt_correction.py scripts/research/qualify_coco_gt_correction_training.py
conda run -n ms pytest -q tests/research/test_coco_gt_correction_bank.py tests/research/test_train_coco_gt_correction.py
# Pytest: 7 passed
conda run -n ms python scripts/research/qualify_coco_gt_correction_training.py
# mechanical_status=QUALIFIED
```

Serena/Pyright diagnostics are empty for the three scripts and two test files.
The tests cover terminal raw-offset EOS/partial-tail handling, real-bank paired
masks, fully resealed semantic owner/token corruption, fixed-denominator
selected-row coefficients, resume-addressable scheduling, resealed foreign
optimizer lineage, and W logit/hidden-state locality.

Current code SHA256 values recorded by the machine receipt:

| File | SHA256 |
|---|---|
| bank | `25df771ecc10ab81582c7b7fa5aeb86c138b494e4a892ba09bcea5aaca9c2abb` |
| trainer | `d9d796c948a3464411b7efe713108ff3ad6d0f68c8afc848e5edd28f08f9a69e` |
| qualification | `7395841149043d0bc75187c9abd89c36dfd3de8b0fd11ff7a869aecc64e67715` |

## Measured bounds and interpretation limits

| Measurement | Qualification maximum / bound |
|---|---:|
| sequence tokens | 4,677 |
| GPU allocated | 16,060,559,872 bytes |
| GPU reserved | 28,122,808,320 bytes |
| host peak RSS | 12,490,358,784 bytes |
| update wall time | 14.446346812 s |
| observed save span | 0.721965201 s |
| checkpoint directory | 216,795,929 bytes |
| conservative 64-update max-length estimate | 2.096213 h |
| retained W 64-checkpoint upper bound | 1,789,254,848 bytes |
| retained DoRA 64-checkpoint upper bound, per arm | 13,874,918,528 bytes |

The extrapolated time is below the 12-hour per-arm operational ceiling, but it
is a qualification extrapolation rather than a production timing observation.
Retaining every checkpoint for three DoRA arms plus W can consume up to
43,414,010,432 bytes (43.4 GB decimal). Reconcile free storage before launch.
W and DoRA have materially different parameter counts, compute, functional step
sizes and payload sizes despite the matched data/loss/LR/update recipe; a null or
negative W result cannot establish representational impossibility. No natural
train256/dev128 model-quality evidence exists yet.

Preserved failed mechanics include an initial DoRA maximum-length OOM before
activation checkpointing, an ineffective checkpointing attempt while the model
remained in eval mode, a generic PEFT resume-key-layout mismatch, and the first
W Source-gate packaging attempt. The accepted fixes enable DoRA checkpointing
in train mode only after proving dropout is disabled, copy the exact sealed live
DoRA parameter layout on resume, and package/resolve the checkpoint-local Source
gate. Failed roots/receipts remain under `qualification/`; none changed the
scientific contract.

## Launch-ready commands (not executed)

Run these as four independent durable jobs, one at a time on GPUs 0/1 unless the
lead assigns other reconciled pairs. The output roots must not already exist.

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n ms torchrun --standalone --nproc_per_node=2 scripts/research/train_coco_gt_correction.py --bank-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json --arm R --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/R --max-updates 64 --global-batch-size 32 --seed 20260907
CUDA_VISIBLE_DEVICES=0,1 conda run -n ms torchrun --standalone --nproc_per_node=2 scripts/research/train_coco_gt_correction.py --bank-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json --arm B --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/B --max-updates 64 --global-batch-size 32 --seed 20260907
CUDA_VISIBLE_DEVICES=0,1 conda run -n ms torchrun --standalone --nproc_per_node=2 scripts/research/train_coco_gt_correction.py --bank-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json --arm M --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/M --max-updates 64 --global-batch-size 32 --seed 20260907
CUDA_VISIBLE_DEVICES=0,1 conda run -n ms torchrun --standalone --nproc_per_node=2 scripts/research/train_coco_gt_correction.py --bank-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json --arm W --output-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/W --max-updates 64 --global-batch-size 32 --seed 20260907
```

Stop here for lead boundary review. Production launch, terminal natural
evaluation, scientific interpretation and any change to the estimand, dose,
resource ceiling or stop rule remain separate decisions.
