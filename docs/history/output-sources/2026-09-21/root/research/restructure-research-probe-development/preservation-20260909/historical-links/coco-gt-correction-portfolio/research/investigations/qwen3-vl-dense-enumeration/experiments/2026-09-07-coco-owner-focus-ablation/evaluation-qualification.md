# COCO owner-focus evaluation qualification

Date: 2026-09-07

Status: **evaluation mechanics QUALIFIED; production evaluation is not launched
by this package.** Scientific status remains
`MECHANICS_ONLY_NO_MODEL_QUALITY_CLAIM` until all fixed update-64 runs and the
paired reduction complete.

## Frozen holdout and identity boundary

The outcome-blind holdout is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/inputs-v1/holdout512.jsonl`,
SHA256 `62aff40429cfc10f0a640a6e86d298b560ab79d25ce0a776393757a12d15accd`.
It contains 512 COCO-2017-val images and 3,759 annotated owners, selected by
seeded SHA256 ordering with seed `20260908`. Its manifest SHA256 is
`c402610384a3589958f95bfb0a37f176b4d7ba2ba2fce6486079b22adeeebb25`.
The manifest records the bounded scan of 26 local input JSONLs / 2,259 prior
unique image IDs and excludes 20 val candidates found there. Source step 2444
was optimized on COCO-train; historical Source forward evaluation did include
COCO-val, so this is a fresh successor outcome panel, not a claim that Source
has never operationally seen these images.

The train diagnostic remains the sealed 256-image / 1,955-owner bank input,
SHA256 `05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5`.
The bank manifest file SHA256 is
`4486f173a98e3ba253798f8aa868a0712056b0480b9a4ee8fc081dfa90412ed2`;
bank ID is
`37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`.

## Cold entry and fixed reducer

`scripts/research/eval_coco_owner_focus.py` clones the admitted Source config
and changes only input, output, source-gate and candidate adapter locations.
It requires the owner-focus experiment, exact R/M/Rweak objective variant,
bank and Source identities, seed `20260908`, global image batch 32, expected
completed update and DoRA payload hashes before opening a model. Source accepts
no checkpoint. Candidates cold-load only their saved adapter; optimizer state
is not loaded.

Every arm uses native greedy HF generation, FP32 SDPA, repetition penalty 1.0,
cap 3,084 and actual per-device batch 4. The qualified production topology is
two active ranks, one GPU per rank. Runtime receipts bind the base, adapter,
special-token embedding delta, loaded/validated status, cold checkpoint and
the physical rank plan.

`scripts/research/reduce_coco_owner_focus.py` validates complete scored
artifacts before evaluation and refuses to overwrite its detection outputs.
At its nearest input boundary it now compares every raw output row to the
frozen JSONL: exact example/row IDs, image path and dimensions, and ordered GT
object ID, description and norm-1000 bbox. This prevents four internally
consistent arms from silently evaluating the wrong panel. Source additionally
must match the bank's base, adapter and embedding paths, validated/loaded
receipts, and the current on-disk Source adapter tensor SHA256. Candidate
payload identity remains checkpoint-bound.

The reducer retains all images and reports only the frozen paired contrasts
M-Rweak, M-R and M-Source at IoU50/60/80. It reports exact gained/lost owner
IDs, train owner cohorts, unmatched predictions, strict duplicate candidates,
invalid/drop/cap/length/token debt, and a 10,000-draw paired image bootstrap
with seed `20260908`. Bootstrap intervals are diagnostic; unmatched
predictions are not called hallucinations under partial COCO annotations.

## Production-shaped evidence

The batch sensitivity panel contains one cap/length case and three natural
`im_end` cases. Batch 1 and batch 4 produced identical raw decoded text,
tokens, stop reasons, parses, predictions and dropped predictions on all four
rows. Receipt SHA256:
`d2f2fd06bee0ed5784d9e39559ab17ea248aec037e22adf6e7c997f51aa62652`.

| Profile | Decode | Generated tokens/s | Peak allocated / reserved |
|---|---:|---:|---:|
| Source, one rank, batch 1 | 317.56s / 4 images | 13.14 | 11.84GB / 12.70GB |
| Source, one rank, batch 4 | 366.55s / 4 images | 11.38 | 20.74GB / 43.94GB |
| Rweak cold, one rank, batch 4 | 368.19s / 4 images | 11.30 | 20.74GB / 43.94GB |
| Rweak cold, two ranks x batch 4 | max-rank 366.32s / 8 images | 22.23 aggregate | 20.74GB / 43.94GB per-rank max |

Batch 4 was 13.36% slower than batch 1 on this deliberately long mixed panel,
but it is prediction-identical and is the frozen uniform profile requested for
all arms. A batch-8 rung was not run: batch 4 already increased reserved memory
by 31.24GB and reduced throughput, so it could not improve the active decision.

The real Rweak one-update checkpoint
`8b35005dc692a049d2a05915ee250ccbb9b37dbcdaa6cf6ba1ba996f6d384161`
cold-loaded and completed both the one-rank and custom two-rank paths. The
accepted one-rank run-manifest SHA256 is
`16c8a069a0d310864ff3330e60d13f00cc4cab9024b53f88e078f3d405dbf6cd`;
the accepted two-rank merged run-manifest SHA256 is
`62b737dfa31b3e05da8c6287ebf4817f34d0541e5ce7c68dc6e4b789ae8c7ec6`.
The latter retained all 8 rows, with 2 length stops and 6 natural EOS stops.
Incomplete earlier attempts remain in their original qualification directories
and are not accepted artifacts.

Using 8 images / 366.32s as a capacity estimate gives about 3.26 hours for
train256 and 6.51 hours for holdout512 per arm, or 9.77 hours per arm. Four
arms serially would be about 39 hours; four independent two-GPU arms in
parallel after training would be about 9.8 hours. These are output-length- and
content-sensitive extrapolations, not wall-time guarantees.

The historical M/train256 duplicate monitor found 348 strict candidates on
only 7 images: image `coco2017_train_000000013169` accounts for 289 (83.05%)
and the top five account for 99.14%. Receipt SHA256:
`13f489285936fb4d0af918965c97d164089d786fdd2561692386922b5311363f`.
This is a concentration warning motivating the fixed debt vector, not a new
successor result or confirmed physical-duplicate label.

## Falsification and fresh verification

The row-identity sensitivity constructs four arms with the same wrong owner ID:
the paired in-memory reducer alone accepts their self-consistency, while the
new frozen-input guard rejects every arm. The Source-composition sensitivity
first accepts a sealed temporary adapter, then rejects a foreign adapter path.
The public verifier also replays both guards on the real qualified Source and
two-rank Rweak artifacts; its result is `QUALIFIED`.
The same raw-input guard also passes the existing complete 256-image M/train
artifact against the sealed train JSONL, closing a small-panel-only concern.

```text
conda run -n ms pytest -q \
  tests/research/test_eval_coco_gt_correction.py \
  tests/research/test_reduce_coco_gt_correction.py \
  tests/research/test_build_coco_owner_focus_holdout.py \
  tests/research/test_eval_coco_owner_focus.py \
  tests/research/test_reduce_coco_owner_focus.py
# 12 passed

conda run -n ms python scripts/research/verify_coco_owner_focus_evaluation.py
# status=QUALIFIED; old_frozen_files_unchanged=true

conda run -n ms python -m py_compile \
  scripts/research/reduce_coco_owner_focus.py \
  scripts/research/verify_coco_owner_focus_evaluation.py \
  tests/research/test_reduce_coco_owner_focus.py
git diff --check
# PASS
```

Source hashes at final verification:

| Artifact | SHA256 |
|---|---|
| holdout builder | `a35793314dcd6db93ef217a6444c028f42e8535f1c982c8c636987393cdb7d47` |
| cold eval entry | `693da524eec702b1a32c0a40ba9ac95d89df42488a9b263955d8091ebcdd14aa` |
| paired reducer | `9ab06d765e3fbe93efbb626075e49a3790a79071b591af962debc12b01a73b48` |
| public verifier | `9d9780d9982a3f850f7f35fb5d48e5f963e1c6254999e72a9ce8d0a1a9d308c7` |
| holdout test | `61aa3f80c3d4c1df105b2ac76f6760fa096a7928690c648c289c6657c605d5da` |
| cold eval test | `4ae0cd9a1d45bb7ad8396dd12ee129912c1b88db78df5842847a142b61c96752` |
| reducer test | `fc9a1454fe0410dd6b7bd3f26958db69e71cb1de6bffadba44fb24d453dcf7e7` |

The verifier also confirms the four old portfolio eval/reducer source and test
hashes remain unchanged.

## Fixed production commands — not executed here

Run only after reconciling current GPU and output ownership. Each invocation
owns a new output root; a failed/incomplete root is retained and requires an
explicit new run name rather than deletion or overwrite. Source can run on its
free GPU pair while training continues. R/M/Rweak require accepted update-64
checkpoints and may then use three independent pairs concurrently.

```bash
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1
BANK=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1/bank/manifest.json
TRAIN=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl
HOLDOUT="$ROOT/inputs-v1/holdout512.jsonl"
HOLDOUT_MANIFEST="$ROOT/inputs-v1/manifest.json"
SOURCE_GATE="$ROOT/qualification/rweak-effective32-two-rank-mb2-v1/source-gate"

# Source baseline: one durable invocation on one two-GPU pair.
for SPLIT in train holdout; do
  if [ "$SPLIT" = train ]; then INPUT="$TRAIN"; PANEL=train256; else INPUT="$HOLDOUT"; PANEL=holdout512; fi
  CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ms \
    python scripts/research/eval_coco_owner_focus.py run \
    --arm Source --bank-manifest "$BANK" --expected-completed-update 64 \
    --input-jsonl "$INPUT" --artifact-root "$ROOT/evaluation/Source/$SPLIT" \
    --run-name "Source-${PANEL}-native-v1" --expected-active-ranks 2 \
    --source-gate-root "$SOURCE_GATE" --batch-size 4
done

# Run this block once per accepted ARM=R, M, Rweak. Assign a distinct GPU pair
# to each block if running the three arms concurrently.
ARM=R
DEVICES=0,1
for SPLIT in train holdout; do
  if [ "$SPLIT" = train ]; then INPUT="$TRAIN"; PANEL=train256; else INPUT="$HOLDOUT"; PANEL=holdout512; fi
  CUDA_VISIBLE_DEVICES="$DEVICES" conda run --no-capture-output -n ms \
    python scripts/research/eval_coco_owner_focus.py run \
    --checkpoint "$ROOT/$ARM/checkpoint-000064" --arm "$ARM" \
    --bank-manifest "$BANK" --expected-completed-update 64 \
    --input-jsonl "$INPUT" --artifact-root "$ROOT/evaluation/$ARM/$SPLIT" \
    --run-name "${ARM}-${PANEL}-native-v1" --expected-active-ranks 2 \
    --source-gate-root "$SOURCE_GATE" --batch-size 4
done

# After all eight Source/R/M/Rweak x train/holdout runs complete:
conda run --no-capture-output -n ms \
  python scripts/research/reduce_coco_owner_focus.py reduce \
  --train-input "$TRAIN" --holdout-input "$HOLDOUT" \
  --holdout-manifest "$HOLDOUT_MANIFEST" --bank-manifest "$BANK" \
  --train-run "Source=$ROOT/evaluation/Source/train/Source-train256-native-v1" \
  --train-run "R=$ROOT/evaluation/R/train/R-train256-native-v1" \
  --train-run "M=$ROOT/evaluation/M/train/M-train256-native-v1" \
  --train-run "Rweak=$ROOT/evaluation/Rweak/train/Rweak-train256-native-v1" \
  --holdout-run "Source=$ROOT/evaluation/Source/holdout/Source-holdout512-native-v1" \
  --holdout-run "R=$ROOT/evaluation/R/holdout/R-holdout512-native-v1" \
  --holdout-run "M=$ROOT/evaluation/M/holdout/M-holdout512-native-v1" \
  --holdout-run "Rweak=$ROOT/evaluation/Rweak/holdout/Rweak-holdout512-native-v1" \
  --evaluation-root "$ROOT/evaluation/detection-reduction-v1" \
  --expected-active-ranks 2 --expected-completed-update 64 \
  --out "$ROOT/evaluation/owner-focus-reduction-v1.json"
```

For concurrent candidate execution, repeat the candidate block in separate
durable invocations with `ARM=R DEVICES=0,1`, `ARM=M DEVICES=4,5`, and
`ARM=Rweak DEVICES=6,7` after the matching final checkpoints exist. Device
placement is operational only; topology, batch and scientific identity remain
fixed.

STOP at qualified evaluation/reduction mechanics. No production decode,
scientific result, retry, extra batch rung, seed, sweep, model promotion,
commit, push or archive was performed here.
