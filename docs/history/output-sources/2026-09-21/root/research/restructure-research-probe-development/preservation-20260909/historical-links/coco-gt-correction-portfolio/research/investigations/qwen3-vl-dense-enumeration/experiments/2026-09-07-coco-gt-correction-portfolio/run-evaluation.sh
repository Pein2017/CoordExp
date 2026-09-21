#!/usr/bin/env bash
# Fixed eight-panel cold evaluation; serial, collision-safe, and never retries.
set -uo pipefail

cd /data/CoordExp/.worktrees/coco-gt-correction-portfolio || exit 1
root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1
source_root=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1
train=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl
dev=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/dev.jsonl
bank="$root/bank/manifest.json"
topology="$root/qualification/source-eval-topology.json"
evaluation="$root/evaluation"
launch="$root/launch-evaluation"
finalized=0

on_exit() {
  local status="$1"
  if [[ "$finalized" -eq 0 && -d "$launch" ]]; then
    printf 'COCO_EVALUATION_FAILED unexpected_exit=%s\n' "$status" | tee "$launch/terminal-marker.txt"
    date -u +%FT%TZ > "$launch/finished-utc.txt"
  fi
}
trap 'on_exit $?' EXIT

if [[ -e "$evaluation" || -e "$launch" ]]; then
  echo "COCO_EVALUATION_FAILED: refusing existing evaluation or launch-evaluation root"
  exit 1
fi

# The same cold loader used by production inference verifies every payload file.
conda run -n ms python -c "$(cat <<'PY'
from pathlib import Path
from scripts.research.coco_gt_correction_bank import file_sha256
from scripts.research.eval_coco_gt_correction import _load_cold

root = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1')
expected = {
    'R': '4efbec695067ec4f5a7c1de30add419046e0ac210067a27bd013aa9df71c033d',
    'B': '67684f1a842c3c454b45c7efd0c2706c9a7248cb571496353025471117da5d9d',
    'M': '4dcccaa341763841d008441bd22bd66ac6f90ba810c2ec38cfeab64b6e79b03b',
    'W': '1a068bf17c237bda42c4f73c592c81331dfa001e3fc4228383f42b089b54a0da',
}
bank = root / 'bank/manifest.json'
for arm, checkpoint_id in expected.items():
    manifest = _load_cold(root / arm / 'checkpoint-000064', arm=arm,
                          bank_manifest=bank, expected_completed_update=64)
    if manifest['checkpoint_id'] != checkpoint_id:
        raise ValueError(f'{arm} checkpoint identity differs')
if file_sha256(root / 'qualification/source-eval-topology.json') != '9e805279a7fde9574260ab4888120c452088158e1747ccf02953961246f43a21':
    raise ValueError('Source topology receipt digest differs')
PY
)"
status=$?
if [[ "$status" -ne 0 ]]; then
  echo "COCO_EVALUATION_FAILED: cold preflight exit=$status"
  exit "$status"
fi

mkdir "$evaluation" "$launch"
exec > >(tee -a "$launch/driver.log") 2>&1
printf '%s\n' "$$" > "$launch/driver.pid"
date -u +%FT%TZ > "$launch/started-utc.txt"
sha256sum \
  scripts/research/eval_coco_gt_correction.py \
  scripts/research/reduce_coco_gt_correction.py \
  research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-gt-correction-portfolio/run-evaluation.sh \
  > "$launch/code.sha256"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader > "$launch/gpu-preflight.csv"

run_panel() {
  local arm="$1" split="$2" input panel status
  if [[ "$split" = train ]]; then
    input="$train"
    panel=train256
  else
    input="$dev"
    panel=dev128
  fi
  local -a cmd=(
    conda run -n ms python scripts/research/eval_coco_gt_correction.py run
    --checkpoint "$root/$arm/checkpoint-000064"
    --arm "$arm"
    --bank-manifest "$bank"
    --expected-completed-update 64
    --input-jsonl "$input"
    --artifact-root "$evaluation/$arm/$split"
    --run-name "${arm}-${panel}-native-v1"
    --expected-active-ranks 8
  )
  date -u +%FT%TZ > "$launch/$arm-$split.started-utc.txt"
  printf 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' > "$launch/$arm-$split.command.txt"
  printf ' %q' "${cmd[@]}" >> "$launch/$arm-$split.command.txt"
  printf '\n' >> "$launch/$arm-$split.command.txt"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${cmd[@]}" > "$launch/$arm-$split.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$launch/$arm-$split.exit"
  date -u +%FT%TZ > "$launch/$arm-$split.finished-utc.txt"
  if [[ "$status" -eq 0 ]]; then
    echo "COCO_EVALUATION_PANEL_COMPLETED: $arm $split"
  else
    echo "COCO_EVALUATION_PANEL_FAILED: $arm $split exit=$status"
  fi
  return "$status"
}

for arm in R B M W; do
  for split in train dev; do
    run_panel "$arm" "$split"
    status=$?
    if [[ "$status" -ne 0 ]]; then
      printf 'COCO_EVALUATION_FAILED arm=%s split=%s exit=%s\n' "$arm" "$split" "$status" | tee "$launch/terminal-marker.txt"
      date -u +%FT%TZ > "$launch/finished-utc.txt"
      finalized=1
      exit "$status"
    fi
  done
done

reduce_cmd=(
  conda run -n ms python scripts/research/reduce_coco_gt_correction.py
  --train-input "$train" --dev-input "$dev"
  --train-run "Source=$source_root/qwen3-vl-2b-sft256-source-train256-natural-v1"
  --train-run "R=$evaluation/R/train/R-train256-native-v1"
  --train-run "B=$evaluation/B/train/B-train256-native-v1"
  --train-run "M=$evaluation/M/train/M-train256-native-v1"
  --train-run "W=$evaluation/W/train/W-train256-native-v1"
  --dev-run "Source=$source_root/qwen3-vl-2b-sft256-source-dev128-natural-v1"
  --dev-run "R=$evaluation/R/dev/R-dev128-native-v1"
  --dev-run "B=$evaluation/B/dev/B-dev128-native-v1"
  --dev-run "M=$evaluation/M/dev/M-dev128-native-v1"
  --dev-run "W=$evaluation/W/dev/W-dev128-native-v1"
  --bank-manifest "$bank"
  --source-topology-receipt "$topology"
  --expected-active-ranks 8 --expected-completed-update 64
  --evaluation-root "$evaluation/detection-reduction-v1"
  --out "$evaluation/portfolio-reduction-v1.json"
)
printf '%q ' "${reduce_cmd[@]}" > "$launch/reducer.command.txt"
printf '\n' >> "$launch/reducer.command.txt"
date -u +%FT%TZ > "$launch/reducer.started-utc.txt"
"${reduce_cmd[@]}" > "$launch/reducer.log" 2>&1
status=$?
printf '%s\n' "$status" > "$launch/reducer.exit"
date -u +%FT%TZ > "$launch/reducer.finished-utc.txt"
date -u +%FT%TZ > "$launch/finished-utc.txt"
if [[ "$status" -eq 0 ]]; then
  echo COCO_EVALUATION_AND_REDUCTION_COMPLETED | tee "$launch/terminal-marker.txt"
else
  printf 'COCO_EVALUATION_REDUCER_FAILED exit=%s\n' "$status" | tee "$launch/terminal-marker.txt"
fi
finalized=1
exit "$status"
