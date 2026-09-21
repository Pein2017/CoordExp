#!/usr/bin/env bash
set -euo pipefail
cd /data/CoordExp/.worktrees/coordexp-infras
packet=/data/CoordExp/outputs/infra_base/untie-axis-20260918-restart1
config="$packet/production.yaml"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$packet/launcher.pid"
stage=preflight
finish() {
  rc=$?
  trap - EXIT
  if [ "$rc" -eq 0 ]; then state=PIPELINE_COMPLETED; else state=PIPELINE_FAILED; fi
  printf '%s stage=%s exit_code=%s time=%s\n' "$state" "$stage" "$rc" "$(date -u +%FT%TZ)" | tee -a "$packet/terminal.log"
  printf '%s\n' "$rc" > "$packet/exit-code"
  exit "$rc"
}
trap finish EXIT
verify_source() {
  python - <<'PY'
import hashlib, json
from pathlib import Path
from src.config.loader import load_train_config
packet=Path('/data/CoordExp/outputs/infra_base/untie-axis-20260918-restart1')
identity=json.loads((packet/'source-identity.json').read_text())
changed=[p for p,h in identity['files'].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed, f'Execution sources changed after acceptance: {changed}'
assert load_train_config(packet/'production.yaml').fingerprint==identity['resolved_config_sha256']
assert json.loads(Path('/data/CoordExp/outputs/infra_base/untie-20260918/axis-acceptance.json').read_text())['status']=='passed'
assert json.loads((packet/'packing-fix-acceptance.json').read_text())['status']=='passed'
PY
}
verify_source
stage=cache_prepare
printf 'CACHE_PREPARATION_STARTED time=%s\n' "$(date -u +%FT%TZ)" | tee -a "$packet/terminal.log"
python -m src.prepare_train_cache --config "$config" --receipt "$packet/cache-preparation.json" 2>&1 | tee "$packet/cache-preparation.log"
stage=cache_admission
python -m src.prepare_train_cache --config "$config" --require-all-hit --receipt "$packet/cache-admission.json" > "$packet/cache-admission.log" 2>&1
verify_source
stage=training
printf 'TRAINING_STARTED time=%s\n' "$(date -u +%FT%TZ)" | tee -a "$packet/terminal.log"
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 --local-addr=127.0.0.1 \
  -m src.train --config "$config" 2>&1 | tee "$packet/training.log"
