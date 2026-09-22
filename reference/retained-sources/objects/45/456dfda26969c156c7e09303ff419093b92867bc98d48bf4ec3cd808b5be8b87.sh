#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/stage01-210457-stop-logits-v1
cd /data/CoordExp/.worktrees/research-probes
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=30s 620s python "$ROOT/run.py" run --manifest "$ROOT/manifest.json" >"$ROOT/model.log" 2>&1
run_rc=$?
printf '%s\n' "$run_rc" > "$ROOT/model.exit"
verify_rc=99
if [ "$run_rc" -eq 0 ]; then
  python "$ROOT/run.py" verify --manifest "$ROOT/manifest.json" --output "$ROOT/verification.json" >"$ROOT/verify.log" 2>&1
  verify_rc=$?
fi
printf '%s\n' "$verify_rc" > "$ROOT/verify.exit"
final_rc=$run_rc
if [ "$run_rc" -eq 0 ] && [ "$verify_rc" -ne 0 ]; then final_rc=$verify_rc; fi
printf '%s\n' "$final_rc" > "$ROOT/controller.exit"
tmux wait-for -S tsc-210457-stop-logits-v1-done
exit "$final_rc"
