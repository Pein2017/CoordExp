#!/usr/bin/env bash
set -uo pipefail

worktree=/data/CoordExp/.worktrees/permutation-bundle-coordinate-noise-pilot
attestation_session=perm_bundle_prod_attestation_w4_20260728
attestation_log="$worktree/outputs/attestations/coordexp_swift/prod_same_image_bundle_single_arm_probe_w4.log"
attestation_path="$worktree/.cache/coordexp_swift/presentation_attestations/prod_same_image_bundle_single_arm_probe.attestation"
config_path="$worktree/configs/coordexp_swift/permutation_bundle_coordinate_noise/prod/same_image_bundle_single_arm_probe.yaml"
long_log="$worktree/outputs/prod/coordexp_swift/permutation_bundle_coordinate_noise/receipt_backed_logs/same_image_long.log"

mkdir -p "$(dirname "$long_log")"

while true; do
    if grep -qx 'ATTESTATION_EXIT_STATUS=0' "$attestation_log" 2>/dev/null; then
        break
    fi
    if grep -q '^ATTESTATION_EXIT_STATUS=' "$attestation_log" 2>/dev/null; then
        printf 'WATCHER_STATUS=attestation_failed\n' | tee -a "$long_log"
        exit 20
    fi
    if ! tmux has-session -t "$attestation_session" 2>/dev/null; then
        printf 'WATCHER_STATUS=attestation_session_missing\n' | tee -a "$long_log"
        exit 21
    fi
    sleep 60
done

if [[ ! -d "$attestation_path" ]]; then
    printf 'WATCHER_STATUS=attestation_artifact_missing\n' | tee -a "$long_log"
    exit 22
fi

cd "$worktree"
if ! conda run --no-capture-output -n ms python - "$config_path" "$attestation_path" >>"$long_log" 2>&1 <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config.loader import load_train_config
from src.training.presentation_admission import (
    admit_presentation_runtime_from_attestation,
)

config_path = Path(sys.argv[1])
attestation_path = Path(sys.argv[2])
resolved = load_train_config(config_path)
admitted = admit_presentation_runtime_from_attestation(
    resolved.config,
    attestation_path=attestation_path,
    resolved_config_fingerprint=resolved.fingerprint,
    live_world_size=8,
    repo_root=Path.cwd(),
)
print(
    json.dumps(
        {
            "watcher_receipt_validation": "passed",
            "cache_fingerprint": admitted.fingerprint,
            "planned_step_count": admitted.execution_stream.planned_step_count,
            "world_size": 8,
        },
        sort_keys=True,
    )
)
PY
then
    printf 'WATCHER_STATUS=receipt_validation_failed\n' | tee -a "$long_log"
    exit 23
fi

gpu_pids="$({ nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits || true; } | awk 'NF { print $1 }' | sort -u)"
if [[ -n "$gpu_pids" ]]; then
    printf 'WATCHER_STATUS=gpus_not_clear pids=%s\n' "$gpu_pids" | tee -a "$long_log"
    exit 24
fi

printf 'WATCHER_STATUS=launching_receipt_backed_w8\n' | tee -a "$long_log"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
conda run --no-capture-output -n ms accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port 29514 \
    -m src.train \
    --config "$config_path" >>"$long_log" 2>&1
train_status=$?
printf 'TRAIN_EXIT_STATUS=%s\n' "$train_status" | tee -a "$long_log"
exit "$train_status"
