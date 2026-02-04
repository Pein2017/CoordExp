#!/usr/bin/env bash
#
# Unified public dataset preparation runner for CoordExp.
# See: openspec/changes/2026-01-22-add-unified-public-data-pipeline

set -euo pipefail

die() {
  echo "[error] $*" >&2
  exit 1
}

warn() {
  echo "[warn] $*" >&2
}

banner() {
  echo
  echo "============================================================"
  echo "$*"
  echo "============================================================"
}

run_cmd() {
  echo "+ $*" >&2
  "$@"
}

require_repo_root() {
  # Repo-root anchored by design: ensures PYTHONPATH=. works and paths stay consistent.
  if [[ ! -d "public_data" || ! -d "src" || ! -f "public_data/run.sh" ]]; then
    die "Run this from the CoordExp repo root (directory containing public_data/ and src/). Example: cd ."
  fi
}

usage() {
  cat <<'EOF'
Unified public dataset pipeline runner (repo-root anchored).

Usage:
  ./public_data/run.sh <dataset> <command> [runner-flags] [-- <passthrough-args>]

Commands:
  download   Dataset-specific download into public_data/<dataset>/raw/
  convert    Dataset-specific conversion into public_data/<dataset>/raw/{train,val}.jsonl
  rescale    Shared smart-resize into public_data/<dataset>/<preset>/
  coord      Shared coord-token conversion inside public_data/<dataset>/<preset>/
  validate   Validate raw and/or preset artifacts; also sanity-check chat template on *.coord.jsonl
  all        download -> convert -> rescale -> coord -> validate
  help       Print this message and exit 0

Runner flags:
  --preset <name>          Preset dir name under public_data/<dataset>/ (used by rescale|coord|validate|all)
  --skip-image-check       Skip image existence checks during validation
  --raw-only               For validate: validate only raw artifacts (no preset required)
  --preset-only            For validate: validate only preset artifacts

Passthrough args:
  Everything after `--` is forwarded verbatim to the underlying implementation:
    - download/convert: dataset plugin script (public_data/datasets/<dataset>.sh)
    - rescale: public_data/scripts/rescale_jsonl.py
    - coord:   public_data/scripts/convert_to_coord_tokens.py
  For `all`, passthrough args are forwarded ONLY to dataset plugin steps (download/convert).

Examples:
  ./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0
  ./public_data/run.sh lvis all --preset rescale_32_768_bbox
  ./public_data/run.sh lvis all --preset rescale_32_768_poly_20 -- --use-polygon
EOF
}

run_py() {
  echo "+ PYTHONPATH=. python $*" >&2
  PYTHONPATH=. python "$@"
}

run_py_best_effort() {
  set +e
  echo "+ PYTHONPATH=. python $*" >&2
  PYTHONPATH=. python "$@"
  local rc=$?
  set -e
  return $rc
}

choose_inspect_model() {
  # Prefer smaller processors when available; fall back to 8B; otherwise skip.
  local candidates=(
    "${REPO_ROOT}/model_cache/Qwen3-VL-4B-Instruct-coordexp"
    "${REPO_ROOT}/model_cache/Qwen3-VL-8B-Instruct-coordexp"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -d "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "Missing required file: ${path}"
}

plugin_path() {
  echo "${REPO_ROOT}/public_data/datasets/${DATASET}.sh"
}

require_plugin_file() {
  local plugin
  plugin="$(plugin_path)"
  if [[ ! -f "${plugin}" ]]; then
    die "Unknown dataset '${DATASET}': missing plugin file public_data/datasets/${DATASET}.sh"
  fi
}

plugin_default_preset() {
  local plugin out
  plugin="$(plugin_path)"
  require_plugin_file

  set +e
  out="$(bash "${plugin}" default-preset 2>/dev/null)"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    return 1
  fi
  out="$(echo "${out}" | tr -d '\r' | head -n 1 | xargs || true)"
  [[ -n "${out}" ]] || return 1
  echo "${out}"
}

run_plugin() {
  local subcmd="$1"
  shift
  local plugin
  plugin="$(plugin_path)"
  require_plugin_file
  run_cmd bash "${plugin}" "${subcmd}" "$@"
}

set_paths_for_preset() {
  # Requires PRESET to already be set (may be empty).
  PRESET_DIR="${DATASET_DIR}/${PRESET}"
  PRESET_IMAGE_DIR="${PRESET_DIR}/images"
  PRESET_TRAIN_JSONL="${PRESET_DIR}/train.jsonl"
  PRESET_VAL_JSONL="${PRESET_DIR}/val.jsonl"
  PRESET_TRAIN_COORD_JSONL="${PRESET_DIR}/train.coord.jsonl"
  PRESET_VAL_COORD_JSONL="${PRESET_DIR}/val.coord.jsonl"
}

require_repo_root

DATASET="${1:-}"
COMMAND="${2:-help}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true

if [[ -z "${DATASET}" || -z "${COMMAND}" ]]; then
  usage
  exit 1
fi

# Runner flags (parsed before --). Only this small surface area is supported.
PRESET=""
SKIP_IMAGE_CHECK="false"
RAW_ONLY="false"
PRESET_ONLY="false"
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      PASSTHROUGH_ARGS=("$@")
      break
      ;;
    --preset)
      shift
      [[ $# -gt 0 ]] || die "--preset requires a value"
      PRESET="$1"
      shift
      ;;
    --skip-image-check)
      SKIP_IMAGE_CHECK="true"
      shift
      ;;
    --raw-only)
      RAW_ONLY="true"
      shift
      ;;
    --preset-only)
      PRESET_ONLY="true"
      shift
      ;;
    -h|--help)
      COMMAND="help"
      shift
      ;;
    *)
      die "Unknown runner flag '${1}'. Use '--' to pass args to dataset/plugin scripts."
      ;;
  esac
done

REPO_ROOT="$(pwd)"
DATASET_DIR="${REPO_ROOT}/public_data/${DATASET}"
RAW_DIR="${DATASET_DIR}/raw"
RAW_IMAGE_DIR="${RAW_DIR}/images"
RAW_TRAIN_JSONL="${RAW_DIR}/train.jsonl"
RAW_VAL_JSONL="${RAW_DIR}/val.jsonl"
set_paths_for_preset

case "${COMMAND}" in
  help)
    usage
    exit 0
    ;;
  download)
    require_plugin_file
    banner "[${DATASET}] download -> ${RAW_DIR}"
    run_cmd mkdir -p "${RAW_DIR}"
    run_plugin download \
      --repo-root "${REPO_ROOT}" \
      --dataset "${DATASET}" \
      --dataset-dir "${DATASET_DIR}" \
      --raw-dir "${RAW_DIR}" \
      --raw-image-dir "${RAW_IMAGE_DIR}" \
      --raw-train-jsonl "${RAW_TRAIN_JSONL}" \
      --raw-val-jsonl "${RAW_VAL_JSONL}" \
      -- "${PASSTHROUGH_ARGS[@]}"
    ;;
  convert)
    require_plugin_file
    banner "[${DATASET}] convert -> ${RAW_TRAIN_JSONL}"
    run_cmd mkdir -p "${RAW_DIR}"
    run_plugin convert \
      --repo-root "${REPO_ROOT}" \
      --dataset "${DATASET}" \
      --dataset-dir "${DATASET_DIR}" \
      --raw-dir "${RAW_DIR}" \
      --raw-image-dir "${RAW_IMAGE_DIR}" \
      --raw-train-jsonl "${RAW_TRAIN_JSONL}" \
      --raw-val-jsonl "${RAW_VAL_JSONL}" \
      -- "${PASSTHROUGH_ARGS[@]}"
    ;;
  rescale)
    [[ -n "${PRESET}" ]] || die "rescale requires --preset <name>"
    set_paths_for_preset
    banner "[${DATASET}] rescale -> ${PRESET_DIR}"
    require_file "${RAW_TRAIN_JSONL}"
    run_cmd mkdir -p "${PRESET_DIR}"
    run_py public_data/scripts/rescale_jsonl.py \
      "${PASSTHROUGH_ARGS[@]}" \
      --input-jsonl "${RAW_TRAIN_JSONL}" \
      --output-jsonl "${PRESET_TRAIN_JSONL}" \
      --output-images "${PRESET_DIR}" \
      --relative-images
    if [[ -f "${RAW_VAL_JSONL}" ]]; then
      run_py public_data/scripts/rescale_jsonl.py \
        "${PASSTHROUGH_ARGS[@]}" \
        --input-jsonl "${RAW_VAL_JSONL}" \
        --output-jsonl "${PRESET_VAL_JSONL}" \
        --output-images "${PRESET_DIR}" \
        --relative-images
    fi
    ;;
  coord)
    [[ -n "${PRESET}" ]] || die "coord requires --preset <name>"
    set_paths_for_preset
    banner "[${DATASET}] coord -> ${PRESET_DIR}"
    require_file "${PRESET_TRAIN_JSONL}"
    run_py public_data/scripts/convert_to_coord_tokens.py \
      "${PASSTHROUGH_ARGS[@]}" \
      --input "${PRESET_TRAIN_JSONL}" \
      --output-tokens "${PRESET_TRAIN_COORD_JSONL}" \
      --keys bbox_2d poly
    if [[ -f "${PRESET_VAL_JSONL}" ]]; then
      run_py public_data/scripts/convert_to_coord_tokens.py \
        "${PASSTHROUGH_ARGS[@]}" \
        --input "${PRESET_VAL_JSONL}" \
        --output-tokens "${PRESET_VAL_COORD_JSONL}" \
        --keys bbox_2d poly
    fi
    ;;
  validate)
    if [[ "${RAW_ONLY}" == "true" && "${PRESET_ONLY}" == "true" ]]; then
      die "--raw-only and --preset-only cannot be used together"
    fi

    # Default: validate both raw and preset.
    DO_RAW="true"
    DO_PRESET="true"
    if [[ "${RAW_ONLY}" == "true" ]]; then
      DO_PRESET="false"
    elif [[ "${PRESET_ONLY}" == "true" ]]; then
      DO_RAW="false"
    fi

    # Preset resolution: only needed if validating preset outputs.
    if [[ "${DO_PRESET}" == "true" && -z "${PRESET}" ]]; then
      if PRESET="$(plugin_default_preset)"; then
        :
      else
        die "validate requires --preset <name> (or a dataset plugin default preset) unless --raw-only is set"
      fi
    fi
    set_paths_for_preset

    banner "[${DATASET}] validate"
    VALIDATE_ARGS=()
    if [[ "${SKIP_IMAGE_CHECK}" == "true" ]]; then
      VALIDATE_ARGS+=(--skip-image-check)
    fi

    if [[ "${DO_RAW}" == "true" ]]; then
      require_file "${RAW_TRAIN_JSONL}"
      run_py public_data/scripts/validate_jsonl.py "${RAW_TRAIN_JSONL}" "${VALIDATE_ARGS[@]}"
      if [[ -f "${RAW_VAL_JSONL}" ]]; then
        run_py public_data/scripts/validate_jsonl.py "${RAW_VAL_JSONL}" "${VALIDATE_ARGS[@]}"
      fi
    fi

    if [[ "${DO_PRESET}" == "true" ]]; then
      require_file "${PRESET_TRAIN_JSONL}"
      run_py public_data/scripts/validate_jsonl.py "${PRESET_TRAIN_JSONL}" "${VALIDATE_ARGS[@]}"
      if [[ -f "${PRESET_VAL_JSONL}" ]]; then
        run_py public_data/scripts/validate_jsonl.py "${PRESET_VAL_JSONL}" "${VALIDATE_ARGS[@]}"
      fi

      require_file "${PRESET_TRAIN_COORD_JSONL}"
      run_py public_data/scripts/validate_jsonl.py "${PRESET_TRAIN_COORD_JSONL}" "${VALIDATE_ARGS[@]}"
      if [[ -f "${PRESET_VAL_COORD_JSONL}" ]]; then
        run_py public_data/scripts/validate_jsonl.py "${PRESET_VAL_COORD_JSONL}" "${VALIDATE_ARGS[@]}"
      fi

      # Prompt/template sanity check on coord-token JSONL.
      if INSPECT_MODEL="$(choose_inspect_model)"; then
        if ! run_py_best_effort scripts/inspect_chat_template.py \
          --jsonl "${PRESET_TRAIN_COORD_JSONL}" \
          --index 0 \
          --model "${INSPECT_MODEL}"; then
          warn "inspect_chat_template.py failed (model/deps missing?); skipping template check."
        fi
      else
        warn "Skipping inspect_chat_template.py (no cached model found under model_cache/)."
      fi
    fi
    ;;
  all)
    require_plugin_file

    # Preset resolution: --preset overrides plugin default; error if neither.
    if [[ -z "${PRESET}" ]]; then
      if PRESET="$(plugin_default_preset)"; then
        :
      else
        die "all requires --preset <name> (or a dataset plugin default preset)"
      fi
    fi
    set_paths_for_preset

    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
      echo "[note] Args after '--' are forwarded only to dataset plugin steps (download/convert) for 'all'." >&2
      echo "[note] To tune shared preprocessing options, run 'rescale'/'coord' as separate commands." >&2
    fi

    banner "[${DATASET}] all (preset: ${PRESET})"
    run_cmd mkdir -p "${RAW_DIR}"
    banner "[${DATASET}] stage: download"
    run_plugin download \
      --repo-root "${REPO_ROOT}" \
      --dataset "${DATASET}" \
      --dataset-dir "${DATASET_DIR}" \
      --raw-dir "${RAW_DIR}" \
      --raw-image-dir "${RAW_IMAGE_DIR}" \
      --raw-train-jsonl "${RAW_TRAIN_JSONL}" \
      --raw-val-jsonl "${RAW_VAL_JSONL}" \
      -- "${PASSTHROUGH_ARGS[@]}"
    banner "[${DATASET}] stage: convert"
    run_plugin convert \
      --repo-root "${REPO_ROOT}" \
      --dataset "${DATASET}" \
      --dataset-dir "${DATASET_DIR}" \
      --raw-dir "${RAW_DIR}" \
      --raw-image-dir "${RAW_IMAGE_DIR}" \
      --raw-train-jsonl "${RAW_TRAIN_JSONL}" \
      --raw-val-jsonl "${RAW_VAL_JSONL}" \
      -- "${PASSTHROUGH_ARGS[@]}"
    banner "[${DATASET}] stage: rescale"
    # Shared steps run with runner defaults (no passthrough args here).
    run_py public_data/scripts/rescale_jsonl.py \
      --input-jsonl "${RAW_TRAIN_JSONL}" \
      --output-jsonl "${PRESET_TRAIN_JSONL}" \
      --output-images "${PRESET_DIR}" \
      --relative-images
    if [[ -f "${RAW_VAL_JSONL}" ]]; then
      run_py public_data/scripts/rescale_jsonl.py \
        --input-jsonl "${RAW_VAL_JSONL}" \
        --output-jsonl "${PRESET_VAL_JSONL}" \
        --output-images "${PRESET_DIR}" \
        --relative-images
    fi
    banner "[${DATASET}] stage: coord"
    run_py public_data/scripts/convert_to_coord_tokens.py \
      --input "${PRESET_TRAIN_JSONL}" \
      --output-tokens "${PRESET_TRAIN_COORD_JSONL}" \
      --keys bbox_2d poly
    if [[ -f "${PRESET_VAL_JSONL}" ]]; then
      run_py public_data/scripts/convert_to_coord_tokens.py \
        --input "${PRESET_VAL_JSONL}" \
        --output-tokens "${PRESET_VAL_COORD_JSONL}" \
        --keys bbox_2d poly
    fi
    banner "[${DATASET}] stage: validate"
    # validate defaults to raw+preset; it will also run inspect_chat_template.py.
    run_cmd bash "${0}" "${DATASET}" validate --preset "${PRESET}" \
      $([[ "${SKIP_IMAGE_CHECK}" == "true" ]] && echo "--skip-image-check" || true)
    ;;
  *)
    echo "[error] Unknown command '${COMMAND}'." >&2
    usage >&2
    exit 1
    ;;
esac
