#!/usr/bin/env bash
#
# Unified public dataset preparation runner for CoordExp.
# See: openspec/changes/refactor-public-data-pipeline-factory

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
  --conda-env <name>       Conda env name for python steps (default: ms)
  --skip-image-check       Skip image existence checks during validation
  --raw-only               For validate: validate only raw artifacts (no preset required)
  --preset-only            For validate: validate only preset artifacts

Passthrough args:
  Everything after `--` is forwarded to the underlying implementation:
    - download/convert: adapter ingestion hook (which invokes dataset plugin contract)
    - rescale/coord/validate: unified pipeline factory (public_data/scripts/run_pipeline_factory.py)
  The pipeline factory supports a curated subset of flags and warns on unsupported args.
  For `all`, passthrough args are forwarded ONLY to dataset plugin steps (download/convert).

Examples:
  ./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0
  ./public_data/run.sh lvis all --preset rescale_32_768_bbox
  ./public_data/run.sh lvis all --preset rescale_32_768_poly_20 -- --use-polygon
EOF
}

_resolve_conda_exe() {
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    echo "${CONDA_EXE}"
    return 0
  fi
  die "Missing 'conda' on PATH (and CONDA_EXE is unset); required for python steps."
}

run_py() {
  local conda_exe
  conda_exe="$(_resolve_conda_exe)"
  echo "+ PYTHONPATH=. ${conda_exe} run -n ${CONDA_ENV} python $*" >&2
  PYTHONPATH=. "${conda_exe}" run -n "${CONDA_ENV}" python "$@"
}

run_py_best_effort() {
  set +e
  local conda_exe
  conda_exe="$(_resolve_conda_exe)"
  echo "+ PYTHONPATH=. ${conda_exe} run -n ${CONDA_ENV} python $*" >&2
  PYTHONPATH=. "${conda_exe}" run -n "${CONDA_ENV}" python "$@"
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

set_paths_for_preset() {
  # Requires PRESET to already be set (may be empty).
  PRESET_DIR="${DATASET_DIR}/${PRESET}"
  PRESET_IMAGE_DIR="${PRESET_DIR}/images"
  PRESET_TRAIN_RAW_JSONL="${PRESET_DIR}/train.raw.jsonl"
  PRESET_VAL_RAW_JSONL="${PRESET_DIR}/val.raw.jsonl"
  PRESET_TRAIN_NORM_JSONL="${PRESET_DIR}/train.norm.jsonl"
  PRESET_VAL_NORM_JSONL="${PRESET_DIR}/val.norm.jsonl"
  PRESET_TRAIN_JSONL="${PRESET_DIR}/train.jsonl"
  PRESET_VAL_JSONL="${PRESET_DIR}/val.jsonl"
  PRESET_TRAIN_COORD_JSONL="${PRESET_DIR}/train.coord.jsonl"
  PRESET_VAL_COORD_JSONL="${PRESET_DIR}/val.coord.jsonl"
}

resolve_effective_preset_name() {
  local base_preset="$1"
  local max_objects="${2:-}"
  if [[ -z "${max_objects}" ]]; then
    echo "${base_preset}"
    return 0
  fi
  [[ "${max_objects}" =~ ^[0-9]+$ ]] || die "PUBLIC_DATA_MAX_OBJECTS must be an integer if set"
  run_py public_data/scripts/resolve_effective_preset.py \
    --dataset-dir "${DATASET_DIR}" \
    --base-preset "${base_preset}" \
    --max-objects "${max_objects}"
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
CONDA_ENV="ms"
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
    --conda-env)
      shift
      [[ $# -gt 0 ]] || die "--conda-env requires a value"
      CONDA_ENV="$1"
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

PIPELINE_MAX_OBJECTS="${PUBLIC_DATA_MAX_OBJECTS:-}"

case "${COMMAND}" in
  help)
    usage
    exit 0
    ;;
  download)
    banner "[${DATASET}] download -> ${RAW_DIR}"
    run_cmd mkdir -p "${RAW_DIR}"
    PIPELINE_ARGS=(
      --mode download
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
    )
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
      run_py public_data/scripts/run_pipeline_factory.py \
        "${PIPELINE_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py public_data/scripts/run_pipeline_factory.py "${PIPELINE_ARGS[@]}"
    fi
    ;;
  convert)
    banner "[${DATASET}] convert -> ${RAW_TRAIN_JSONL}"
    run_cmd mkdir -p "${RAW_DIR}"
    PIPELINE_ARGS=(
      --mode convert
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
    )
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
      run_py public_data/scripts/run_pipeline_factory.py \
        "${PIPELINE_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py public_data/scripts/run_pipeline_factory.py "${PIPELINE_ARGS[@]}"
    fi
    ;;
  rescale)
    [[ -n "${PRESET}" ]] || die "rescale requires --preset <name>"
    PRESET="$(resolve_effective_preset_name "${PRESET}" "${PIPELINE_MAX_OBJECTS}")"
    set_paths_for_preset
    banner "[${DATASET}] rescale -> ${PRESET_DIR}"
    require_file "${RAW_TRAIN_JSONL}"
    PIPELINE_ARGS=(
      --mode rescale
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
      --preset "${PRESET}"
    )
    if [[ -n "${PIPELINE_MAX_OBJECTS}" ]]; then
      PIPELINE_ARGS+=(--max-objects "${PIPELINE_MAX_OBJECTS}")
    fi
    run_py public_data/scripts/run_pipeline_factory.py \
      "${PIPELINE_ARGS[@]}" \
      "${PASSTHROUGH_ARGS[@]}"
    set_paths_for_preset
    ;;
  coord)
    [[ -n "${PRESET}" ]] || die "coord requires --preset <name>"
    PRESET="$(resolve_effective_preset_name "${PRESET}" "${PIPELINE_MAX_OBJECTS}")"
    set_paths_for_preset
    banner "[${DATASET}] coord -> ${PRESET_DIR}"
    PIPELINE_ARGS=(
      --mode coord
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
      --preset "${PRESET}"
    )
    if [[ -n "${PIPELINE_MAX_OBJECTS}" ]]; then
      PIPELINE_ARGS+=(--max-objects "${PIPELINE_MAX_OBJECTS}")
    fi
    run_py public_data/scripts/run_pipeline_factory.py \
      "${PIPELINE_ARGS[@]}" \
      "${PASSTHROUGH_ARGS[@]}"
    set_paths_for_preset
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
    if [[ "${DO_PRESET}" == "true" ]]; then
      PRESET="$(resolve_effective_preset_name "${PRESET}" "${PIPELINE_MAX_OBJECTS}")"
    fi
    set_paths_for_preset

    banner "[${DATASET}] validate"
    PIPELINE_ARGS=(
      --mode validate
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
      --preset "${PRESET}"
    )
    if [[ -n "${PIPELINE_MAX_OBJECTS}" ]]; then
      PIPELINE_ARGS+=(--max-objects "${PIPELINE_MAX_OBJECTS}")
    fi
    if [[ "${DO_RAW}" == "true" ]]; then
      PIPELINE_ARGS+=(--validate-raw)
    fi
    if [[ "${DO_PRESET}" == "true" ]]; then
      PIPELINE_ARGS+=(--validate-preset)
    fi
    if [[ "${SKIP_IMAGE_CHECK}" == "true" ]]; then
      PIPELINE_ARGS+=(--skip-image-check)
    fi
    run_py public_data/scripts/run_pipeline_factory.py "${PIPELINE_ARGS[@]}"

    if [[ "${DO_PRESET}" == "true" ]]; then
      # Prompt/template sanity check on coord-token JSONL.
      if INSPECT_MODEL="$(choose_inspect_model)"; then
        if ! run_py_best_effort scripts/tools/inspect_chat_template.py \
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
    # Preset resolution: --preset overrides plugin default; error if neither.
    if [[ -z "${PRESET}" ]]; then
      if PRESET="$(plugin_default_preset)"; then
        :
      else
        die "all requires --preset <name> (or a dataset plugin default preset)"
      fi
    fi
    PRESET="$(resolve_effective_preset_name "${PRESET}" "${PIPELINE_MAX_OBJECTS}")"
    set_paths_for_preset

    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
      echo "[note] Args after '--' are forwarded only to dataset plugin steps (download/convert) for 'all'." >&2
      echo "[note] To tune shared preprocessing options, run 'rescale'/'coord' as separate commands." >&2
    fi

    banner "[${DATASET}] all (preset: ${PRESET})"
    run_cmd mkdir -p "${RAW_DIR}"
    banner "[${DATASET}] stage: download"
    PIPELINE_INGEST_ARGS=(
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
    )
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
      run_py public_data/scripts/run_pipeline_factory.py \
        --mode download \
        "${PIPELINE_INGEST_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py public_data/scripts/run_pipeline_factory.py \
        --mode download \
        "${PIPELINE_INGEST_ARGS[@]}"
    fi
    banner "[${DATASET}] stage: convert"
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
      run_py public_data/scripts/run_pipeline_factory.py \
        --mode convert \
        "${PIPELINE_INGEST_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py public_data/scripts/run_pipeline_factory.py \
        --mode convert \
        "${PIPELINE_INGEST_ARGS[@]}"
    fi
    banner "[${DATASET}] stage: shared-pipeline"
    PIPELINE_ARGS=(
      --mode full
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
      --preset "${PRESET}"
      --run-validation-stage
    )
    if [[ -n "${PIPELINE_MAX_OBJECTS}" ]]; then
      PIPELINE_ARGS+=(--max-objects "${PIPELINE_MAX_OBJECTS}")
    fi
    if [[ "${SKIP_IMAGE_CHECK}" == "true" ]]; then
      PIPELINE_ARGS+=(--skip-image-check)
    fi
    run_py public_data/scripts/run_pipeline_factory.py "${PIPELINE_ARGS[@]}"
    set_paths_for_preset

    # Prompt/template sanity check on coord-token JSONL.
    if INSPECT_MODEL="$(choose_inspect_model)"; then
      if ! run_py_best_effort scripts/tools/inspect_chat_template.py \
        --jsonl "${PRESET_TRAIN_COORD_JSONL}" \
        --index 0 \
        --model "${INSPECT_MODEL}"; then
        warn "inspect_chat_template.py failed (model/deps missing?); skipping template check."
      fi
    else
      warn "Skipping inspect_chat_template.py (no cached model found under model_cache/)."
    fi
    ;;
  *)
    echo "[error] Unknown command '${COMMAND}'." >&2
    usage >&2
    exit 1
    ;;
esac
