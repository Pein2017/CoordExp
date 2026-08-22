#!/usr/bin/env bash
#
# Unified public dataset preparation runner for CoordExp.
# Retained COCO/LVIS preparation entry.

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
  --preset <name>          Preset dir name under public_data/<dataset>/
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
  ./public_data/run.sh lvis all --preset rescale_32_768_bbox
  ./public_data/run.sh coco all --preset rescale_32_1024_bbox
EOF
}

run_py() {
  local python_bin="${PYTHON:-python}"
  echo "+ PYTHONPATH=. ${python_bin} $*" >&2
  PYTHONPATH=. "${python_bin}" "$@"
}

PIPELINE_LAST_OUTPUT_DIR=""

run_pipeline_factory_capture_output_dir() {
  local tmp_log
  tmp_log="$(mktemp)"

  local python_bin="${PYTHON:-python}"
  echo "+ PYTHONPATH=. ${python_bin} -m public_data.scripts.run_pipeline_factory $*" >&2
  set +e
  PYTHONPATH=. "${python_bin}" -m public_data.scripts.run_pipeline_factory "$@" 2>&1 | tee "${tmp_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    rm -f "${tmp_log}"
    return "${rc}"
  fi

  PIPELINE_LAST_OUTPUT_DIR="$(sed -n 's/^\[pipeline\] output_dir=//p' "${tmp_log}" | tail -n 1)"
  rm -f "${tmp_log}"
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
  PRESET_TRAIN_JSONL="${PRESET_DIR}/train.jsonl"
  PRESET_VAL_JSONL="${PRESET_DIR}/val.jsonl"
  PRESET_TRAIN_NORM_JSONL="${PRESET_DIR}/train.norm.jsonl"
  PRESET_VAL_NORM_JSONL="${PRESET_DIR}/val.norm.jsonl"
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
      run_py -m public_data.scripts.run_pipeline_factory \
        "${PIPELINE_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py -m public_data.scripts.run_pipeline_factory "${PIPELINE_ARGS[@]}"
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
      run_py -m public_data.scripts.run_pipeline_factory \
        "${PIPELINE_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py -m public_data.scripts.run_pipeline_factory "${PIPELINE_ARGS[@]}"
    fi
    ;;
  rescale)
    [[ -n "${PRESET}" ]] || die "rescale requires --preset <name>"
    set_paths_for_preset
    banner "[${DATASET}] rescale -> ${PRESET_DIR}"
    require_file "${RAW_TRAIN_JSONL}"
    if [[ -n "${PIPELINE_MAX_OBJECTS}" ]]; then
      die "PUBLIC_DATA_MAX_OBJECTS is only supported for 'coord'. Run rescale first, then coord with PUBLIC_DATA_MAX_OBJECTS."
    fi
    PIPELINE_ARGS=(
      --mode rescale
      --dataset-id "${DATASET}"
      --dataset-dir "${DATASET_DIR}"
      --raw-dir "${RAW_DIR}"
      --preset "${PRESET}"
    )
    run_py -m public_data.scripts.run_pipeline_factory \
      "${PIPELINE_ARGS[@]}" \
      "${PASSTHROUGH_ARGS[@]}"
    set_paths_for_preset
    ;;
  coord)
    [[ -n "${PRESET}" ]] || die "coord requires --preset <name>"
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
    run_py -m public_data.scripts.run_pipeline_factory \
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
      die "PUBLIC_DATA_MAX_OBJECTS is only supported for 'coord'. Validate derived outputs by passing the derived preset name directly."
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
    run_pipeline_factory_capture_output_dir "${PIPELINE_ARGS[@]}"

    if [[ "${DO_PRESET}" == "true" ]]; then
      if [[ -n "${PIPELINE_LAST_OUTPUT_DIR}" ]]; then
        PRESET_DIR="${PIPELINE_LAST_OUTPUT_DIR}"
        PRESET_TRAIN_COORD_JSONL="${PRESET_DIR}/train.coord.jsonl"
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
      run_py -m public_data.scripts.run_pipeline_factory \
        --mode download \
        "${PIPELINE_INGEST_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py -m public_data.scripts.run_pipeline_factory \
        --mode download \
        "${PIPELINE_INGEST_ARGS[@]}"
    fi
    banner "[${DATASET}] stage: convert"
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
      run_py -m public_data.scripts.run_pipeline_factory \
        --mode convert \
        "${PIPELINE_INGEST_ARGS[@]}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    else
      run_py -m public_data.scripts.run_pipeline_factory \
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
      die "PUBLIC_DATA_MAX_OBJECTS is only supported for 'coord'. For two-step flow: run 'all' without max_objects, then run 'coord' with PUBLIC_DATA_MAX_OBJECTS."
    fi
    if [[ "${SKIP_IMAGE_CHECK}" == "true" ]]; then
      PIPELINE_ARGS+=(--skip-image-check)
    fi
    run_pipeline_factory_capture_output_dir "${PIPELINE_ARGS[@]}"
    set_paths_for_preset

    ;;
  *)
    echo "[error] Unknown command '${COMMAND}'." >&2
    usage >&2
    exit 1
    ;;
esac
