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
  --skip-image-check       Skip image existence checks during validation
  --raw-only               For validate: validate only raw artifacts (no preset required)
  --preset-only            For validate: validate only preset artifacts

Passthrough args:
  Everything after `--` is forwarded to the underlying implementation:
    - download/convert: dataset plugin script (public_data/datasets/<dataset>.sh)
    - rescale/coord/validate: unified pipeline factory (public_data/scripts/run_pipeline_factory.py)
  The pipeline factory supports a curated subset of flags and warns on unsupported args.
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

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Missing required command on PATH: ${cmd}"
}

download_coco_raw_aria2c() {
  # Fast path for COCO 2017 raw download using aria2c multi-connection downloads.
  # Layout matches public_data/coco/raw/ conventions used elsewhere in this repo.
  require_cmd aria2c
  require_cmd unzip
  require_cmd sha256sum

  local downloads_dir="${RAW_DIR}/downloads"
  run_cmd mkdir -p "${downloads_dir}"
  run_cmd mkdir -p "${RAW_IMAGE_DIR}"

  # Canonical URLs. Prefer plain HTTP here because some cluster environments
  # MITM/terminate TLS and can cause aria2c hostname mismatch failures.
  local url_train="http://images.cocodataset.org/zips/train2017.zip"
  local url_val="http://images.cocodataset.org/zips/val2017.zip"
  local url_ann="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

  banner "[coco] aria2c download (multi-conn) -> ${downloads_dir}"
  local train_zip="${downloads_dir}/train2017.zip"
  local val_zip="${downloads_dir}/val2017.zip"
  local ann_zip="${downloads_dir}/annotations_trainval2017.zip"

  # Skip already-complete downloads (prevents re-hitting the server and failing on transient 503s).
  # If a corresponding *.aria2 file exists, treat it as incomplete and resume with -c.
  if [[ -f "${train_zip}" && ! -f "${train_zip}.aria2" ]]; then
    echo "[coco] skip download: already present ${train_zip}" >&2
  else
    run_cmd aria2c -c -x 16 -s 16 -k 1M -d "${downloads_dir}" "${url_train}"
  fi

  if [[ -f "${val_zip}" && ! -f "${val_zip}.aria2" ]]; then
    echo "[coco] skip download: already present ${val_zip}" >&2
  else
    run_cmd aria2c -c -x 16 -s 16 -k 1M -d "${downloads_dir}" "${url_val}"
  fi

  if [[ -f "${ann_zip}" && ! -f "${ann_zip}.aria2" ]]; then
    echo "[coco] skip download: already present ${ann_zip}" >&2
  else
    run_cmd aria2c -c -x 16 -s 16 -k 1M -d "${downloads_dir}" "${url_ann}"
  fi

  banner "[coco] sha256 checksums -> ${downloads_dir}/SHA256SUMS.txt"
  run_cmd bash -c "cd \"${downloads_dir}\" && sha256sum train2017.zip val2017.zip annotations_trainval2017.zip > SHA256SUMS.txt"

  banner "[coco] extract images -> ${RAW_IMAGE_DIR}"
  run_cmd unzip -q -n "${downloads_dir}/train2017.zip" -d "${RAW_IMAGE_DIR}"
  run_cmd unzip -q -n "${downloads_dir}/val2017.zip" -d "${RAW_IMAGE_DIR}"

  banner "[coco] extract annotations -> ${RAW_DIR}"
  run_cmd unzip -q -n "${downloads_dir}/annotations_trainval2017.zip" -d "${RAW_DIR}"
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
  local root
  root="$(echo "${base_preset}" | sed -E 's/(_max_?[0-9]+)+$//')"
  local canonical="${root}_max_${max_objects}"
  local legacy="${root}_max${max_objects}"
  if [[ -d "${DATASET_DIR}/${legacy}" && ! -d "${DATASET_DIR}/${canonical}" ]]; then
    echo "${legacy}"
  else
    echo "${canonical}"
  fi
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
    require_plugin_file
    banner "[${DATASET}] download -> ${RAW_DIR}"
    run_cmd mkdir -p "${RAW_DIR}"
    if [[ "${DATASET}" == "coco" && ${#PASSTHROUGH_ARGS[@]} -eq 0 ]]; then
      download_coco_raw_aria2c
    else
      run_plugin download \
        --repo-root "${REPO_ROOT}" \
        --dataset "${DATASET}" \
        --dataset-dir "${DATASET_DIR}" \
        --raw-dir "${RAW_DIR}" \
        --raw-image-dir "${RAW_IMAGE_DIR}" \
        --raw-train-jsonl "${RAW_TRAIN_JSONL}" \
        --raw-val-jsonl "${RAW_VAL_JSONL}" \
        -- "${PASSTHROUGH_ARGS[@]}"
    fi
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
    require_plugin_file

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
