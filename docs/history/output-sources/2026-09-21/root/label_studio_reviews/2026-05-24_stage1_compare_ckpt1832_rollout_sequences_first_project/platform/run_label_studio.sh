#!/usr/bin/env bash
set -euo pipefail

BUNDLE=/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project
PLATFORM="$BUNDLE/platform"
CREDS="$PLATFORM/label_studio_credentials.env"

set -a
. "$CREDS"
set +a

exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY='*' no_proxy='*' \
  LATEST_VERSION_CHECK=false \
  COORDEXP_LABEL_STUDIO_SINGLE_USER=true \
  LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true \
  DISABLE_SIGNUP_WITHOUT_LINK=true \
  INACTIVITY_SESSION_TIMEOUT_ENABLED=0 \
  MAX_SESSION_AGE=315360000 \
  MAX_TIME_BETWEEN_ACTIVITY=315360000 \
  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="$LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED" \
  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT" \
  LABEL_STUDIO_BASE_DATA_DIR="$LABEL_STUDIO_BASE_DATA_DIR" \
  SECRET_KEY="$LABEL_STUDIO_SECRET_KEY" \
  USERNAME="$LABEL_STUDIO_USERNAME" \
  PASSWORD="$LABEL_STUDIO_PASSWORD" \
  USER_TOKEN="$LABEL_STUDIO_API_KEY" \
  "$PLATFORM/.venv/bin/label-studio" start \
    --no-browser \
    --data-dir "$LABEL_STUDIO_BASE_DATA_DIR" \
    --database "$LABEL_STUDIO_BASE_DATA_DIR/label_studio.sqlite3" \
    --internal-host 127.0.0.1 \
    --port 18080 \
    --host "$LABEL_STUDIO_URL" \
    --username "$LABEL_STUDIO_USERNAME" \
    --password "$LABEL_STUDIO_PASSWORD" \
    --user-token "$LABEL_STUDIO_API_KEY" \
    --log-level INFO
