#!/bin/bash
# Test runner for LVIS converter
# Always uses fixed ms python

set -e

PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"

echo "Running tests with python: ${PYTHON_BIN}"
echo "======================================"

# Run from this repo's public_data folder (avoid hard-coded absolute paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_DATA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PUBLIC_DATA_DIR}"

# Run converter tests
echo -e "\n[1/3] Running LVIS Pre-sorting Tests..."
"${PYTHON_BIN}" tests/test_lvis_presort.py

echo -e "\n[2/3] Running LVIS Converter Tests..."
"${PYTHON_BIN}" tests/test_lvis_converter.py

# Run polygon cap smoke tests
echo -e "\n[3/3] Running Polygon Cap Smoke Tests..."
"${PYTHON_BIN}" tests/test_poly_cap.py

echo -e "\n======================================"
echo "All tests completed!"
echo "======================================"
