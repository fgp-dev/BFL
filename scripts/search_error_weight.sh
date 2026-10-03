#!/usr/bin/env bash
# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/results/local-weight-search}"
BUILD_DIR="${ROOT_DIR}/build/generalization"
mkdir -p "${OUT_DIR}" "${BUILD_DIR}"

"${CXX:-g++}" -std=c++17 -O3 -Wall -Wextra -Wpedantic \
  -I"${ROOT_DIR}/include" "${ROOT_DIR}/src/bfl.cpp" \
  "${ROOT_DIR}/src/greedy.cpp" "${ROOT_DIR}/benchmarks/generalization.cpp" \
  -o "${BUILD_DIR}/generalization"

WEIGHTS=()
for ((weight = 1; weight <= 128; ++weight)); do WEIGHTS+=("${weight}"); done
WEIGHTS+=(256 512 1024 25000)
"${BUILD_DIR}/generalization" --sweep "${WEIGHTS[@]}" > "${OUT_DIR}/validation_sweep.csv"
python3 "${ROOT_DIR}/scripts/select_error_weight.py" \
  "${OUT_DIR}/validation_sweep.csv" "${OUT_DIR}"
read -r SELECTED < "${OUT_DIR}/selected_weight.txt"
"${BUILD_DIR}/generalization" --selected-weight "${SELECTED}" > "${OUT_DIR}/benchmark.csv"

: > "${OUT_DIR}/tests.txt"
for NAME in test_bfl test_greedy; do
  "${CXX:-g++}" -std=c++17 -O3 -Wall -Wextra -Wpedantic \
    -I"${ROOT_DIR}/include" "${ROOT_DIR}/src/bfl.cpp" \
    "${ROOT_DIR}/src/greedy.cpp" "${ROOT_DIR}/tests/${NAME}.cpp" \
    -o "${BUILD_DIR}/${NAME}"
  "${BUILD_DIR}/${NAME}" >> "${OUT_DIR}/tests.txt"
done
printf 'Weights searched: integers 1..128, 256, 512, 1024, 25000\nSelection: minimum pooled validation errors across 3 tasks x 20 seeds; smallest X breaks ties; no test data used\n' > "${OUT_DIR}/protocol.txt"
printf 'Results written to %s\n' "${OUT_DIR}"
