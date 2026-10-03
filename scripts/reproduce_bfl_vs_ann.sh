#!/usr/bin/env bash
# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${1:-${ROOT_DIR}/results}"
SEEDS="${2:-50}"
BUILD_DIR="${ROOT_DIR}/build/bfl-vs-ann"
mkdir -p "${RESULT_DIR}" "${BUILD_DIR}"
for NAME in test_bfl test_greedy test_routed_binary; do
  "${CXX:-g++}" -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic \
    -I"${ROOT_DIR}/include" "${ROOT_DIR}/src/bfl.cpp" \
    "${ROOT_DIR}/src/greedy.cpp" "${ROOT_DIR}/src/routed_binary.cpp" \
    "${ROOT_DIR}/tests/${NAME}.cpp" \
    -o "${BUILD_DIR}/${NAME}"
  "${BUILD_DIR}/${NAME}"
done
"${CXX:-g++}" -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic \
  -I"${ROOT_DIR}/include" "${ROOT_DIR}/src/bfl.cpp" \
  "${ROOT_DIR}/src/greedy.cpp" "${ROOT_DIR}/src/routed_binary.cpp" \
  "${ROOT_DIR}/benchmarks/benchmark_bfl_vs_ann.cpp" \
  -o "${BUILD_DIR}/benchmark_bfl_vs_ann"
"${CXX:-g++}" -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic \
  -I"${ROOT_DIR}/include" "${ROOT_DIR}/src/bfl.cpp" \
  "${ROOT_DIR}/src/greedy.cpp" "${ROOT_DIR}/src/routed_binary.cpp" \
  "${ROOT_DIR}/benchmarks/routed_binary_andu.cpp" \
  -o "${BUILD_DIR}/routed_binary_andu"
"${BUILD_DIR}/routed_binary_andu" 12345 > "${RESULT_DIR}/routed_binary_andu_12345.txt"
"${BUILD_DIR}/benchmark_bfl_vs_ann" "${SEEDS}" > "${RESULT_DIR}/bfl_vs_ann.csv"
python3 "${ROOT_DIR}/scripts/summarize_bfl_vs_ann.py" \
  "${RESULT_DIR}/bfl_vs_ann.csv" > "${RESULT_DIR}/summary.txt"
{
  printf 'UTC date: '; date -u +'%Y-%m-%d %H:%M:%S UTC'
  printf 'Compiler: '; "${CXX:-g++}" --version | head -1
  printf 'Flags: -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic\n'
  printf 'CPU: '; lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1
  printf 'Seeds: 1..%s\n' "${SEEDS}"
} > "${RESULT_DIR}/environment_bfl_vs_ann.txt"
printf 'Results written to %s\n' "${RESULT_DIR}"
