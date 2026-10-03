#!/usr/bin/env bash
# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/results/local-generalization}"
BUILD_DIR="${ROOT_DIR}/build/generalization"
mkdir -p "${OUT_DIR}" "${BUILD_DIR}"

: > "${OUT_DIR}/tests.txt"
for NAME in test_bfl test_greedy; do
  "${CXX:-g++}" -std=c++17 -O3 -Wall -Wextra -Wpedantic \
    -I"${ROOT_DIR}/include" "${ROOT_DIR}/src/bfl.cpp" \
    "${ROOT_DIR}/src/greedy.cpp" "${ROOT_DIR}/tests/${NAME}.cpp" \
    -o "${BUILD_DIR}/${NAME}"
  "${BUILD_DIR}/${NAME}" >> "${OUT_DIR}/tests.txt"
done

"${CXX:-g++}" -std=c++17 -O3 -Wall -Wextra -Wpedantic \
  -I"${ROOT_DIR}/include" "${ROOT_DIR}/src/bfl.cpp" \
  "${ROOT_DIR}/src/greedy.cpp" "${ROOT_DIR}/benchmarks/generalization.cpp" \
  -o "${BUILD_DIR}/generalization"
"${BUILD_DIR}/generalization" > "${OUT_DIR}/benchmark.csv"
{
  printf 'UTC date: '; date -u +'%Y-%m-%d %H:%M:%S UTC'
  printf 'C++ compiler: '; "${CXX:-g++}" --version | head -1
  printf 'CPU: '; lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1
  printf 'Implementation: BFL C++17 greedy MUX port with single-bit and CHANGE/USED masked rules; no C dependency\n'
  printf 'Task and split: benchmarks/generalization.cpp; 8 variable bits plus constant one; shuffled finite domain; 128 train, 64 validation, 64 test; seeds 1..20\n'
  printf 'Greedy BFL: 3 hidden layers x 64 MUXes; 100 epochs maximum; 128 candidates/epoch; patience 15; 1 or 16 topology starts, selected by validation only\n'
  printf 'Masked BFL: joint flips on active paths of wrong outputs; global objective error_weight * error bits plus used, unused, or zero usage term; error_weight 10 or 1; same topology seeds and training limits\n'
  printf 'Direct-state control: one adaptive lookup leaf per 8-bit input, count routing, 3 online epochs\n'
  printf 'MLPs: 9 inputs, 4 or 16 tanh hidden units, sigmoid output, full-batch Adam 1500 epochs; one initialization each\n'
} > "${OUT_DIR}/environment.txt"
printf 'Results written to %s\n' "${OUT_DIR}"
