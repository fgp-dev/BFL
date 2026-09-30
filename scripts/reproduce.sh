#!/usr/bin/env bash
# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/results/local}"
mkdir -p "${OUT_DIR}" "${ROOT_DIR}/build/reproduce"
CXX_BIN="${CXX:-g++}"
FLAGS=(-std=c++17 -O3 -Wall -Wextra -Wpedantic -I"${ROOT_DIR}/include")
"${CXX_BIN}" "${FLAGS[@]}" "${ROOT_DIR}/src/bfl.cpp" "${ROOT_DIR}/tests/test_bfl.cpp" -o "${ROOT_DIR}/build/reproduce/bfl_tests"
"${ROOT_DIR}/build/reproduce/bfl_tests" > "${OUT_DIR}/tests.txt"
"${CXX_BIN}" "${FLAGS[@]}" "${ROOT_DIR}/src/bfl.cpp" "${ROOT_DIR}/benchmarks/benchmark.cpp" -o "${ROOT_DIR}/build/reproduce/bfl_benchmark"
"${ROOT_DIR}/build/reproduce/bfl_benchmark" --sizes 4,6,8 --seeds 1,2,3 --epochs 1 --repeats 100 --route-prefix-bits 0 > "${OUT_DIR}/benchmark.csv"
for EPOCHS in 1 3; do
  for PREFIX_BITS in 0 4 8 16; do
    if [[ "${EPOCHS}" == 1 && "${PREFIX_BITS}" == 0 ]]; then continue; fi
    "${ROOT_DIR}/build/reproduce/bfl_benchmark" --sizes 4,6,8 --seeds 1,2,3 --epochs "${EPOCHS}" --repeats 100 --route-prefix-bits "${PREFIX_BITS}" | tail -n +2 >> "${OUT_DIR}/benchmark.csv"
  done
done
{
    printf 'UTC date: '; date -u +'%Y-%m-%d %H:%M:%S UTC'
    printf 'Compiler: '; "${CXX_BIN}" --version | head -1
    printf 'Kernel: '; uname -srvmo
    printf 'CPU: '; lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1
    printf 'Logical CPUs: '; nproc
    printf 'Memory bytes: '; awk '/MemTotal:/ {print $2 * 1024}' /proc/meminfo
    printf 'Test command: g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/bfl.cpp tests/test_bfl.cpp -o build/reproduce/bfl_tests\n'
    printf 'Benchmark command: build/reproduce/bfl_benchmark --sizes 4,6,8 --seeds 1,2,3 --epochs {1,3} --repeats 100 --route-prefix-bits {0,4,8,16}\n'
    printf 'Data generator: benchmarks/benchmark.cpp::make_split and target_bits\n'
    printf 'Split: shuffled unique domain; first half train, next quarter test\n'
} > "${OUT_DIR}/environment.txt"
printf 'Results written to %s\n' "${OUT_DIR}"
