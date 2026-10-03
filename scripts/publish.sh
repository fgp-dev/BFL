#!/usr/bin/env bash
# Copyright 2026 Facundo Gomez Prates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODE="${1:---prepare}"
if [[ "${MODE}" != "--prepare" && "${MODE}" != "--publish" ]]; then
    printf 'Usage: bash scripts/publish.sh [--prepare|--publish]\n' >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT_DIR}/build" --parallel
ctest --test-dir "${ROOT_DIR}/build" --output-on-failure

if [[ "${MODE}" == "--prepare" ]]; then
    git -C "${ROOT_DIR}" status --short
    printf 'Checks passed. Review changes, commit them, then run --publish.\n'
    exit 0
fi

if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
    printf 'Commit or remove uncommitted changes before publishing.\n' >&2
    exit 1
fi
if [[ "$(git -C "${ROOT_DIR}" branch --show-current)" != main ]]; then
    printf 'Switch to the main branch before publishing.\n' >&2
    exit 1
fi
if [[ "$(git -C "${ROOT_DIR}" remote get-url origin)" != 'https://github.com/fgp-dev/BFL.git' ]]; then
    printf 'The origin remote is not https://github.com/fgp-dev/BFL.git.\n' >&2
    exit 1
fi
EXPECTED_ACCOUNT_ID=334613428
if [[ "$(gh api user --jq '.id')" != "${EXPECTED_ACCOUNT_ID}" ]]; then
    printf 'Authenticated GitHub account is not fgp-dev.\n' >&2
    exit 1
fi
git -C "${ROOT_DIR}" push -u origin main
