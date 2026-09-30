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
EXPECTED_ACCOUNT_ID=334613428
AUTHOR_EMAIL="${EXPECTED_ACCOUNT_ID}+fgp-dev@users.noreply.github.com"
mkdir -p "${ROOT_DIR}/build"
STAGE_DIR="$(mktemp -d "${ROOT_DIR}/build/publish.XXXXXX")"

# Copy only intended release assets; local credentials and build output stay out.
cp -a "${ROOT_DIR}/CMakeLists.txt" "${ROOT_DIR}/README.md" \
      "${ROOT_DIR}/LICENSE" "${ROOT_DIR}/NOTICE" "${ROOT_DIR}/CITATION.cff" \
      "${ROOT_DIR}/BFL_Theory.tex" "${ROOT_DIR}/.gitignore" "${STAGE_DIR}/"
cp -a "${ROOT_DIR}/include" "${ROOT_DIR}/src" "${ROOT_DIR}/examples" \
      "${ROOT_DIR}/tests" "${ROOT_DIR}/benchmarks" "${ROOT_DIR}/scripts" \
      "${ROOT_DIR}/docs" \
      "${ROOT_DIR}/results" "${ROOT_DIR}/.github" "${STAGE_DIR}/"

if [[ "${MODE}" == "--publish" ]]; then
    gh auth status >/dev/null
    ACCOUNT_ID="$(gh api user --jq '.id')"
    if [[ "${ACCOUNT_ID}" != "${EXPECTED_ACCOUNT_ID}" ]]; then
        printf 'Authenticated GitHub account is not fgp-dev.\n' >&2
        exit 1
    fi
    if gh repo view fgp-dev/BFL >/dev/null 2>&1; then
        printf 'fgp-dev/BFL already exists; refusing to overwrite it.\n' >&2
        exit 1
    fi
fi

git -C "${STAGE_DIR}" init --initial-branch=main >/dev/null
git -C "${STAGE_DIR}" config user.name "Facundo Gomez Prates"
git -C "${STAGE_DIR}" config user.email "${AUTHOR_EMAIL}"
git -C "${STAGE_DIR}" add -A
git -C "${STAGE_DIR}" commit -m "Initial BFL release" >/dev/null

printf 'Prepared source tree: %s\n' "${STAGE_DIR}"
git -C "${STAGE_DIR}" status --short
git -C "${STAGE_DIR}" log -1 --format='%h %an <%ae> %s'

if [[ "${MODE}" == "--publish" ]]; then
    gh repo create fgp-dev/BFL --public --source "${STAGE_DIR}" --remote origin --push \
        --description "Bit Flip Learning library in C++17 by Facundo Gomez Prates"
    printf 'Published: https://github.com/fgp-dev/BFL\n'
fi
