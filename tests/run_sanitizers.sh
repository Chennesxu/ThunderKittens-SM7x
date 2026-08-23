#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
build_dir=${BUILD_DIR:-"$root_dir/build"}
if [[ "$build_dir" != "$root_dir/build" ]]; then
    echo "BUILD_DIR must be $root_dir/build" >&2
    exit 2
fi

gemm_test="$build_dir/gemm-sm75"
if [[ ! -x "$gemm_test" ]]; then
    echo "sanitizer input is missing; run make build-sm75" >&2
    exit 1
fi
if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "SKIP: compute-sanitizer is unavailable" >&2
    exit 77
fi

set +e
"$gemm_test"
preflight_status=$?
set -e
if [[ $preflight_status -eq 77 ]]; then
    exit 77
fi
if [[ $preflight_status -ne 0 ]]; then
    exit "$preflight_status"
fi

if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    echo "Compute Sanitizer preserving CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    echo "Compute Sanitizer preserving unrestricted CUDA visibility"
fi

compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 99 "$gemm_test"
compute-sanitizer --tool racecheck --error-exitcode 99 "$gemm_test"
echo "SM75 sanitizers: PASS"
