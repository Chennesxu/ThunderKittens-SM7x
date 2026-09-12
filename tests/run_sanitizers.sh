#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
build_dir=${BUILD_DIR:-"$root_dir/build"}
if [[ "$build_dir" != "$root_dir/build" ]]; then
    echo "BUILD_DIR must be $root_dir/build" >&2
    exit 2
fi

selector=${1:-}
if [[ $# -gt 1 ]]; then
    echo "unknown sanitizer selector: $*" >&2
    exit 2
fi
case "$selector" in
    "")
        sanitizer_test="$build_dir/gemm-sm75"
        build_hint=build-sm75
        memcheck_api=(--report-api-errors no)
        pass_message="SM75 sanitizers: PASS"
        ;;
    ptx-mma)
        sanitizer_test="$build_dir/ptx-mma-correctness-sm75"
        build_hint=build-ptx-mma-sm75
        memcheck_api=()
        pass_message="PTX MMA SM75 sanitizers: PASS"
        ;;
    *)
        echo "unknown sanitizer selector: $selector" >&2
        exit 2
        ;;
esac

if [[ ! -x "$sanitizer_test" ]]; then
    echo "sanitizer input is missing; run make $build_hint" >&2
    exit 1
fi
if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "SKIP: compute-sanitizer is unavailable" >&2
    exit 77
fi

set +e
"$sanitizer_test"
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

compute-sanitizer --tool memcheck "${memcheck_api[@]}" --error-exitcode 99 \
    "$sanitizer_test"
compute-sanitizer --tool racecheck --error-exitcode 99 "$sanitizer_test"
echo "$pass_message"
