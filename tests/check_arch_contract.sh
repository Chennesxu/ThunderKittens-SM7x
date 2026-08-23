#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
nvcc_bin=${NVCC:-nvcc}
build_dir=${BUILD_DIR:-"$root_dir/build"}

if [[ "$build_dir" != "$root_dir/build" ]]; then
    echo "BUILD_DIR must be $root_dir/build" >&2
    exit 2
fi

mkdir -p "$build_dir"
common=(-std=c++17 -I"$root_dir/include")

compile_positive() {
    local label=$1
    shift
    "$nvcc_bin" "${common[@]}" "$@" -c "$root_dir/tests/arch_contract.cu" \
        -o "$build_dir/arch-$label.o"
}

expect_failure() {
    local label=$1
    local diagnostic=$2
    shift 2
    local object="$build_dir/arch-negative-$label.o"
    local log="$build_dir/arch-negative-$label.log"
    rm -f -- "$object" "$log"
    set +e
    "$nvcc_bin" "${common[@]}" "$@" -c "$root_dir/tests/arch_contract.cu" \
        -o "$object" >"$log" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 0 ]]; then
        echo "negative architecture case unexpectedly compiled: $label" >&2
        exit 1
    fi
    if ! grep -Fq "$diagnostic" "$log"; then
        echo "negative architecture case missed diagnostic: $label" >&2
        sed -n '1,120p' "$log" >&2
        exit 1
    fi
}

compile_positive sm70 -DKITTENS_SM70 -gencode arch=compute_70,code=sm_70
compile_positive sm75 -DKITTENS_SM75 -gencode arch=compute_75,code=sm_75

expect_failure no-macro "Define exactly one" \
    -gencode arch=compute_70,code=sm_70
expect_failure both-macros "Define exactly one" \
    -DKITTENS_SM70 -DKITTENS_SM75 -gencode arch=compute_70,code=sm_70
expect_failure sm70-on-sm75 "KITTENS_SM70 requires sm_70" \
    -DKITTENS_SM70 -gencode arch=compute_75,code=sm_75
expect_failure sm75-on-sm70 "KITTENS_SM75 requires sm_75" \
    -DKITTENS_SM75 -gencode arch=compute_70,code=sm_70
expect_failure sm70-on-sm80 "KITTENS_SM70 requires sm_70" \
    -DKITTENS_SM70 -gencode arch=compute_80,code=sm_80
expect_failure sm75-on-sm80 "KITTENS_SM75 requires sm_75" \
    -DKITTENS_SM75 -gencode arch=compute_80,code=sm_80

echo "architecture contract: PASS"
