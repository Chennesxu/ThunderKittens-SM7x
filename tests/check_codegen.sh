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
nvcc_path=$(command -v "$nvcc_bin")
cuobjdump_bin=${CUOBJDUMP:-"$(dirname "$nvcc_path")/cuobjdump"}
if [[ ! -x "$cuobjdump_bin" ]]; then
    cuobjdump_bin=$(command -v cuobjdump)
fi

compile_target() {
    local prefix=$1
    local source=$2
    local label=$3
    local macro=$4
    local compute=$5
    local sm=$6
    "$nvcc_bin" -std=c++17 -O3 -I"$root_dir/include" "-D$macro" \
        --ptx "-arch=$compute" "$root_dir/$source" \
        -o "$build_dir/$prefix-$label.ptx"
    "$nvcc_bin" -std=c++17 -O3 -I"$root_dir/include" "-D$macro" \
        --cubin "-arch=$sm" "$root_dir/$source" \
        -o "$build_dir/$prefix-$label.cubin"
    "$cuobjdump_bin" --dump-sass "$build_dir/$prefix-$label.cubin" \
        >"$build_dir/$prefix-$label.sass"
}

expect_backend_mismatch() {
    local label=$1
    local macro=$2
    local sm=$3
    local log="$build_dir/backend-mismatch-$label.log"
    rm -f -- "$build_dir/backend-mismatch-$label.o" "$log"
    set +e
    "$nvcc_bin" -std=c++17 -I"$root_dir/include" "-D$macro" "-arch=$sm" \
        -c "$root_dir/tests/backend_target_mismatch.cu" \
        -o "$build_dir/backend-mismatch-$label.o" >"$log" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 0 ]] || ! grep -Fq \
        "MMA backend architecture must match active target" "$log"; then
        echo "wrong backend instantiation gate failed: $label" >&2
        sed -n '1,120p' "$log" >&2
        exit 1
    fi
}

require_pattern() {
    local file=$1
    local pattern=$2
    local label=$3
    if ! grep -Eq "$pattern" "$file"; then
        echo "missing $label in $file" >&2
        exit 1
    fi
}

reject_pattern() {
    local file=$1
    local pattern=$2
    local label=$3
    if grep -Eiq "$pattern" "$file"; then
        echo "found prohibited $label in $file" >&2
        exit 1
    fi
}

compile_target mma tests/codegen_mma.cu sm70 KITTENS_SM70 compute_70 sm_70
compile_target mma tests/codegen_mma.cu sm75 KITTENS_SM75 compute_75 sm_75
compile_target gemm src/gemm.cu sm70 KITTENS_SM70 compute_70 sm_70
compile_target gemm src/gemm.cu sm75 KITTENS_SM75 compute_75 sm_75
expect_backend_mismatch sm70 KITTENS_SM70 sm_70
expect_backend_mismatch sm75 KITTENS_SM75 sm_75

for prefix in mma gemm; do
    require_pattern "$build_dir/$prefix-sm70.ptx" '\.target[[:space:]]+sm_70' "sm_70 target"
    require_pattern "$build_dir/$prefix-sm75.ptx" '\.target[[:space:]]+sm_75' "sm_75 target"
done

version_text=$("$nvcc_bin" --version)
if grep -Fq 'release 12.4' <<<"$version_text"; then
    load_a='wmma\.load\.a\.sync\.aligned\.row\.m16n16k16\.shared\.f16'
    load_b='wmma\.load\.b\.sync\.aligned\.col\.m16n16k16\.shared\.f16'
    mma='wmma\.mma\.sync\.aligned\.row\.col\.m16n16k16\.f32\.f32'
    store='wmma\.store\.d\.sync\.aligned\.row\.m16n16k16\.shared\.f32'
elif grep -Eq 'release 11\.[0-9]+' <<<"$version_text"; then
    load_a='wmma\.load\.a\.sync\.aligned\.row\.m16n16k16(\.shared)?\.f16'
    load_b='wmma\.load\.b\.sync\.aligned\.col\.m16n16k16(\.shared)?\.f16'
    mma='wmma\.mma\.sync\.aligned\.row\.col\.m16n16k16\.f32\.f32'
    store='wmma\.store\.d\.sync\.aligned\.row\.m16n16k16(\.shared)?\.f32'
else
    load_a='wmma\.load\.a\.sync\.aligned\.row\.m16n16k16\.shared\.f16'
    load_b='wmma\.load\.b\.sync\.aligned\.col\.m16n16k16\.shared\.f16'
    mma='wmma\.mma\.sync\.aligned\.row\.col\.m16n16k16\.f32\.f32'
    store='wmma\.store\.d\.sync\.aligned\.row\.m16n16k16\.shared\.f32'
fi

for ptx in "$build_dir/mma-sm70.ptx" "$build_dir/mma-sm75.ptx" \
           "$build_dir/gemm-sm70.ptx" "$build_dir/gemm-sm75.ptx"; do
    require_pattern "$ptx" "$load_a" "WMMA A load"
    require_pattern "$ptx" "$load_b" "WMMA B load"
    require_pattern "$ptx" "$mma" "WMMA FP32 accumulate"
    require_pattern "$ptx" "$store" "WMMA FP32 store"
    reject_pattern "$ptx" 'cp\.async|mbarrier|wgmma|tensormap|stmatrix' \
        "SM80+ instruction"
done

for ptx in "$build_dir/mma-sm70.ptx" "$build_dir/gemm-sm70.ptx"; do
    reject_pattern "$ptx" 'ldmatrix|m16n8k8|m16n8k16' "SM75+ matrix instruction"
done
for sass in "$build_dir/mma-sm70.sass" "$build_dir/mma-sm75.sass" \
            "$build_dir/gemm-sm70.sass" "$build_dir/gemm-sm75.sass"; do
    require_pattern "$sass" 'HMMA' "Tensor Core SASS"
    reject_pattern "$sass" 'FFMA' "scalar FFMA fallback"
done

echo "codegen gate: PASS"
