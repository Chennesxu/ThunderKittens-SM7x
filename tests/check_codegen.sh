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
    local definition=${7:-}
    local extra_flags=()
    if [[ -n "$definition" ]]; then
        extra_flags+=("-D$definition")
    fi
    "$nvcc_bin" -std=c++17 -O3 -I"$root_dir/include" "-D$macro" \
        "${extra_flags[@]}" \
        --ptx "-arch=$compute" "$root_dir/$source" \
        -o "$build_dir/$prefix-$label.ptx"
    "$nvcc_bin" -std=c++17 -O3 -I"$root_dir/include" "-D$macro" \
        "${extra_flags[@]}" \
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

expect_layout_reject() {
    local label=$1
    local macro=$2
    local sm=$3
    local log="$build_dir/tile-layout-reject-$label.log"
    rm -f -- "$build_dir/tile-layout-reject-$label.o" "$log"
    set +e
    "$nvcc_bin" -std=c++17 -I"$root_dir/include" "-D$macro" "-arch=$sm" \
        -c "$root_dir/tests/tile_layout_reject.cu" \
        -o "$build_dir/tile-layout-reject-$label.o" >"$log" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 0 ]] || ! grep -Fq \
        "shared tile layout must be row_major or col_major" "$log"; then
        echo "unsupported shared tile layout gate failed: $label" >&2
        sed -n '1,120p' "$log" >&2
        exit 1
    fi
}

expect_ptx_backend_capability() {
    local positive_log="$build_dir/ptx-backend-capability-sm75.log"
    local negative_log="$build_dir/ptx-backend-capability-sm70.log"
    rm -f -- "$build_dir/ptx-backend-capability-sm75.o" \
        "$build_dir/ptx-backend-capability-sm70.o" "$positive_log" "$negative_log"
    "$nvcc_bin" -std=c++17 -I"$root_dir/include" -DKITTENS_SM75 -arch=sm_75 \
        -c "$root_dir/tests/ptx_backend_capability_reject.cu" \
        -o "$build_dir/ptx-backend-capability-sm75.o" >"$positive_log" 2>&1
    set +e
    "$nvcc_bin" -std=c++17 -I"$root_dir/include" -DKITTENS_SM70 -arch=sm_70 \
        -c "$root_dir/tests/ptx_backend_capability_reject.cu" \
        -o "$build_dir/ptx-backend-capability-sm70.o" >"$negative_log" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 0 ]] || ! grep -Fq \
        "PTX MMA backend is not supported by the active target" "$negative_log"; then
        echo "PTX backend capability gate failed" >&2
        sed -n '1,120p' "$negative_log" >&2
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

require_tensor_core_per_kernel() {
    local sass=$1
    local expected=$2
    local report
    report=$(awk -v RS='Function : ' '
        NR > 1 {
            n = split($0, line, "\n")
            if (line[1] !~ /gemm_kernel/) next
            hmma = 0
            ffma = 0
            for (i = 1; i <= n; i++) {
                if (line[i] ~ /HMMA/) hmma++
                if (line[i] ~ /FFMA/) ffma++
            }
            printf "%s %d %d\n", line[1], hmma, ffma
        }' "$sass")
    local seen
    seen=$(printf '%s\n' "$report" | grep -c . || true)
    if [[ "$seen" -ne "$expected" ]]; then
        echo "expected $expected GEMM kernels in $sass, found $seen" >&2
        printf '%s\n' "$report" >&2
        exit 1
    fi
    while read -r name hmma ffma; do
        [[ -z "$name" ]] && continue
        if [[ "$hmma" -eq 0 ]]; then
            echo "GEMM kernel without Tensor Core SASS: $name in $sass" >&2
            exit 1
        fi
        if [[ "$ffma" -ne 0 ]]; then
            echo "GEMM kernel with scalar FFMA fallback: $name in $sass" >&2
            exit 1
        fi
    done <<<"$report"
}

boundary_start='(^|[^A-Za-z0-9_.])'
boundary_end='([^A-Za-z0-9_.]|$)'
any_hmma="$boundary_start""HMMA[.]"
any_ffma="$boundary_start""FFMA""$boundary_end"
ptx_m8="$boundary_start"'mma[.]sync[.]aligned[.]m8n8k4[.]row[.]col[.]f32[.]f16[.]f16[.]f32'"$boundary_end"
ptx_m16="$boundary_start"'mma[.]sync[.]aligned[.]m16n8k8[.]row[.]col[.]f32[.]f16[.]f16[.]f32'"$boundary_end"
ptx_x2="$boundary_start"'ldmatrix[.]sync[.]aligned[.]m8n8[.]x2[.]shared[.]b16'"$boundary_end"
ptx_x1="$boundary_start"'ldmatrix[.]sync[.]aligned[.]m8n8[.]x1[.]shared[.]b16'"$boundary_end"

require_kernel_opcode() {
    local sass=$1
    local kernel=$2
    local opcode=$3
    local expected=$4
    local seen
    seen=$(awk -v RS='Function : ' -v kernel="$kernel" -v opcode="$opcode" '
        NR > 1 {
            n = split($0, line, "\n")
            if (line[1] != kernel) next
            found = 1
            for (i = 2; i <= n; i++) {
                if (line[i] ~ opcode) count++
            }
        }
        END { if (!found) { print "absent"; exit } print count + 0 }' "$sass")
    if [[ "$seen" != "$expected" ]]; then
        echo "expected $expected $opcode in $kernel of $sass, found $seen" >&2
        exit 1
    fi
}

require_ptx_kernel_opcode() {
    local ptx=$1
    local kernel=$2
    local opcode=$3
    local expected=$4
    local seen
    seen=$(awk -v kernel="$kernel" -v opcode="$opcode" '
        $0 ~ "^[[:space:]]*([.]visible[[:space:]]+)?[.]entry[[:space:]]+" kernel "[(]" {
            found = 1
            active = 1
        }
        active {
            if ($0 ~ opcode) count++
            opened = $0
            closed = $0
            open_count = gsub(/{/, "", opened)
            close_count = gsub(/}/, "", closed)
            depth += open_count - close_count
            if (open_count > 0) started = 1
            if (started && depth == 0) active = 0
        }
        END { if (!found) { print "absent"; exit } print count + 0 }' "$ptx")
    if [[ "$seen" != "$expected" ]]; then
        echo "expected $expected $opcode in $kernel of $ptx, found $seen" >&2
        exit 1
    fi
}

gemm_kernel_name() {
    local ptx=$1
    local signature=$2
    local names
    names=$(awk -v signature="$signature" '
        ($0 ~ "^[[:space:]]*([.]visible[[:space:]]+)?[.]entry[[:space:]]+" ||
                $0 ~ "^[[:space:]]*Function[[:space:]]*:[[:space:]]+") &&
                index($0, signature) {
            name = $0
            sub(/^.*[.]entry[[:space:]]+/, "", name)
            sub(/^.*Function[[:space:]]*:[[:space:]]+/, "", name)
            sub(/[(].*$/, "", name)
            print name
        }' "$ptx")
    local seen
    seen=$(printf '%s\n' "$names" | grep -c . || true)
    if [[ "$seen" -ne 1 ]]; then
        echo "expected one GEMM kernel matching $signature in $ptx, found $seen" >&2
        printf '%s\n' "$names" >&2
        exit 1
    fi
    printf '%s\n' "$names"
}

exercise_gemm_kernel_name_boundaries() {
    local ptx_probe="$build_dir/gemm-kernel-name.ptx.probe"
    local sass_probe="$build_dir/gemm-kernel-name.sass.probe"
    printf '%s\n' \
        '.entry _ZN7tk_sm7x63_GLOBAL__N__39_tmpxft_0000000c_gemm_kernelILi64ELi64ELi2ELi2EEE(' \
        '.entry _ZN7tk_sm7x63_GLOBAL__N__39_tmpxft_0000000c_gemm_kernelILi128ELi128ELi4ELi2EEE(' \
        >"$ptx_probe"
    printf '%s\n' \
        'Function : _ZN7tk_sm7x63_GLOBAL__N__39_tmpxft_00000014_gemm_kernelILi128ELi128ELi4ELi2EEE' \
        'Function : _ZN7tk_sm7x63_GLOBAL__N__39_tmpxft_00000014_gemm_kernelILi64ELi64ELi2ELi2EEE' \
        >"$sass_probe"
    local ptx_small
    local sass_small
    ptx_small=$(gemm_kernel_name "$ptx_probe" \
        'gemm_kernelILi64ELi64ELi2ELi2EEE')
    sass_small=$(gemm_kernel_name "$sass_probe" \
        'gemm_kernelILi64ELi64ELi2ELi2EEE')
    if [[ "$ptx_small" == "$sass_small" ]]; then
        echo "GEMM kernel name resolver coupled independent artifacts" >&2
        exit 1
    fi
    gemm_kernel_name "$ptx_probe" \
        'gemm_kernelILi128ELi128ELi4ELi2EEE' >/dev/null
    gemm_kernel_name "$sass_probe" \
        'gemm_kernelILi128ELi128ELi4ELi2EEE' >/dev/null
    rm -f -- "$ptx_probe" "$sass_probe"
}

require_ptx_gemm_kernel_population() {
    local ptx=$1
    local expected=$2
    local seen
    seen=$(awk '
        $0 ~ /^[[:space:]]*([.]visible[[:space:]]+)?[.]entry[[:space:]]+/ &&
                /gemm_kernel/ { count++ }
        END { print count + 0 }' "$ptx")
    if [[ "$seen" -ne "$expected" ]]; then
        echo "expected $expected GEMM kernels in $ptx, found $seen" >&2
        exit 1
    fi
}

exercise_ptx_parser_boundaries() {
    local probe="$build_dir/ptx-parser-boundary.probe"
    printf '%s\n' \
        '.visible .entry codegen_ptx_parser_boundary()' \
        '{' \
        'xmma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32;' \
        'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32x;' \
        'xldmatrix.sync.aligned.m8n8.x2.shared.b16;' \
        'ldmatrix.sync.aligned.m8n8.x1.shared.b16x;' \
        '}' >"$probe"
    require_ptx_kernel_opcode "$probe" codegen_ptx_parser_boundary "$ptx_m8" 0
    require_ptx_kernel_opcode "$probe" codegen_ptx_parser_boundary "$ptx_m16" 0
    require_ptx_kernel_opcode "$probe" codegen_ptx_parser_boundary "$ptx_x2" 0
    require_ptx_kernel_opcode "$probe" codegen_ptx_parser_boundary "$ptx_x1" 0
    rm -f -- "$probe"
}

exercise_ptx_parser_boundaries
exercise_gemm_kernel_name_boundaries

compile_target ptx tests/codegen_ptx_mma.cu sm70 KITTENS_SM70 compute_70 sm_70
compile_target ptx tests/codegen_ptx_mma.cu sm75 KITTENS_SM75 compute_75 sm_75
compile_target ldmatrix tests/codegen_ldmatrix.cu sm75 KITTENS_SM75 compute_75 sm_75
compile_target ptx-backend tests/codegen_ptx_backend.cu sm70 KITTENS_SM70 compute_70 sm_70
compile_target ptx-backend tests/codegen_ptx_backend.cu sm75 KITTENS_SM75 compute_75 sm_75
compile_target mma tests/codegen_mma.cu sm70 KITTENS_SM70 compute_70 sm_70
compile_target mma tests/codegen_mma.cu sm75 KITTENS_SM75 compute_75 sm_75
compile_target gemm src/gemm.cu sm70 KITTENS_SM70 compute_70 sm_70
compile_target gemm src/gemm.cu sm75 KITTENS_SM75 compute_75 sm_75
compile_target gemm-ptx src/gemm.cu sm70 KITTENS_SM70 compute_70 sm_70 KITTENS_MMA_PTX
compile_target gemm-ptx src/gemm.cu sm75 KITTENS_SM75 compute_75 sm_75 KITTENS_MMA_PTX
expect_backend_mismatch sm70 KITTENS_SM70 sm_70
expect_backend_mismatch sm75 KITTENS_SM75 sm_75
expect_layout_reject sm70 KITTENS_SM70 sm_70
expect_layout_reject sm75 KITTENS_SM75 sm_75
expect_ptx_backend_capability

for prefix in ldmatrix ptx-backend mma gemm gemm-ptx; do
    if [[ "$prefix" == ldmatrix ]]; then
        require_pattern "$build_dir/$prefix-sm75.ptx" '\.target[[:space:]]+sm_75' \
            "sm_75 target"
        continue
    fi
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

for ptx in "$build_dir/gemm-ptx-sm70.ptx" "$build_dir/gemm-ptx-sm75.ptx"; do
    reject_pattern "$ptx" 'wmma[.]' "WMMA instruction in PTX GEMM"
    reject_pattern "$ptx" 'cp[.]async|mbarrier|wgmma|tensormap|stmatrix|m16n8k16' \
        "SM80+ instruction"
    require_ptx_gemm_kernel_population "$ptx" 2
done

gemm_ptx_sm70_small=$(gemm_kernel_name "$build_dir/gemm-ptx-sm70.ptx" \
    'gemm_kernelILi64ELi64ELi2ELi2EEE')
gemm_ptx_sm70_large=$(gemm_kernel_name "$build_dir/gemm-ptx-sm70.ptx" \
    'gemm_kernelILi128ELi128ELi4ELi2EEE')
gemm_sass_sm70_small=$(gemm_kernel_name "$build_dir/gemm-ptx-sm70.sass" \
    'gemm_kernelILi64ELi64ELi2ELi2EEE')
gemm_sass_sm70_large=$(gemm_kernel_name "$build_dir/gemm-ptx-sm70.sass" \
    'gemm_kernelILi128ELi128ELi4ELi2EEE')
gemm_ptx_sm75_small=$(gemm_kernel_name "$build_dir/gemm-ptx-sm75.ptx" \
    'gemm_kernelILi64ELi64ELi2ELi2EEE')
gemm_ptx_sm75_large=$(gemm_kernel_name "$build_dir/gemm-ptx-sm75.ptx" \
    'gemm_kernelILi128ELi128ELi4ELi2EEE')
gemm_sass_sm75_small=$(gemm_kernel_name "$build_dir/gemm-ptx-sm75.sass" \
    'gemm_kernelILi64ELi64ELi2ELi2EEE')
gemm_sass_sm75_large=$(gemm_kernel_name "$build_dir/gemm-ptx-sm75.sass" \
    'gemm_kernelILi128ELi128ELi4ELi2EEE')

for kernel_count in \
    "$gemm_ptx_sm70_small $gemm_sass_sm70_small 16" \
    "$gemm_ptx_sm70_large $gemm_sass_sm70_large 32"; do
    read -r ptx_kernel sass_kernel expected <<<"$kernel_count"
    require_ptx_kernel_opcode "$build_dir/gemm-ptx-sm70.ptx" "$ptx_kernel" \
        "$ptx_m8" "$expected"
    require_ptx_kernel_opcode "$build_dir/gemm-ptx-sm70.ptx" "$ptx_kernel" \
        "$ptx_m16" 0
    require_ptx_kernel_opcode "$build_dir/gemm-ptx-sm70.ptx" "$ptx_kernel" \
        'wmma[.]' 0
    require_ptx_kernel_opcode "$build_dir/gemm-ptx-sm70.ptx" "$ptx_kernel" \
        'ldmatrix' 0
    for step in STEP0 STEP1 STEP2 STEP3; do
        require_kernel_opcode "$build_dir/gemm-ptx-sm70.sass" "$sass_kernel" \
            "$boundary_start"'HMMA[.]884[.]F32[.]F32[.]'"$step""$boundary_end" \
            "$expected"
    done
    require_kernel_opcode "$build_dir/gemm-ptx-sm70.sass" "$sass_kernel" \
        "$any_hmma" "$((expected * 4))"
    require_kernel_opcode "$build_dir/gemm-ptx-sm70.sass" "$sass_kernel" \
        "$boundary_start"'HMMA[.]1688[.]' 0
    require_kernel_opcode "$build_dir/gemm-ptx-sm70.sass" "$sass_kernel" \
        "$boundary_start"'LDSM[.]' 0
    require_kernel_opcode "$build_dir/gemm-ptx-sm70.sass" "$sass_kernel" \
        "$any_ffma" 0
done

for kernel_count in \
    "$gemm_ptx_sm75_small $gemm_sass_sm75_small 16" \
    "$gemm_ptx_sm75_large $gemm_sass_sm75_large 32"; do
    read -r ptx_kernel sass_kernel expected <<<"$kernel_count"
    require_ptx_kernel_opcode "$build_dir/gemm-ptx-sm75.ptx" "$ptx_kernel" \
        "$ptx_m16" "$expected"
    require_ptx_kernel_opcode "$build_dir/gemm-ptx-sm75.ptx" "$ptx_kernel" \
        "$ptx_m8" 0
    require_ptx_kernel_opcode "$build_dir/gemm-ptx-sm75.ptx" "$ptx_kernel" \
        'wmma[.]' 0
    require_kernel_opcode "$build_dir/gemm-ptx-sm75.sass" "$sass_kernel" \
        "$boundary_start"'HMMA[.]1688[.]F32'"$boundary_end" "$expected"
    require_kernel_opcode "$build_dir/gemm-ptx-sm75.sass" "$sass_kernel" \
        "$boundary_start"'HMMA[.]884[.]' 0
    require_kernel_opcode "$build_dir/gemm-ptx-sm75.sass" "$sass_kernel" \
        "$any_hmma" "$expected"
    require_kernel_opcode "$build_dir/gemm-ptx-sm75.sass" "$sass_kernel" \
        "$any_ffma" 0
done
reject_pattern "$build_dir/ldmatrix-sm75.ptx" \
    'cp\.async|mbarrier|wgmma|tensormap|stmatrix' "SM80+ instruction"
for ptx in "$build_dir/ptx-backend-sm70.ptx" \
           "$build_dir/ptx-backend-sm75.ptx"; do
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

# Whole-file checks cannot see one dispatched kernel degrading while another
# still emits HMMA, so every GEMM kernel instantiation is inspected on its own.
require_tensor_core_per_kernel "$build_dir/gemm-sm70.sass" 2
require_tensor_core_per_kernel "$build_dir/gemm-sm75.sass" 2

# The inline-PTX wrappers must keep emitting their intended instruction shape and
# must not decay into a scalar path.
for ptx in "$build_dir/ptx-sm70.ptx" "$build_dir/ptx-sm75.ptx"; do
    require_pattern "$ptx" "$ptx_m8" \
        "m8n8k4 instruction"
done
require_pattern "$build_dir/ptx-sm75.ptx" \
    "$ptx_m16" "m16n8k8 instruction"
reject_pattern "$build_dir/ptx-sm70.ptx" 'ldmatrix|m16n8k8|m16n8k16' \
    "SM75+ matrix instruction"
reject_pattern "$build_dir/ptx-backend-sm70.ptx" 'ldmatrix|m16n8k8|m16n8k16' \
    "SM75+ matrix instruction"
for sass in "$build_dir/ptx-sm70.sass" "$build_dir/ptx-sm75.sass"; do
    require_pattern "$sass" 'HMMA' "Tensor Core SASS"
    reject_pattern "$sass" 'FFMA' "scalar FFMA fallback"

    # A file-wide HMMA lets one wrapper hide behind another, so each wrapper is
    # pinned to the exact machine instructions its shape must lower to, and to
    # how many Tensor Core instructions it may issue in total.
    for step in STEP0 STEP1 STEP2 STEP3; do
        require_kernel_opcode "$sass" codegen_ptx_m8n8k4 \
            "$boundary_start""HMMA[.]884[.]F32[.]F32[.]$step""$boundary_end" 1
    done
    require_kernel_opcode "$sass" codegen_ptx_m8n8k4 "$any_hmma" 4
    require_kernel_opcode "$sass" codegen_ptx_m8n8k4 "$any_ffma" 0
done
require_kernel_opcode "$build_dir/ptx-sm75.sass" codegen_ptx_m16n8k8 \
    "$boundary_start""HMMA[.]1688[.]F32""$boundary_end" 1
require_kernel_opcode "$build_dir/ptx-sm75.sass" codegen_ptx_m16n8k8 "$any_hmma" 1
require_kernel_opcode "$build_dir/ptx-sm75.sass" codegen_ptx_m16n8k8 "$any_ffma" 0
reject_pattern "$build_dir/ptx-sm70.sass" 'HMMA[.]1688' "SM75+ Tensor Core SASS"
reject_pattern "$build_dir/ptx-backend-sm70.sass" 'HMMA[.]1688|LDSM' \
    "SM75+ matrix SASS"

for kernel in codegen_ldmatrix_x2 codegen_ldmatrix_x1; do
    require_kernel_opcode "$build_dir/ldmatrix-sm75.sass" "$kernel" \
        "$boundary_start"'LDSM[.]' 1
    require_kernel_opcode "$build_dir/ldmatrix-sm75.sass" "$kernel" "$any_ffma" 0
done
require_ptx_kernel_opcode "$build_dir/ldmatrix-sm75.ptx" codegen_ldmatrix_x2 \
    "$ptx_x2" 1
require_ptx_kernel_opcode "$build_dir/ldmatrix-sm75.ptx" codegen_ldmatrix_x1 \
    "$ptx_x1" 1

for label in sm70 sm75; do
    require_ptx_kernel_opcode "$build_dir/ptx-backend-$label.ptx" \
        codegen_ptx_backend_sm70 "$ptx_m8" 4
    require_ptx_kernel_opcode "$build_dir/ptx-backend-$label.ptx" \
        codegen_ptx_backend_sm70 "$ptx_m16" 0
    require_ptx_kernel_opcode "$build_dir/ptx-backend-$label.ptx" \
        codegen_ptx_backend_sm70 "$ptx_x2" 0
    require_ptx_kernel_opcode "$build_dir/ptx-backend-$label.ptx" \
        codegen_ptx_backend_sm70 "$ptx_x1" 0
    for step in STEP0 STEP1 STEP2 STEP3; do
        require_kernel_opcode "$build_dir/ptx-backend-$label.sass" \
            codegen_ptx_backend_sm70 \
            "$boundary_start"'HMMA[.]884[.]F32[.]F32[.]'"$step""$boundary_end" 4
    done
    require_kernel_opcode "$build_dir/ptx-backend-$label.sass" \
        codegen_ptx_backend_sm70 "$any_hmma" 16
    require_kernel_opcode "$build_dir/ptx-backend-$label.sass" \
        codegen_ptx_backend_sm70 "$any_ffma" 0
    require_kernel_opcode "$build_dir/ptx-backend-$label.sass" \
        codegen_ptx_backend_sm70 "$boundary_start"'LDSM[.]' 0
done

require_ptx_kernel_opcode "$build_dir/ptx-backend-sm75.ptx" \
    codegen_ptx_backend_sm75 \
    "$ptx_m16" 4
require_ptx_kernel_opcode "$build_dir/ptx-backend-sm75.ptx" \
    codegen_ptx_backend_sm75 \
    "$ptx_x2" 2
require_ptx_kernel_opcode "$build_dir/ptx-backend-sm75.ptx" \
    codegen_ptx_backend_sm75 \
    "$ptx_x1" 4
require_ptx_kernel_opcode "$build_dir/ptx-backend-sm75.ptx" \
    codegen_ptx_backend_sm75 'ldmatrix[.].*[.]trans' 0
require_kernel_opcode "$build_dir/ptx-backend-sm75.sass" \
    codegen_ptx_backend_sm75 \
    "$boundary_start"'HMMA[.]1688[.]F32'"$boundary_end" 4
require_kernel_opcode "$build_dir/ptx-backend-sm75.sass" \
    codegen_ptx_backend_sm75 "$any_hmma" 4
require_kernel_opcode "$build_dir/ptx-backend-sm75.sass" \
    codegen_ptx_backend_sm75 "$any_ffma" 0
require_kernel_opcode "$build_dir/ptx-backend-sm75.sass" \
    codegen_ptx_backend_sm75 "$boundary_start"'LDSM[.]' 6

echo "codegen gate: PASS"
