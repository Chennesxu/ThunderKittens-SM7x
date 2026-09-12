SHELL := /bin/bash
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override BUILD_DIR := $(ROOT_DIR)/build
NVCC ?= nvcc
COMMON_FLAGS := -std=c++17 -O2 -lineinfo -I$(ROOT_DIR)/include -Xcompiler=-Wall,-Wextra
SM70_FLAGS := -DKITTENS_SM70 -gencode arch=compute_70,code=sm_70
SM75_FLAGS := -DKITTENS_SM75 -gencode arch=compute_75,code=sm_75

.PHONY: all check-arch check-codegen build-mma-sm70 build-mma-sm75 test-mma-sm75 build-sm70 build-sm75 test-sm75 sanitize-sm75 build-bench-sm75 bench-sm75 build-layout-sm70 build-layout-sm75 test-layout-sm75 build-ldmatrix-sm75 test-ldmatrix-sm75 build-ptx-mma-sm75 test-ptx-mma-sm75 clean

all: check-arch check-codegen build-mma-sm70 build-mma-sm75 build-layout-sm70 build-layout-sm75 build-ldmatrix-sm75 build-ptx-mma-sm75 build-sm70 build-sm75

$(BUILD_DIR):
	mkdir -p "$(BUILD_DIR)"

check-arch:
	NVCC="$(NVCC)" BUILD_DIR="$(BUILD_DIR)" bash tests/check_arch_contract.sh

check-codegen: check-arch
	NVCC="$(NVCC)" BUILD_DIR="$(BUILD_DIR)" bash tests/check_codegen.sh

$(BUILD_DIR)/mma-correctness-sm70: tests/mma_correctness.cu tests/test_utils.cuh include/tk_sm7x/arch.cuh include/tk_sm7x/mma.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM70_FLAGS) tests/mma_correctness.cu -o $@

$(BUILD_DIR)/mma-correctness-sm75: tests/mma_correctness.cu tests/test_utils.cuh include/tk_sm7x/arch.cuh include/tk_sm7x/mma.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM75_FLAGS) tests/mma_correctness.cu -o $@

build-mma-sm70: $(BUILD_DIR)/mma-correctness-sm70

build-mma-sm75: $(BUILD_DIR)/mma-correctness-sm75

test-mma-sm75: build-mma-sm75
	@status=0; "$(BUILD_DIR)/mma-correctness-sm75" || status=$$?; \
	if [[ $$status -eq 77 ]]; then echo "SKIP: MMA SM75 runtime validation pending (binary exit 77)"; exit 0; fi; \
	exit $$status

$(BUILD_DIR)/gemm-sm70: src/gemm.cu tests/gemm_correctness.cu tests/test_utils.cuh include/tk_sm7x/arch.cuh include/tk_sm7x/mma.cuh include/tk_sm7x/tile.cuh include/tk_sm7x/gemm.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM70_FLAGS) src/gemm.cu tests/gemm_correctness.cu -o $@

$(BUILD_DIR)/gemm-sm75: src/gemm.cu tests/gemm_correctness.cu tests/test_utils.cuh include/tk_sm7x/arch.cuh include/tk_sm7x/mma.cuh include/tk_sm7x/tile.cuh include/tk_sm7x/gemm.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM75_FLAGS) src/gemm.cu tests/gemm_correctness.cu -o $@

build-sm70: $(BUILD_DIR)/gemm-sm70

build-sm75: $(BUILD_DIR)/gemm-sm75

test-sm75: build-sm75
	@status=0; "$(BUILD_DIR)/gemm-sm75" || status=$$?; \
	if [[ $$status -eq 77 ]]; then echo "SKIP: GEMM SM75 runtime validation pending (binary exit 77)"; exit 0; fi; \
	exit $$status

$(BUILD_DIR)/mma-layout-sm70: tests/mma_layout.cu tests/mma_layout_oracle.cuh tests/test_utils.cuh include/tk_sm7x/arch.cuh include/tk_sm7x/ptx_mma.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM70_FLAGS) tests/mma_layout.cu -o $@

$(BUILD_DIR)/mma-layout-sm75: tests/mma_layout.cu tests/mma_layout_oracle.cuh tests/test_utils.cuh include/tk_sm7x/arch.cuh include/tk_sm7x/ptx_mma.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM75_FLAGS) tests/mma_layout.cu -o $@

build-layout-sm70: $(BUILD_DIR)/mma-layout-sm70

build-layout-sm75: $(BUILD_DIR)/mma-layout-sm75

test-layout-sm75: build-layout-sm75
	@status=0; "$(BUILD_DIR)/mma-layout-sm75" || status=$$?; \
	if [[ $$status -eq 77 ]]; then echo "SKIP: MMA layout SM75 runtime validation pending (binary exit 77)"; exit 0; fi; \
	exit $$status

$(BUILD_DIR)/ldmatrix-layout-sm75: tests/ldmatrix_layout.cu \
		tests/mma_layout_oracle.cuh tests/test_utils.cuh \
		include/tk_sm7x/arch.cuh include/tk_sm7x/ptx_ldmatrix.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM75_FLAGS) \
		tests/ldmatrix_layout.cu -o $@

build-ldmatrix-sm75: $(BUILD_DIR)/ldmatrix-layout-sm75

test-ldmatrix-sm75: build-ldmatrix-sm75
	@status=0; "$(BUILD_DIR)/ldmatrix-layout-sm75" || status=$$?; \
	if [[ $$status -eq 77 ]]; then echo "SKIP: ldmatrix SM75 runtime validation pending (binary exit 77)"; exit 0; fi; \
	exit $$status

$(BUILD_DIR)/ptx-mma-correctness-sm75: tests/ptx_mma_correctness.cu \
		tests/mma_layout_oracle.cuh tests/test_utils.cuh \
		include/tk_sm7x/arch.cuh include/tk_sm7x/ptx_mma.cuh \
		include/tk_sm7x/ptx_ldmatrix.cuh include/tk_sm7x/ptx_backend.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM75_FLAGS) \
		tests/ptx_mma_correctness.cu -o $@

build-ptx-mma-sm75: $(BUILD_DIR)/ptx-mma-correctness-sm75

test-ptx-mma-sm75: build-ptx-mma-sm75
	@status=0; "$(BUILD_DIR)/ptx-mma-correctness-sm75" || status=$$?; \
	if [[ $$status -eq 77 ]]; then echo "SKIP: PTX MMA SM75 runtime validation pending (binary exit 77)"; exit 0; fi; \
	exit $$status

$(BUILD_DIR)/gemm-throughput-sm75: bench/gemm_throughput.cu tests/test_utils.cuh src/gemm.cu include/tk_sm7x/arch.cuh include/tk_sm7x/mma.cuh include/tk_sm7x/tile.cuh include/tk_sm7x/gemm.cuh | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) -I$(ROOT_DIR)/tests $(SM75_FLAGS) src/gemm.cu bench/gemm_throughput.cu -o $@

build-bench-sm75: $(BUILD_DIR)/gemm-throughput-sm75

bench-sm75: build-bench-sm75
	@status=0; "$(BUILD_DIR)/gemm-throughput-sm75" || status=$$?; \
	if [[ $$status -eq 77 ]]; then echo "SKIP: GEMM SM75 benchmark pending (binary exit 77)"; exit 0; fi; \
	exit $$status

sanitize-sm75: build-sm75
	@status=0; BUILD_DIR="$(BUILD_DIR)" bash tests/run_sanitizers.sh || status=$$?; \
	if [[ $$status -eq 77 ]]; then echo "SKIP: SM75 sanitizer validation pending (script exit 77)"; exit 0; fi; \
	exit $$status

clean:
	@test "$(BUILD_DIR)" = "$(ROOT_DIR)/build"
	rm -rf -- "$(ROOT_DIR)/build"
