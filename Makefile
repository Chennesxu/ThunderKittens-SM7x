SHELL := /bin/bash
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override BUILD_DIR := $(ROOT_DIR)/build
NVCC ?= nvcc
COMMON_FLAGS := -std=c++17 -O2 -lineinfo -I$(ROOT_DIR)/include -Xcompiler=-Wall,-Wextra
SM70_FLAGS := -DKITTENS_SM70 -gencode arch=compute_70,code=sm_70
SM75_FLAGS := -DKITTENS_SM75 -gencode arch=compute_75,code=sm_75

.PHONY: all check-arch check-codegen build-mma-sm70 build-mma-sm75 test-mma-sm75 clean

all: check-arch check-codegen build-mma-sm70 build-mma-sm75

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

clean:
	@test "$(BUILD_DIR)" = "$(ROOT_DIR)/build"
	rm -rf -- "$(ROOT_DIR)/build"
