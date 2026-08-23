SHELL := /bin/bash
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override BUILD_DIR := $(ROOT_DIR)/build
NVCC ?= nvcc
COMMON_FLAGS := -std=c++17 -O2 -lineinfo -I$(ROOT_DIR)/include -Xcompiler=-Wall,-Wextra
SM70_FLAGS := -DKITTENS_SM70 -gencode arch=compute_70,code=sm_70
SM75_FLAGS := -DKITTENS_SM75 -gencode arch=compute_75,code=sm_75

.PHONY: all check-arch clean

all: check-arch

check-arch:
	NVCC="$(NVCC)" BUILD_DIR="$(BUILD_DIR)" bash tests/check_arch_contract.sh

clean:
	@test "$(BUILD_DIR)" = "$(ROOT_DIR)/build"
	rm -rf -- "$(ROOT_DIR)/build"
