# Compiler and flags
NVCC := nvcc
NVCC_FLAGS := -O3 -arch=native -std=c++17
INCLUDE_FLAGS := -Iinclude -Iicicle

# Target executable
TARGET := sumcheck.bin

# Source file
SRC := src/sumcheck_cuda.cu

# Default field to use
USE_FIELD ?= useM31ext3

# Set the minimum required version
NVCC_MIN_VERSION := 12.5

# Get NVCC version
NVCC_VERSION := $(shell $(NVCC) --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -n 1)
NVCC_MAJOR := $(shell echo $(NVCC_VERSION) | cut -d. -f1)
NVCC_MINOR := $(shell echo $(NVCC_VERSION) | cut -d. -f2)

# Phony targets
.PHONY: all check_nvcc clean

# Default target
all: check_nvcc $(TARGET)

# Check NVCC version
check_nvcc:
	@if ! command -v $(NVCC) > /dev/null 2>&1; then \
		echo "Error: $(NVCC) is not installed."; \
		exit 1; \
	elif [ $(NVCC_MAJOR) -lt 12 ] || ([ $(NVCC_MAJOR) -eq 12 ] && [ $(NVCC_MINOR) -lt 5 ]); then \
		echo "Error: $(NVCC) version must be >= $(NVCC_MIN_VERSION). Current version: $(NVCC_VERSION)"; \
		exit 1; \
	else \
		echo "$(NVCC) version $(NVCC_VERSION) is sufficient."; \
	fi

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -D$(USE_FIELD) -o $@ $<

# Clean build
clean:
	rm -f $(TARGET)
