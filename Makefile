CMAKE ?= cmake
BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= 4

.PHONY: all configure build clean help

all: configure build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build:
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean

help:
	@printf '%s\n' 'make             Configure and build a Release executable'
	@printf '%s\n' 'make JOBS=8      Use eight parallel build jobs'
	@printf '%s\n' 'make BUILD_TYPE=Debug  Configure and build a Debug executable'
	@printf '%s\n' 'make clean       Clean generated build artifacts'
