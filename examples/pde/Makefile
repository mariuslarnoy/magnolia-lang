TARGET := $(shell basename `pwd`).bin

.PHONY: all clean setup

BIN_DIR := bin
CPP_SOURCE_DIR := cpp-src
CPP_SOURCES := $(shell find $(CPP_SOURCE_DIR) -name "*.cu")
DEFINES := -DPAD0=0 -DPAD1=0 -DPAD2=0

CC := clang++

# if clang++, can use
# -mllvm -inline-threshold=1000000000
# 
CCFLAGS := -g -I$(CPP_SOURCE_DIR) -mllvm -inline-threshold=1000000000 --cuda-gpu-arch=sm_86 -nocudalib --no-cuda-version-check

MKDIR := @mkdir
RM := @rm

all: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

setup:
	$(MKDIR) -p $(BIN_DIR)

clean:
	$(RM) -r $(BIN_DIR)
