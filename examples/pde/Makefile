TARGET := $(shell basename `pwd`).bin

.PHONY: all clean setup

BIN_DIR := bin
CU_SOURCE_DIR := cuda-src
CU_SOURCES := $(shell find $(CU_SOURCE_DIR) -name "*.cu")
DEFINES := -DPAD0=0 -DPAD1=0 -DPAD2=0

MKDIR := @mkdir
RM := @rm

DIM512 := -DSIZE=512 -DBLOCK_SIZE=65536 -DBLOCK_DIM=2 -DTHREAD_SIZE=1024
DIM256 := -DSIZE=256 -DBLOCK_SIZE=16384 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM128 := -DSIZE=128 -DBLOCK_SIZE=2048 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM96 := -DSIZE=96 -DBLOCK_SIZE=864 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM64 := -DSIZE=64 -DBLOCK_SIZE=256 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM32 := -DSIZE=32 -DBLOCK_SIZE=64 -DBLOCK_DIM=1 -DTHREAD_SIZE=512

CC := nvcc 

CCFLAGS := -I$(CU_SOURCE_DIR) -arch=sm_61 -std=c++17 -rdc=true -g -lcudadevrt -lineinfo

MKDIR := @mkdir
RM := @rm

d32: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(DIM32) $(CU_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CU_SOURCES)

d64: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(DIM64) $(CU_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CU_SOURCES)

d128: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(DIM128) $(CU_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CU_SOURCES)

d256: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(DIM256) $(CU_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CU_SOURCES)

d512: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(DIM512) $(CU_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CU_SOURCES)

all: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(CU_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CU_SOURCES)

setup:
	$(MKDIR) -p $(BIN_DIR)

clean:
	$(RM) -r $(BIN_DIR)
