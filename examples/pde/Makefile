TARGET := $(shell basename `pwd`)

.PHONY: all clean setup

BIN_DIR := bin
CPP_SOURCE_DIR := cpp-src
CPP_SOURCES := $(shell find $(CPP_SOURCE_DIR) -name "*.cu")

DIM512 := -DSIZE=512 -DBLOCK_SIZE=65536 -DBLOCK_DIM=2 -DTHREAD_SIZE=1024
DIM256 := -DSIZE=256 -DBLOCK_SIZE=16384 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM128 := -DSIZE=128 -DBLOCK_SIZE=2048 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM96 := -DSIZE=96 -DBLOCK_SIZE=864 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM64 := -DSIZE=64 -DBLOCK_SIZE=256 -DBLOCK_DIM=1 -DTHREAD_SIZE=1024
DIM32 := -DSIZE=32 -DBLOCK_SIZE=64 -DBLOCK_DIM=1 -DTHREAD_SIZE=512

CC := nvcc 

CCFLAGS := -I$(CPP_SOURCE_DIR) -arch=sm_80 -std=c++17 -rdc=true -g -lcudadevrt -lineinfo
#clang: --cuda-gpu-arch=sm_80 -Lcm/shared/apps/cuda11.6/lib64 -lcudart_static -ldl -lrt -pthread -std=c++17
#nvcc : -arch=sm_80 -std=c++17 -rdc=true -g -lcudadevrt
#-g -mtune=native -march=native -O3 -std=c++17 -I$(CPP_SOURCE_DIR) -fopenmp
MKDIR := @mkdir
RM := @rm
#$(CC) $(CCFLAGS) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
d32: setup
	$(CC) $(CCFLAGS) $(DIM32) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

d64: setup
	$(CC) $(CCFLAGS) $(DIM64) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

d96: setup
	$(CC) $(CCFLAGS) $(DIM96) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

d128: setup
	$(CC) $(CCFLAGS) $(DIM128) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

d256: setup
	$(CC) $(CCFLAGS) $(DIM256) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

d512: setup
	$(CC) $(CCFLAGS) $(DIM512) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)


setup:
	$(MKDIR) -p $(BIN_DIR)

clean:
	$(RM) -r $(BIN_DIR)
