TARGET := $(shell basename `pwd`).bin

.PHONY: all clean setup

BIN_DIR := bin
CPP_SOURCE_DIR := cpp-src
CPP_SOURCES := $(shell find $(CPP_SOURCE_DIR) -name "*.cu")
DEFINES := -DPAD0=0 -DPAD1=0 -DPAD2=0

CC := nvcc 

# if clang++, can use
# -mllvm -inline-threshold=1000000000
# 
CCFLAGS := -I$(CPP_SOURCE_DIR) -arch=sm_80 -std=c++17 -rdc=true -g -lcudadevrt 

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
