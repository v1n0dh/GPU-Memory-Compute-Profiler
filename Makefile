CUDA_PATH ?= /usr/local/cuda
NVML_LIB ?= -lnvidia-ml

CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

CXXFLAGS := -O3 -std=c++17 -Iinclude -I$(CUDA_PATH)/include
LDFLAGS  := -L$(CUDA_PATH)/lib64 $(NVML_LIB) -lcudart -lpthread

SRC_CPP := src/profiler.cpp src/nvml_helpers.cpp src/system_stats.cpp src/main.cpp
SRC_CU  := src/kernels.cu

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

BIN := gpuprof

all: build $(BIN)

build:
	mkdir -p build out

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -O3 -std=c++17 -Iinclude -c $< -o $@

$(BIN): $(OBJ_CPP) $(OBJ_CU)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(BIN)
	rm -rf out

.PHONY: all clean build

