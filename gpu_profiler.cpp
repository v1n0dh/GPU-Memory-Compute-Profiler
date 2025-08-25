// gpu_profiler_full.cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <mutex>

#define N (1<<24) // 16M elements
#define CPU_THREADS 4

std::mutex log_mutex;

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU Kernels
__global__ void memory_kernel(float* data, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += value;
}

__global__ void compute_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float tmp = data[idx];
        for (int i = 0; i < 100; ++i)
            tmp = tmp * 1.0001f + 0.0001f;
        data[idx] = tmp;
    }
}

__global__ void shared_mem_kernel(float* data) {
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        sdata[threadIdx.x] = data[idx];
        __syncthreads();
        sdata[threadIdx.x] += 1.0f;
        data[idx] = sdata[threadIdx.x];
    }
}

// Logging
void log_metrics(const std::string &metric, double value, std::ofstream &file) {
    std::lock_guard<std::mutex> guard(log_mutex);
    file << metric << "," << value << "\n";
}

// CPU workload
void cpu_workload(std::vector<float> &data, int thread_id) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = thread_id; i < data.size(); i += CPU_THREADS) {
        data[i] = data[i] * 1.0001f + 0.0001f;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "CPU thread " << thread_id << " finished in " << duration << " ms\n";
}

// Occupancy & launch configuration
void configure_kernel(dim3 &blocks, dim3 &threads, size_t total_elements) {
    int minGrid, blockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, memory_kernel, 0, 0));
    threads.x = blockSize;
    blocks.x = (total_elements + blockSize - 1) / blockSize;
}

// Print device properties
void print_device_properties() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Shared Memory/Block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Registers/Block: " << prop.regsPerBlock << "\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n";
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";
}

// Measure GPU kernel execution with bandwidth estimation
double run_kernel_with_bandwidth(void (*kernel)(float*, float), float* d_data, dim3 blocks, dim3 threads, float value, size_t size_bytes) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    kernel<<<blocks, threads>>>(d_data, value);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double bandwidthGBs = (size_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0);
    std::cout << "Kernel time: " << ms << " ms, Approx. Bandwidth: " << bandwidthGBs << " GB/s\n";
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    std::cout << "Full GPU Profiler with CUDA Streams and Bandwidth Analysis\n";
    print_device_properties();

    std::ofstream report("gpu_profiler_full_report.csv");
    report << "Metric,Time(ms)\n";

    // Host and device memory
    std::vector<float> h_data(N, 1.0f);
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads, blocks;
    configure_kernel(blocks, threads, N);

    // Memory kernel
    double mem_time = run_kernel_with_bandwidth(memory_kernel, d_data, blocks, threads, 1.0f, N * sizeof(float));
    log_metrics("GPU_Memory_Kernel", mem_time, report);

    // Compute kernel
    double comp_time = run_kernel_with_bandwidth(compute_kernel, d_data, blocks, threads, 0.0f, N * sizeof(float));
    log_metrics("GPU_Compute_Kernel", comp_time, report);

    // Shared memory kernel
    double shared_time = run_kernel_with_bandwidth(shared_mem_kernel, d_data, blocks, threads, 0.0f, N * sizeof(float));
    log_metrics("GPU_Shared_Mem_Kernel", shared_time, report);

    // Host-Device PCIe-like transfer simulation
    auto start_transfer = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    auto end_transfer = std::chrono::high_resolution_clock::now();
    double transfer_time = std::chrono::duration<double, std::milli>(end_transfer - start_transfer).count();
    std::cout << "PCIe-like Host-Device Transfer: " << transfer_time << " ms\n";
    log_metrics("Host_Device_Transfer", transfer_time, report);

    // CPU workload
    std::vector<std::thread> cpu_threads;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < CPU_THREADS; ++i)
        cpu_threads.emplace_back(cpu_workload, std::ref(h_data), i);
    for (auto &t : cpu_threads) t.join();
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    log_metrics("CPU_Workload", cpu_time, report);

    CUDA_CHECK(cudaFree(d_data));
    report.close();

    std::cout << "Profiler completed. Sample value: " << h_data[0] << "\n";
    return 0;
}
