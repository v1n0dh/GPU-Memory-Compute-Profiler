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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU kernels simulating ML accelerator workloads
__global__ void memory_bandwidth_kernel(float* data, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += value;
}

__global__ void compute_intensive_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float tmp = data[idx];
        for (int i = 0; i < 200; ++i)  // heavy loop simulating math ops
            tmp = tmp * 1.0001f + 0.0001f;
        data[idx] = tmp;
    }
}

__global__ void shared_memory_kernel(float* data) {
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        sdata[threadIdx.x] = data[idx];
        __syncthreads();
        sdata[threadIdx.x] = sdata[threadIdx.x] * 1.1f; // simulate shared memory ops
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

// Configure kernel
void configure_kernel(dim3 &blocks, dim3 &threads, size_t total_elements) {
    int minGrid, blockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, memory_bandwidth_kernel, 0, 0));
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

// Run kernel with timing
template <typename Kernel>
double run_kernel_with_stream(Kernel kernel, float* d_data, dim3 blocks, dim3 threads, float value, size_t size_bytes, cudaStream_t stream) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    
    kernel<<<blocks, threads, 0, stream>>>(d_data, value);
    
    CUDA_CHECK(cudaEventRecord(stop, stream));
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
    std::cout << "GPU/ML Accelerator Profiler (C++/CUDA)\n";
    print_device_properties();

    std::ofstream report("gpu_profiler_report.csv");
    report << "Metric,Time(ms)\n";

    // Allocate memory
    std::vector<float> h_data(N, 1.0f);
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads, blocks;
    configure_kernel(blocks, threads, N);

    // CUDA streams
    cudaStream_t stream1, stream2, stream3;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaStreamCreate(&stream3));

    // Run kernels asynchronously
    double mem_time = run_kernel_with_stream(memory_bandwidth_kernel, d_data, blocks, threads, 1.0f, N * sizeof(float), stream1);
    log_metrics("GPU_Memory_Bandwidth", mem_time, report);

    double comp_time = run_kernel_with_stream(compute_intensive_kernel, d_data, blocks, threads, 0.0f, N * sizeof(float), stream2);
    log_metrics("GPU_Compute_Intensive", comp_time, report);

    double shared_time = run_kernel_with_stream(shared_memory_kernel, d_data, blocks, threads, 0.0f, N * sizeof(float), stream3);
    log_metrics("GPU_Shared_Memory", shared_time, report);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Host-device transfer (PCIe-like)
    auto start_transfer = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    auto end_transfer = std::chrono::high_resolution_clock::now();
    double transfer_time = std::chrono::duration<double, std::milli>(end_transfer - start_transfer).count();
    std::cout << "PCIe-like Transfer: " << transfer_time << " ms\n";
    log_metrics("Host_Device_Transfer", transfer_time, report);

    // CPU workload
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> cpu_threads;
    for (int i = 0; i < CPU_THREADS; ++i)
        cpu_threads.emplace_back(cpu_workload, std::ref(h_data), i);
    for (auto &t : cpu_threads) t.join();
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    log_metrics("CPU_Workload", cpu_time, report);

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaStreamDestroy(stream3));
    CUDA_CHECK(cudaFree(d_data));
    report.close();

    std::cout << "Profiler completed. Sample value: " << h_data[0] << "\n";
    return 0;
}
