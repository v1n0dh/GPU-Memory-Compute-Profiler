#include <cuda_runtime.h>
#include <cstdio>

__global__ void fma_kernel(float* a, float* b, float* c, size_t n, int iters) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float y = b[i];
    float z = c[i];
    #pragma unroll 4
    for (int k = 0; k < iters; ++k) {
        // 2 FLOPs per FMA
        z = fmaf(x, y, z);
        x = fmaf(y, z, x);
        y = fmaf(z, x, y);
    }
    c[i] = z;
}

extern "C" cudaError_t launch_fma_kernel(float* a, float* b, float* c,
                                         size_t n, int iters, float* ms_out) {
    cudaError_t err;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaDeviceSynchronize(); if (err) return err;
    cudaEventRecord(start);
    fma_kernel<<<grid, block>>>(a, b, c, n, iters);
    cudaEventRecord(stop);
    err = cudaGetLastError(); if (err) return err;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(ms_out, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return cudaSuccess;
}

