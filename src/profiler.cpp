#include "profiler.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

extern "C" cudaError_t launch_fma_kernel(float*, float*, float*, size_t, int, float*);

double now_seconds() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    auto t = steady_clock::now();
    return duration<double>(t - t0).count();
}

bool ensure_dir(const std::string& p) {
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return !ec;
}

static std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (auto c : s) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"':  o << "\\\""; break;
            case '\n': o << "\\n";  break;
            default:   o << c;      break;
        }
    }
    return o.str();
}

bool write_json_report(const std::string& path,
                       const ProfilerConfig& cfg,
                       const MemBWResult* mbw,
                       const ComputeResult* comp,
                       const std::vector<SampleNVML>& nvml,
                       const std::vector<SampleSystem>& sys) {
    std::ofstream f(path);
    if (!f) return false;
    f << "{\n";
    f << "  \"config\": {\n";
    f << "    \"device_id\": " << cfg.device_id << ",\n";
    f << "    \"sample_ms\": " << cfg.sample_ms << ",\n";
    f << "    \"duration_s\": " << cfg.duration_s << ",\n";
    f << "    \"mb_test_size_MB\": " << cfg.mb_test_size << ",\n";
    f << "    \"compute_elems\": " << cfg.compute_elems << ",\n";
    f << "    \"compute_iters\": " << cfg.compute_iters << "\n";
    f << "  },\n";

    if (mbw) {
        f << "  \"mem_bw\": {\n";
        f << "    \"bytes\": " << mbw->bytes << ",\n";
        f << "    \"h2d_gbps\": " << std::fixed << std::setprecision(2) << mbw->h2d_gbps << ",\n";
        f << "    \"d2h_gbps\": " << mbw->d2h_gbps << ",\n";
        f << "    \"d2d_gbps\": " << mbw->d2d_gbps << "\n";
        f << "  },\n";
    }
    if (comp) {
        f << "  \"compute\": {\n";
        f << "    \"elements\": " << comp->elements << ",\n";
        f << "    \"iterations\": " << comp->iterations << ",\n";
        f << "    \"gflops\": " << std::fixed << std::setprecision(2) << comp->gflops << "\n";
        f << "  },\n";
    }
    f << "  \"nvml\": [\n";
    for (size_t i = 0; i < nvml.size(); ++i) {
        const auto& s = nvml[i];
        f << "    {\"t\":" << s.timestamp_s
          << ",\"gpu\":" << s.gpu_util
          << ",\"mem\":" << s.mem_util
          << ",\"mem_used_mb\":" << s.mem_used_mb
          << ",\"mem_total_mb\":" << s.mem_total_mb
          << ",\"sm_mhz\":" << s.sm_clock_mhz
          << ",\"mem_mhz\":" << s.mem_clock_mhz
          << ",\"pcie_tx_kb\":" << s.pcie_tx_kb
          << ",\"pcie_rx_kb\":" << s.pcie_rx_kb << "}";
        f << (i + 1 == nvml.size() ? "\n" : ",\n");
    }
    f << "  ],\n  \"system\": [\n";
    for (size_t i = 0; i < sys.size(); ++i) {
        const auto& s = sys[i];
        f << "    {\"t\":" << s.timestamp_s
          << ",\"cpu\":" << std::fixed << std::setprecision(1) << s.cpu_util_pct
          << ",\"mem_total_kb\":" << s.mem_total_kb
          << ",\"mem_free_kb\":" << s.mem_free_kb
          << ",\"load1\":" << s.load_avg_1m
          << ",\"net_rx_kb\":" << s.net_rx_kb
          << ",\"net_tx_kb\":" << s.net_tx_kb << "}";
        f << (i + 1 == sys.size() ? "\n" : ",\n");
    }
    f << "  ]\n}\n";
    return true;
}

bool write_csv_samples(const std::string& out_dir,
                       const std::vector<SampleNVML>& nvml,
                       const std::vector<SampleSystem>& sys) {
    std::ofstream f1(out_dir + "/nvml.csv");
    std::ofstream f2(out_dir + "/system.csv");
    if (!f1 || !f2) return false;
    f1 << "t_s,gpu_util,mem_util,mem_used_mb,mem_total_mb,sm_mhz,mem_mhz,pcie_tx_kb,pcie_rx_kb\n";
    for (const auto& s : nvml) {
        f1 << s.timestamp_s << "," << s.gpu_util << "," << s.mem_util << ","
           << s.mem_used_mb << "," << s.mem_total_mb << ","
           << s.sm_clock_mhz << "," << s.mem_clock_mhz << ","
           << s.pcie_tx_kb << "," << s.pcie_rx_kb << "\n";
    }
    f2 << "t_s,cpu_util_pct,mem_total_kb,mem_free_kb,load1,net_rx_kb,net_tx_kb\n";
    for (const auto& s : sys) {
        f2 << s.timestamp_s << "," << s.cpu_util_pct << ","
           << s.mem_total_kb << "," << s.mem_free_kb << ","
           << s.load_avg_1m << "," << s.net_rx_kb << "," << s.net_tx_kb << "\n";
    }
    return true;
}

// CUDA microbenchmarks
static double time_copy(size_t bytes, cudaMemcpyKind kind) {
    void *hbuf = nullptr, *dbuf = nullptr;
    cudaHostAlloc(&hbuf, bytes, cudaHostAllocDefault);
    cudaMalloc(&dbuf, bytes);
    cudaMemset(dbuf, 0, bytes);

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    cudaMemcpy(dbuf, hbuf, bytes, kind);               // H2D
    if (kind == cudaMemcpyDeviceToHost) cudaMemcpy(hbuf, dbuf, bytes, kind);
    if (kind == cudaMemcpyDeviceToDevice) cudaMemcpy(dbuf, dbuf, bytes, kind);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dbuf); cudaFreeHost(hbuf);
    return ms / 1e3; // seconds
}

MemBWResult run_mem_bw_test(int device_id, size_t bytes) {
    cudaSetDevice(device_id);
    MemBWResult r{};
    r.bytes = bytes;

    // H2D
    {
        void *h=nullptr,*d=nullptr;
        cudaHostAlloc(&h, bytes, cudaHostAllocDefault);
        cudaMalloc(&d, bytes);
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e);
        r.h2d_gbps = (bytes / (1024.0*1024.0*1024.0)) / (ms/1e3);
        cudaEventDestroy(s); cudaEventDestroy(e);
        cudaFree(d); cudaFreeHost(h);
    }
    // D2H
    {
        void *h=nullptr,*d=nullptr;
        cudaHostAlloc(&h, bytes, cudaHostAllocDefault);
        cudaMalloc(&d, bytes);
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e);
        r.d2h_gbps = (bytes / (1024.0*1024.0*1024.0)) / (ms/1e3);
        cudaEventDestroy(s); cudaEventDestroy(e);
        cudaFree(d); cudaFreeHost(h);
    }
    // D2D
    {
        void *d1=nullptr,*d2=nullptr;
        cudaMalloc(&d1, bytes); cudaMalloc(&d2, bytes);
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        cudaMemcpy(d2, d1, bytes, cudaMemcpyDeviceToDevice);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e);
        r.d2d_gbps = (bytes / (1024.0*1024.0*1024.0)) / (ms/1e3);
        cudaEventDestroy(s); cudaEventDestroy(e);
        cudaFree(d1); cudaFree(d2);
    }
    return r;
}

ComputeResult run_compute_test(int device_id, size_t elems, int iters) {
    cudaSetDevice(device_id);
    ComputeResult r{};
    r.elements = elems; r.iterations = iters;

    float *a=nullptr,*b=nullptr,*c=nullptr;
    cudaMalloc(&a, elems*sizeof(float));
    cudaMalloc(&b, elems*sizeof(float));
    cudaMalloc(&c, elems*sizeof(float));
    cudaMemset(a, 0x3f, elems*sizeof(float));
    cudaMemset(b, 0x3f, elems*sizeof(float));
    cudaMemset(c, 0,    elems*sizeof(float));

    float ms=0;
    auto err = launch_fma_kernel(a,b,c, elems, iters, &ms);
    if (err != cudaSuccess) {
        r.gflops = 0.0;
    } else {
        // Rough FLOP count: 2 FMAs x 3 per loop = 6 FLOPs per iteration per element
        double flops = double(elems) * double(iters) * 6.0;
        r.gflops = (flops / (ms/1e3)) / 1e9;
    }
    cudaFree(a); cudaFree(b); cudaFree(c);
    return r;
}

