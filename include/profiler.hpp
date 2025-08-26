#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <functional>
#include <cstdint>

struct SampleNVML {
    double timestamp_s;
    unsigned gpu_util;            // %
    unsigned mem_util;            // %
    unsigned mem_used_mb;         // MB
    unsigned mem_total_mb;        // MB
    unsigned sm_clock_mhz;        // MHz
    unsigned mem_clock_mhz;       // MHz
    unsigned pcie_tx_kb;          // KB/s (instant)
    unsigned pcie_rx_kb;          // KB/s (instant)
};

struct SampleSystem {
    double timestamp_s;
    double cpu_util_pct;          // %
    uint64_t mem_total_kb;
    uint64_t mem_free_kb;
    double load_avg_1m;
    uint64_t net_rx_kb;           // cumulative KB
    uint64_t net_tx_kb;           // cumulative KB
};

struct MemBWResult {
    size_t bytes;
    double h2d_gbps;
    double d2h_gbps;
    double d2d_gbps;
};

struct ComputeResult {
    size_t elements;
    int iterations;
    double gflops;
};

struct ProfilerConfig {
    int device_id = 0;
    int sample_ms = 100;
    int duration_s = 20;
    std::string out_dir = "out";
    bool sample_only = false;     // if true, skip microbenchmarks
    size_t mb_test_size = 256;    // MB
    size_t compute_elems = 1 << 26; // ~67M floats (~256MB)
    int compute_iters = 50;
};

class NVMLSampler {
public:
    NVMLSampler();
    ~NVMLSampler();
    bool init(int device_id);
    void start(int interval_ms);
    void stop();
    const std::vector<SampleNVML>& samples() const { return samples_; }

private:
    std::atomic<bool> running_{false};
    std::thread th_;
    std::vector<SampleNVML> samples_;
    int device_index_ = 0;
    void loop(int interval_ms);
};

class SystemSampler {
public:
    void start(int interval_ms);
    void stop();
    const std::vector<SampleSystem>& samples() const { return samples_; }

private:
    std::atomic<bool> running_{false};
    std::thread th_;
    std::vector<SampleSystem> samples_;
    void loop(int interval_ms);
};

// helpers
double now_seconds();
bool ensure_dir(const std::string& path);
bool write_json_report(const std::string& path,
                       const ProfilerConfig& cfg,
                       const MemBWResult* mbw,
                       const ComputeResult* comp,
                       const std::vector<SampleNVML>& nvml,
                       const std::vector<SampleSystem>& sys);
bool write_csv_samples(const std::string& out_dir,
                       const std::vector<SampleNVML>& nvml,
                       const std::vector<SampleSystem>& sys);

// CUDA microbenchmarks
MemBWResult run_mem_bw_test(int device_id, size_t bytes);
ComputeResult run_compute_test(int device_id, size_t elems, int iters);

