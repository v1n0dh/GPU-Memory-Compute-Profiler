#include "profiler.hpp"
#include <nvml.h>
#include <cstdio>
#include <thread>
#include <chrono>

static nvmlDevice_t g_dev;

NVMLSampler::NVMLSampler() {}
NVMLSampler::~NVMLSampler() {
    stop();
    nvmlShutdown();
}

bool NVMLSampler::init(int device_id) {
    nvmlReturn_t r = nvmlInit_v2();
    if (r != NVML_SUCCESS) {
        fprintf(stderr, "NVML init failed: %s\n", nvmlErrorString(r));
        return false;
    }
    unsigned count=0; nvmlDeviceGetCount_v2(&count);
    if ((unsigned)device_id >= count) {
        fprintf(stderr, "NVML: invalid device %d (count=%u)\n", device_id, count);
        return false;
    }
    if (nvmlDeviceGetHandleByIndex_v2(device_id, &g_dev) != NVML_SUCCESS) {
        fprintf(stderr, "NVML: get handle failed\n"); return false;
    }
    device_index_ = device_id;
    return true;
}

void NVMLSampler::start(int interval_ms) {
    running_ = true;
    samples_.clear();
    th_ = std::thread([this, interval_ms]{ loop(interval_ms); });
}

void NVMLSampler::stop() {
    if (running_) {
        running_ = false;
        if (th_.joinable()) th_.join();
    }
}

void NVMLSampler::loop(int interval_ms) {
    while (running_) {
        SampleNVML s{};
        s.timestamp_s = now_seconds();

        nvmlUtilization_t util{};
        nvmlDeviceGetUtilizationRates(g_dev, &util);
        s.gpu_util = util.gpu;
        s.mem_util = util.memory;

        unsigned long long mem_total=0, mem_free=0, mem_used=0;
        nvmlDeviceGetMemoryInfo(g_dev, & (nvmlMemory_t&) {mem_total, mem_free, mem_used});
        // The above line uses layout trick; safer split:
        nvmlMemory_t memInfo; nvmlDeviceGetMemoryInfo(g_dev, &memInfo);
        s.mem_total_mb = (unsigned)(memInfo.total / (1024*1024));
        s.mem_used_mb  = (unsigned)(memInfo.used  / (1024*1024));

        unsigned sm=0, mem=0;
        nvmlDeviceGetClockInfo(g_dev, NVML_CLOCK_SM, &sm);
        nvmlDeviceGetClockInfo(g_dev, NVML_CLOCK_MEM, &mem);
        s.sm_clock_mhz = sm;
        s.mem_clock_mhz = mem;

        unsigned tx=0, rx=0;
        nvmlDeviceGetPcieThroughput(g_dev, NVML_PCIE_UTIL_TX_BYTES, &tx);
        nvmlDeviceGetPcieThroughput(g_dev, NVML_PCIE_UTIL_RX_BYTES, &rx);
        s.pcie_tx_kb = tx;  // NVML returns KB/s
        s.pcie_rx_kb = rx;

        samples_.push_back(s);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}

