#include "profiler.hpp"
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <unistd.h>

static bool read_cpu_times(uint64_t& idle, uint64_t& total) {
    std::ifstream f("/proc/stat");
    if (!f) return false;
    std::string line; std::getline(f, line);
    std::istringstream iss(line);
    std::string cpu; iss >> cpu;
    uint64_t user,nice,system,idle_t,iowait,irq,softirq,steal,guest,guest_nice;
    iss >> user >> nice >> system >> idle_t >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
    idle = idle_t + iowait;
    total = user+nice+system+idle_t+iowait+irq+softirq+steal+guest+guest_nice;
    return true;
}

static bool read_meminfo(uint64_t& total_kb, uint64_t& free_kb) {
    std::ifstream f("/proc/meminfo");
    if (!f) return false;
    std::string k; uint64_t v; std::string unit;
    total_kb = free_kb = 0;
    while (f >> k >> v >> unit) {
        if (k == "MemTotal:") total_kb = v;
        if (k == "MemAvailable:") free_kb = v; // better proxy than MemFree
    }
    return true;
}

static bool read_loadavg(double& l1) {
    std::ifstream f("/proc/loadavg");
    if (!f) return false;
    f >> l1;
    return true;
}

static bool read_net(uint64_t& rx_kb, uint64_t& tx_kb) {
    std::ifstream f("/proc/net/dev");
    if (!f) return false;
    std::string line;
    rx_kb = tx_kb = 0;
    // sum over all interfaces except lo
    for (int i=0; i<2; ++i) std::getline(f, line);
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string iface; iss >> iface;
        if (iface.find("lo:") != std::string::npos) continue;
        uint64_t rbytes, tbytes;
        char c;
        // iface: rbytes ...
        iss >> rbytes;
        for (int i=0;i<7;i++) iss >> c; // skip
        // transmit starts after 8th field
        for (int i=0;i<8;i++) iss >> c;
        iss >> tbytes;
        rx_kb += rbytes / 1024;
        tx_kb += tbytes / 1024;
    }
    return true;
}

void SystemSampler::start(int interval_ms) {
    running_ = true;
    samples_.clear();
    th_ = std::thread([this, interval_ms]{ loop(interval_ms); });
}

void SystemSampler::stop() {
    if (running_) {
        running_ = false;
        if (th_.joinable()) th_.join();
    }
}

void SystemSampler::loop(int interval_ms) {
    uint64_t idle0=0, total0=0;
    read_cpu_times(idle0, total0);
    uint64_t rx0=0, tx0=0; read_net(rx0, tx0);
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        uint64_t idle1=0, total1=0; read_cpu_times(idle1, total1);
        double cpu_util = 0.0;
        if (total1 > total0) {
            cpu_util = 100.0 * (1.0 - double(idle1 - idle0) / double(total1 - total0));
        }
        idle0 = idle1; total0 = total1;

        uint64_t total_kb=0, free_kb=0; read_meminfo(total_kb, free_kb);
        double l1=0; read_loadavg(l1);
        uint64_t rx=0, tx=0; read_net(rx, tx);

        SampleSystem s{};
        s.timestamp_s = now_seconds();
        s.cpu_util_pct = cpu_util;
        s.mem_total_kb = total_kb;
        s.mem_free_kb = free_kb;
        s.load_avg_1m = l1;
        s.net_rx_kb = rx;
        s.net_tx_kb = tx;
        samples_.push_back(s);
    }
}

