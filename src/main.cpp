#include "profiler.hpp"
#include <cuda_runtime.h>
#include <getopt.h>
#include <iostream>

static void usage(const char* prog) {
    std::cout <<
    "Usage: " << prog << " [options]\n"
    "Options:\n"
    "  --device N           CUDA device id (default 0)\n"
    "  --sample-ms N        Sampling interval ms (default 100)\n"
    "  --duration N         Duration seconds (default 20)\n"
    "  --out DIR            Output directory (default out)\n"
    "  --sample-only        Skip microbenchmarks, only sample\n"
    "  --mb-mb N            Memory BW test size in MB (default 256)\n"
    "  --comp-elems N       Compute elements (floats) (default 67,108,864)\n"
    "  --comp-iters N       Compute iterations per element (default 50)\n";
}

int main(int argc, char** argv) {
    ProfilerConfig cfg;
    static struct option opts[] = {
        {"device", required_argument, 0, 'd'},
        {"sample-ms", required_argument, 0, 's'},
        {"duration", required_argument, 0, 't'},
        {"out", required_argument, 0, 'o'},
        {"sample-only", no_argument, 0, 'S'},
        {"mb-mb", required_argument, 0, 'm'},
        {"comp-elems", required_argument, 0, 'e'},
        {"comp-iters", required_argument, 0, 'i'},
        {0,0,0,0}
    };
    int c;
    while ((c = getopt_long(argc, argv, "", opts, nullptr)) != -1) {
        switch (c) {
            case 'd': cfg.device_id = std::stoi(optarg); break;
            case 's': cfg.sample_ms = std::stoi(optarg); break;
            case 't': cfg.duration_s = std::stoi(optarg); break;
            case 'o': cfg.out_dir = optarg; break;
            case 'S': cfg.sample_only = true; break;
            case 'm': cfg.mb_test_size = std::stoul(optarg); break;
            case 'e': cfg.compute_elems = std::stoull(optarg); break;
            case 'i': cfg.compute_iters = std::stoi(optarg); break;
            default: usage(argv[0]); return 1;
        }
    }

    if (!ensure_dir(cfg.out_dir)) {
        std::cerr << "Failed to create out dir\n"; return 1;
    }

    cudaSetDevice(cfg.device_id);
    int devcount=0; cudaGetDeviceCount(&devcount);
    if (devcount == 0) { std::cerr << "No CUDA device\n"; return 1; }

    NVMLSampler nvml;
    if (!nvml.init(cfg.device_id)) {
        std::cerr << "NVML init failed\n"; return 1;
    }
    SystemSampler sys;

    nvml.start(cfg.sample_ms);
    sys.start(cfg.sample_ms);

    MemBWResult mbw{};
    ComputeResult comp{};

    if (!cfg.sample_only) {
        std::cout << "[*] Running memory bandwidth test...\n";
        mbw = run_mem_bw_test(cfg.device_id, cfg.mb_test_size * (size_t)1024 * 1024);
        std::cout << "    H2D: " << mbw.h2d_gbps << " GB/s, "
                  << "D2H: " << mbw.d2h_gbps << " GB/s, "
                  << "D2D: " << mbw.d2d_gbps << " GB/s\n";

        std::cout << "[*] Running compute test...\n";
        comp = run_compute_test(cfg.device_id, cfg.compute_elems, cfg.compute_iters);
        std::cout << "    Compute: " << comp.gflops << " GFLOP/s\n";
    }

    // sampling window
    int elapsed = 0;
    while (elapsed < cfg.duration_s) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        elapsed++;
    }

    nvml.stop();
    sys.stop();

    write_csv_samples(cfg.out_dir, nvml.samples(), sys.samples());
    write_json_report(cfg.out_dir + "/report.json", cfg,
                      cfg.sample_only ? nullptr : &mbw,
                      cfg.sample_only ? nullptr : &comp,
                      nvml.samples(), sys.samples());

    std::cout << "[*] Reports written to: " << cfg.out_dir << "\n";
    return 0;
}

