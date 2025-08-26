#!/usr/bin/python3

import os, json, time, subprocess, pathlib, statistics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""
Runs a synthetic ResNet18 training loop while the C++ profiler samples NVML + /proc.
Requires:
  - PyTorch with CUDA
  - Built ./gpuprof in project root
Usage:
  python python/pytorch_integration.py
Outputs:
  - out/report.json, out/nvml.csv, out/system.csv
  - prints per-step latency stats
"""

def run_profiler(duration_s=20, sample_ms=100, out_dir="out"):
    exe = str(pathlib.Path(__file__).resolve().parents[1] / "gpuprof")
    args = [exe, "--duration", str(duration_s), "--sample-ms", str(sample_ms), "--out", out_dir, "--sample-only"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def synthetic_loader(batch_size=128, num_batches=100, num_classes=10, image_size=224):
    x = torch.randn(num_batches * batch_size, 3, image_size, image_size)
    y = torch.randint(0, num_classes, (num_batches * batch_size,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def train_loop(device="cuda", steps=100):
    model = torch.hub.load('pytorch/vision', 'resnet18', weights=None)  # no internet weights needed
    model = model.to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    dl = synthetic_loader(num_batches=steps)

    step_times_ms = []
    comp_times_ms = []

    for i, (x, y) in enumerate(dl):
        if i >= steps: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        kstart = torch.cuda.Event(enable_timing=True)
        kend   = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()

        opt.zero_grad(set_to_none=True)
        kstart.record()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        kend.record()

        end.record()
        torch.cuda.synchronize()

        step_ms = start.elapsed_time(end)      # full step
        comp_ms = kstart.elapsed_time(kend)    # fwd+bwd+opt only
        step_times_ms.append(step_ms)
        comp_times_ms.append(comp_ms)

        if (i+1) % 10 == 0:
            print(f"[step {i+1}] step_ms={step_ms:.2f}, comp_ms={comp_ms:.2f}")

    return {
        "steps": steps,
        "step_times_ms": step_times_ms,
        "comp_times_ms": comp_times_ms,
        "step_ms_mean": statistics.mean(step_times_ms),
        "step_ms_p95": statistics.quantiles(step_times_ms, n=20)[18],
        "comp_ms_mean": statistics.mean(comp_times_ms),
        "comp_ms_p95": statistics.quantiles(comp_times_ms, n=20)[18],
    }

def main():
    os.makedirs("out", exist_ok=True)
    dur = 20
    prof = run_profiler(duration_s=dur, sample_ms=100, out_dir="out")
    try:
        metrics = train_loop(device="cuda", steps=100)
    finally:
        try:
            prof.wait(timeout=dur+5)
        except subprocess.TimeoutExpired:
            prof.kill()

    print("\n=== PyTorch Timing Summary ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        elif isinstance(v, list):
            print(f"{k}: {len(v)} values")
        else:
            print(f"{k}: {v}")

    # Merge PyTorch timing into report.json
    rpt_path = "out/report.json"
    if os.path.exists(rpt_path):
        with open(rpt_path, "r") as f:
            rpt = json.load(f)
    else:
        rpt = {}
    rpt["pytorch"] = metrics
    with open(rpt_path, "w") as f:
        json.dump(rpt, f, indent=2)
    print("\nCombined report at out/report.json")

if __name__ == "__main__":
    main()

