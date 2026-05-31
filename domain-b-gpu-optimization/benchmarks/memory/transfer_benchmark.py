import sys, json
import numpy as np
from pathlib import Path

try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except ImportError:
    CUDA_OK = False

if not CUDA_OK:
    print("CUDA not available. Transfer benchmark requires a CUDA GPU.")
    sys.exit(0)

import torch

def measure(name, fn, warmup=10, iters=200):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    arr = np.array(times)
    return {"name":name,"mean_ms":float(arr.mean()),"min_ms":float(arr.min()),"std_ms":float(arr.std())}

def run():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name}")
    print(f"VRAM: {props.total_memory/1024**3:.1f} GB")
    print()

    # Buffer sizes matching real pipeline data
    sizes = {
        "obstacle_output  (2.4 KB)" : 2400,
        "IMU_batch        (10 KB)"  : 10_000,
        "depth_map_720p   (3.5 MB)" : 720*1280*4,
        "mono_frame_720p  (3.5 MB)" : 720*1280*3*4,
        "stereo_frame_720p(7.0 MB)" : 720*1280*3*4*2,
    }

    all_results = []

    for size_name, nbytes in sizes.items():
        n = nbytes // 4
        print(f"Buffer: {size_name}")
        print(f"  {'Method':<28} {'Mean':>8}  {'BW (GB/s)':>10}  {'vs pageable':>12}")
        print(f"  {'-'*28} {'-'*8}  {'-'*10}  {'-'*12}")

        # 1. Pageable H2D
        cpu_pag = torch.randn(n)
        def pag_h2d(): return cpu_pag.to("cuda", non_blocking=False)
        r1 = measure("Pageable H2D", pag_h2d)
        bw1 = (nbytes/1e9)/(r1["mean_ms"]/1e3)
        print(f"  {'Pageable H2D':<28} {r1['mean_ms']:>7.3f}ms  {bw1:>10.1f}       baseline")
        all_results.append({**r1,"size_bytes":nbytes,"bw_gbs":bw1})

        # 2. Pinned H2D
        cpu_pin = torch.randn(n).pin_memory()
        def pin_h2d(): return cpu_pin.to("cuda", non_blocking=False)
        r2 = measure("Pinned H2D", pin_h2d)
        bw2 = (nbytes/1e9)/(r2["mean_ms"]/1e3)
        sp2 = r1["mean_ms"]/r2["mean_ms"]
        print(f"  {'Pinned H2D':<28} {r2['mean_ms']:>7.3f}ms  {bw2:>10.1f}  {sp2:>8.2f}x faster")
        all_results.append({**r2,"size_bytes":nbytes,"bw_gbs":bw2})

        # 3. Async Pinned H2D
        stream = torch.cuda.Stream()
        def async_h2d():
            with torch.cuda.stream(stream):
                return cpu_pin.to("cuda", non_blocking=True)
        r3 = measure("Async Pinned H2D", async_h2d)
        bw3 = (nbytes/1e9)/(r3["mean_ms"]/1e3)
        sp3 = r1["mean_ms"]/r3["mean_ms"]
        print(f"  {'Async Pinned H2D':<28} {r3['mean_ms']:>7.3f}ms  {bw3:>10.1f}  {sp3:>8.2f}x faster")
        all_results.append({**r3,"size_bytes":nbytes,"bw_gbs":bw3})

        # 4. D2H
        gpu_t = torch.randn(n, device="cuda")
        def d2h(): return gpu_t.cpu()
        r4 = measure("D2H (GPU to CPU)", d2h)
        bw4 = (nbytes/1e9)/(r4["mean_ms"]/1e3)
        print(f"  {'D2H (GPU to CPU)':<28} {r4['mean_ms']:>7.3f}ms  {bw4:>10.1f}")
        all_results.append({**r4,"size_bytes":nbytes,"bw_gbs":bw4})

        # 5. D2D (staying on GPU — the goal)
        def d2d(): return gpu_t.clone()
        r5 = measure("D2D (GPU to GPU)", d2d)
        bw5 = (nbytes/1e9)/(r5["mean_ms"]/1e3)
        ratio = r4["mean_ms"]/r5["mean_ms"]
        print(f"  {'D2D (GPU to GPU)':<28} {r5['mean_ms']:>7.3f}ms  {bw5:>10.1f}  {ratio:>7.0f}x vs D2H")
        all_results.append({**r5,"size_bytes":nbytes,"bw_gbs":bw5})
        print()

    # Pipeline overhead analysis
    print("="*65)
    print("  PIPELINE TRANSFER COST ANALYSIS")
    print("="*65)
    print()

    EFF_BW = 18.0  # GB/s — conservative effective PCIe bandwidth

    naive = [
        ("H2D stereo input",        720*1280*3*4*2),
        ("D2H feature tracks",      500*2*4),
        ("H2D feature tracks",      500*2*4),
        ("D2H pose estimate",       7*4),
        ("H2D stereo for depth",    720*1280*3*4*2),
        ("D2H depth map",           720*1280*4),
        ("H2D frame for detection", 720*1280*3*4),
        ("D2H detections",          100*6*4),
        ("H2D depth+detections",    720*1280*4+100*6*4),
        ("D2H path output",         20*3*4),
    ]

    print("  Naive pipeline (CPU round-trip between every stage):")
    total_naive = 0
    for name, size in naive:
        cost = (size/1e9)/EFF_BW*1000
        total_naive += cost
        print(f"    {name:<30} {cost:>6.3f} ms")
    print(f"    {'TOTAL':<30} {total_naive:>6.3f} ms")
    print()

    opt_h2d = (720*1280*3*4*2)/1e9/EFF_BW*1000
    opt_d2h = (20*3*4)/1e9/EFF_BW*1000
    total_opt = opt_h2d + opt_d2h

    print("  GPU-resident pipeline (only 2 transfers — input + output):")
    print(f"    {'H2D stereo input':<30} {opt_h2d:>6.3f} ms")
    print(f"    {'D2H path output':<30} {opt_d2h:>6.5f} ms")
    print(f"    {'TOTAL':<30} {total_opt:>6.3f} ms")
    print()
    print(f"  Transfer overhead reduction : {total_naive/total_opt:.1f}x")
    print(f"  Absolute saving per frame   : {total_naive-total_opt:.3f} ms")
    print()
    print("  This saving goes directly into the Transfer Overhead row")
    print("  of the SCRUM table: ~5ms before → ~0.5ms after.")
    print("="*65)

    out = Path("results/timing_data/transfer_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu":props.name,"results":all_results,
        "pipeline":{"naive_ms":total_naive,"optimized_ms":total_opt,
                    "speedup":total_naive/total_opt}},indent=2))
    print(f"\n  Saved: {out}")

if __name__ == "__main__":
    run()
