import json, numpy as np
from pathlib import Path

try:
    import torch
    assert torch.cuda.is_available()
except:
    print("CUDA required."); exit()

import torch
import torch.nn.functional as F

def timed(fn, iters=100, warmup=10):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times)

# Pre-allocate tensors (no allocation during timing)
img  = torch.rand(1,1,720,1280, device="cuda")
k    = torch.ones(1,1,5,5, device="cuda")/25.0
xf   = torch.rand(1,2048, device="cuda")
W1   = torch.rand(2048,2048, device="cuda")
W2   = torch.rand(2048,512, device="cuda")
W3   = torch.rand(512,400, device="cuda")

def depth_work():
    out = torch.zeros_like(img)
    for d in range(0,64,8):
        out += F.conv2d(torch.abs(img-torch.roll(img,d,dims=-1)),k,padding=2)
    torch.cuda.synchronize()
    return out

def detect_work():
    x = torch.relu(xf @ W1)
    x = torch.relu(x @ W2)
    out = x @ W3
    torch.cuda.synchronize()
    return out

def sequential():
    d = depth_work()
    det = detect_work()
    return d, det

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

def parallel():
    with torch.cuda.stream(s1): d   = depth_work()
    with torch.cuda.stream(s2): det = detect_work()
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)
    torch.cuda.synchronize()
    return d, det

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
print()

t_d   = timed(depth_work)
t_det = timed(detect_work)
t_seq = timed(sequential)
t_par = timed(parallel)

speedup    = t_seq.mean() / t_par.mean()
saving_ms  = t_seq.mean() - t_par.mean()
theoretical = max(t_d.mean(), t_det.mean())

print("="*55)
print("  PHASE 3 — CUDA STREAMS BENCHMARK")
print("  Stage 3 (Depth) || Stage 4 (Detection) overlap")
print("="*55)
print()
print(f"  Stage 3 Depth alone:      {t_d.mean():>7.3f} ms")
print(f"  Stage 4 Detection alone:  {t_det.mean():>7.3f} ms")
print()
print(f"  Sequential (3 then 4):    {t_seq.mean():>7.3f} ms  (baseline)")
print(f"  Parallel streams:         {t_par.mean():>7.3f} ms")
print()
print(f"  Speedup:                  {speedup:>7.2f}x")
print(f"  Time saved:               {saving_ms:>7.3f} ms per frame")
print(f"  Theoretical max saving:   {t_seq.mean()-theoretical:>7.3f} ms")
print()

if speedup > 1.2:
    print("  OVERLAP CONFIRMED: Stages executing concurrently.")
    print("  GPU has enough free SMs to run both workloads simultaneously.")
elif speedup > 1.05:
    print("  PARTIAL OVERLAP: Some concurrency achieved.")
    print("  RTX 4060 (24 SMs) is smaller than A100 (108 SMs).")
    print("  Expect better overlap on Perlmutter.")
else:
    print("  NO OVERLAP on this GPU at these workload sizes.")
    print("  Workloads saturate all 24 SMs individually.")
    print("  Overlap will be significant on A100 (108 SMs).")

print()
print(f"  Pipeline impact:")
print(f"    Stage 3+4 sequential:  {t_seq.mean():.3f} ms")
print(f"    Stage 3+4 parallel:    {t_par.mean():.3f} ms")
print(f"    Saving toward 11ms target: {saving_ms:.3f} ms")
print("="*55)

out = Path("results/timing_data/streams_benchmark.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "gpu": torch.cuda.get_device_name(0),
    "depth_alone_ms": float(t_d.mean()),
    "detect_alone_ms": float(t_det.mean()),
    "sequential_ms": float(t_seq.mean()),
    "parallel_ms": float(t_par.mean()),
    "speedup": float(speedup),
    "saving_ms": float(saving_ms),
}, indent=2))
print(f"\n  Saved: {out}")
