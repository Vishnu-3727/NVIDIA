import json, numpy as np
from pathlib import Path

try:
    import torch
    assert torch.cuda.is_available()
except:
    print("CUDA required."); exit()

import torch
import torch.nn.functional as F

props = torch.cuda.get_device_properties(0)
print(f"GPU : {props.name}")
print(f"SMs : {props.multi_processor_count}")
print()

# RTX 4060 Ada Lovelace hardware ceilings
# These are the lines on your roofline plot
PEAK_FP32_GFLOPS = 15100.0   # 15.1 TFLOPS
PEAK_BW_GBS      = 272.0     # 272 GB/s GDDR6
RIDGE_POINT      = PEAK_FP32_GFLOPS / PEAK_BW_GBS  # ~55.5 FLOPs/byte

def timed(fn, iters=200, warmup=20):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.mean(times))

print("="*65)
print("  PHASE 4A — ROOFLINE SWEEP")
print("  Operational Intensity vs Achieved Performance")
print(f"  Ridge point: {RIDGE_POINT:.1f} FLOPs/byte")
print(f"  Below ridge = memory-bound. Above = compute-bound.")
print("="*65)
print()
print(f"  {'N':>6} {'FLOPs':>14} {'Bytes':>12} {'OI':>8} {'Bound':>10} "
      f"{'ms':>8} {'GFLOPS':>10}")
print(f"  {'-'*6} {'-'*14} {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")

roofline = []
for N in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    fn = lambda: torch.matmul(A, B)
    t_ms   = timed(fn)
    flops  = 2 * N**3
    nbytes = 3 * N**2 * 4
    oi     = flops / nbytes
    gflops = (flops / 1e9) / (t_ms / 1e3)
    bound  = "COMPUTE" if oi > RIDGE_POINT else "MEMORY"
    pct    = min(gflops / PEAK_FP32_GFLOPS * 100, 100)
    print(f"  {N:>6} {flops:>14,} {nbytes:>12,} {oi:>8.1f} {bound:>10} "
          f"{t_ms:>8.3f} {gflops:>10.1f}  ({pct:.0f}% of peak)")
    roofline.append({"N":N,"flops":flops,"bytes":nbytes,"oi":oi,
                     "bound":bound,"ms":t_ms,"gflops":gflops})

print()
print("  Pipeline stage OI estimates (theoretical):")
print(f"  Stage 1 Feature Extraction : ~0.3 FLOPs/byte  → MEMORY-BOUND")
print(f"  Stage 2 VIO EKF (15x15)    : ~0.3 FLOPs/byte  → MEMORY-BOUND (tiny matrix)")
print(f"  Stage 3 Depth Estimation   : ~2-4 FLOPs/byte  → MEMORY-BOUND (conv)")
print(f"  Stage 4 Obstacle Detection : ~{2*2048*2048/(3*2048*2048*4):.1f} FLOPs/byte → depends on batch")
print(f"  Stage 5 Path Planning      : ~0.1 FLOPs/byte  → MEMORY-BOUND (scatter)")
print()

print("="*65)
print("  PHASE 4B — OCCUPANCY IMPACT")
print("  Effective throughput vs work chunk size")
print("="*65)
print()

N_ELEM = 16 * 1024 * 1024
a = torch.randn(N_ELEM, device="cuda")
b = torch.randn(N_ELEM, device="cuda")

print(f"  Vector size: {N_ELEM:,} elements ({N_ELEM*4/1024**2:.0f} MB)")
print()
print(f"  {'Chunk size':>12} {'Time (ms)':>12} {'GB/s':>10} {'% peak BW':>12}")
print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

occ_results = []
for chunk in [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, N_ELEM]:
    def fn():
        out = torch.zeros(N_ELEM, device="cuda")
        for start in range(0, N_ELEM, chunk):
            end = min(start+chunk, N_ELEM)
            out[start:end] = a[start:end] + b[start:end]
        torch.cuda.synchronize()
        return out
    t_ms = timed(fn, iters=50, warmup=5)
    nbytes = N_ELEM * 3 * 4
    bw = (nbytes/1e9)/(t_ms/1e3)
    pct = bw/PEAK_BW_GBS*100
    print(f"  {chunk:>12,} {t_ms:>12.3f} {bw:>10.1f} {pct:>11.1f}%")
    occ_results.append({"chunk":chunk,"ms":t_ms,"bw_gbs":bw,"pct_peak":pct})

print()
print("  Interpretation:")
print("  Small chunks → many kernel launches → overhead dominates → low BW")
print("  Large chunks → fewer launches → SM fully occupied → approaches peak BW")
print("  Stage 5 path planning uses scatter/gather on small chunks → memory-bound")
print()

print("="*65)
print("  PHASE 4C — PIPELINE STAGE CLASSIFICATION")
print("  Based on workload characteristics and roofline sweep")
print("="*65)
print()

img  = torch.rand(1,1,720,1280, device="cuda")
k    = torch.ones(1,1,5,5, device="cuda")/25.0
left = torch.rand(3,720,1280, device="cuda")
xf   = torch.rand(1,2048, device="cuda")
W1   = torch.rand(2048,2048, device="cuda")
W2   = torch.rand(2048,512, device="cuda")
W3   = torch.rand(512,400, device="cuda")
samp = torch.rand(10000,3, device="cuda")
occ  = torch.rand(250,250, device="cuda")

def s1_fn():
    Ix = left[:,:,1:]-left[:,:,:-1]
    Iy = left[:,1:,:]-left[:,:-1,:]
    H,W=left.shape[1],left.shape[2]
    R=(Ix[:,:H-1,:W-1]**2)*(Iy[:,:H-1,:W-1]**2)
    torch.cuda.synchronize()

def s2_fn():
    F=torch.randn(15,15,device="cuda")
    cov=torch.eye(15,device="cuda")
    _=torch.matmul(F,torch.matmul(cov,F.t()))
    torch.cuda.synchronize()

def s3_fn():
    out=torch.zeros_like(img)
    for d in range(0,64,8):
        out+=F.conv2d(torch.abs(img-torch.roll(img,d,dims=-1)),k,padding=2)
    torch.cuda.synchronize()

def s4_fn():
    x=torch.relu(xf@W1); x=torch.relu(x@W2); _=x@W3
    torch.cuda.synchronize()

def s5_fn():
    samp.uniform_(); samp[:,:2].mul_(250); samp[:,2].mul_(50)
    xi=samp[:,0].long().clamp_(0,249)
    yi=samp[:,1].long().clamp_(0,249)
    _=samp[occ[xi,yi]<0.5]
    torch.cuda.synchronize()

stages = [
    ("Stage 1 Feature Extraction", s1_fn,
     2*719*1279*3, 3*720*1280*4*2, "elementwise"),
    ("Stage 2 VIO EKF (15x15)",    s2_fn,
     2*15**3,      3*15*15*4,      "tiny matmul"),
    ("Stage 3 Depth Estimation",   s3_fn,
     8*720*1280*25*2, 8*720*1280*4*3, "sliding window"),
    ("Stage 4 Obstacle Detection", s4_fn,
     2*(2048*2048+2048*512+512*400), 4*(2048*2048+2048*512+512*400)*3, "dense matmul"),
    ("Stage 5 Path Planning",      s5_fn,
     10000*3, 10000*3*4*3, "scatter/gather"),
]

print(f"  {'Stage':<32} {'ms':>7} {'OI':>8} {'Bound':>10}  Action")
print(f"  {'-'*32} {'-'*7} {'-'*8} {'-'*10}  {'-'*30}")

stage_results = []
for name, fn, flops, nbytes, kind in stages:
    t_ms  = timed(fn, iters=100, warmup=10)
    oi    = flops / nbytes if nbytes > 0 else 0
    bound = "COMPUTE" if oi > RIDGE_POINT else "MEMORY"
    if bound == "MEMORY":
        action = "Improve mem coalescing / reduce transfers"
    else:
        action = "Increase occupancy / use tensor cores"
    print(f"  {name:<32} {t_ms:>7.3f} {oi:>8.2f} {bound:>10}  {action}")
    stage_results.append({"stage":name,"ms":t_ms,"oi":oi,"bound":bound})

print()
print("  NOTE: OI values are approximations from workload analysis.")
print("  Exact values require Nsight Compute per-kernel measurement.")
print("  Run: ncu --set full -o profiling\\nsight_reports\\phase4_ncu")
print("       uv run python benchmarks\\pipeline\\pipeline_benchmark.py --frames 3")
print()
print("="*65)

out = Path("results/timing_data/occupancy_benchmark.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "gpu": props.name,
    "peak_fp32_gflops": PEAK_FP32_GFLOPS,
    "peak_bw_gbs": PEAK_BW_GBS,
    "ridge_point": RIDGE_POINT,
    "roofline_sweep": roofline,
    "occupancy_sweep": occ_results,
    "stage_classification": stage_results,
}, indent=2))
print(f"  Saved: {out}")
