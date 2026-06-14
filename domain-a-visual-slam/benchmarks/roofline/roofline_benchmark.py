"""
Phase 3 — Roofline Analysis: Domain A SLAM Kernels

Computes Operational Intensity (FLOPs/byte) for every kernel in Stage 1+2,
measures achieved memory bandwidth, classifies each as memory-bound or
compute-bound, and projects A100 speedup.

Mirrors Domain B Phase 4 — same hardware ceilings, same methodology.

RTX 4060 Laptop ceilings (from Domain B):
  FP32 peak : 15,100 GFLOPS
  BW peak   : 272 GB/s
  Ridge pt  : 55.5 FLOPs/byte
"""

import time, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn.functional as F
    CUDA_OK = torch.cuda.is_available()
except ImportError:
    CUDA_OK = False

try:
    import nvtx
    _NVTX = True
except ImportError:
    _NVTX = False


# ---------------------------------------------------------------------------
# Hardware ceilings
# ---------------------------------------------------------------------------
RTX_FP32_GFLOPS = 15_100      # RTX 4060 Laptop FP32 peak
RTX_BW_GBS      = 272         # RTX 4060 Laptop GDDR6 peak
RTX_RIDGE       = RTX_FP32_GFLOPS / RTX_BW_GBS   # 55.5 FLOPs/byte

A100_FP32_GFLOPS = 19_500     # A100 SXM FP32 peak
A100_BW_GBS      = 2_039      # A100 SXM HBM2e peak  (Perlmutter spec)
A100_RIDGE       = A100_FP32_GFLOPS / A100_BW_GBS  # 9.6 FLOPs/byte

# Frame and problem dimensions
H, W       = 720, 1280
H1, W1     = H - 1, W - 1    # gradient output dims: 719 x 1279
N_FEAT     = 500
N_STATE    = 15
KERNEL_SZ  = 5


def setup_device():
    if not CUDA_OK:
        print("Device : CPU")
        return None, "cpu"
    props = torch.cuda.get_device_properties(0)
    print(f"Device : CUDA")
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1024**3:.1f} GB")
    print(f"SMs    : {props.multi_processor_count}")
    print(f"Torch  : {torch.__version__}")
    return torch, "cuda"


def sync():
    if CUDA_OK:
        torch.cuda.synchronize()


def time_kernel(fn, warmup=20, frames=100):
    for _ in range(warmup):
        fn()
    sync()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(frames):
        fn()
    ev1.record()
    sync()
    return ev0.elapsed_time(ev1) / frames


# ---------------------------------------------------------------------------
# Operational Intensity — analytical (FLOPs / bytes moved)
# ---------------------------------------------------------------------------

def oi_gradients():
    """Ix = left[:,1:] - left[:,:-1], Iy = left[1:,:] - left[:-1,:]"""
    flops = H1 * W1 * 2           # 2 subtractions per pixel, 2 directions
    bytes_io = (H * W              # read left_gray (float32)
                + H1 * W1 * 2     # write Ix, Iy
               ) * 4
    return flops, bytes_io

def oi_structure_tensor():
    """Ixx=Ix^2, Iyy=Iy^2, Ixy=Ix*Iy"""
    flops = H1 * W1 * 3           # 3 multiplications per pixel
    bytes_io = (H1 * W1 * 2       # read Ix, Iy
                + H1 * W1 * 3     # write Ixx, Iyy, Ixy
               ) * 4
    return flops, bytes_io

def oi_conv2d_harris():
    """5x5 box filter on Ixx, Iyy, Ixy (3 separate conv2d calls)"""
    flops = 3 * H1 * W1 * KERNEL_SZ * KERNEL_SZ * 2   # MACs x2 for fused mul+add
    bytes_io = (3 * H1 * W1       # read 3 input channels
                + 3 * H1 * W1     # write Sxx, Syy, Sxy
               ) * 4
    return flops, bytes_io

def oi_harris_response():
    """R = det - 0.05 * trace^2 = (Sxx*Syy - Sxy^2) - 0.05*(Sxx+Syy)^2"""
    flops = H1 * W1 * 6           # mul, mul, sub, add, mul, sub = 6 ops
    bytes_io = (H1 * W1 * 3       # read Sxx, Syy, Sxy
                + H1 * W1         # write R
               ) * 4
    return flops, bytes_io

def oi_topk():
    """torch.topk(R.reshape(-1), 500) — radix sort approximation"""
    n = H1 * W1
    flops = int(n * np.log2(n))   # radix sort ~ N log N comparisons
    bytes_io = n * 4 * 2          # read + write R flattened
    return flops, bytes_io

def oi_lk_flow():
    """Lucas-Kanade: It=left-prev, u=-(Ixx*It)/(Ixx+Iyy+eps), v=...   """
    flops = H1 * W1 * 7           # sub, mul, mul, add, add, div, div per pixel
    bytes_io = (H1 * W1 * 4       # read: left, prev, Ixx, Iyy
                + H1 * W1 * 2     # write: flow_u, flow_v
               ) * 4
    return flops, bytes_io

def oi_ekf_predict():
    """x = F@x, FP = F@P, FPFt = FP@F.T, P = FPFt + Q"""
    n = N_STATE
    flops = (2*n*n           # F@x (matmul N x N @ N)
             + 2*n*n*n       # F@P (N x N @ N x N)
             + 2*n*n*n       # FP@F.T
             + n*n)          # FPFt + Q (elementwise add)
    bytes_io = (n*n + n      # read F, x
                + n*n        # read P
                + n*n        # read Q
                + n          # write x
                + n*n        # write P
               ) * 4
    return flops, bytes_io

def oi_ekf_update():
    """HP=H@P, S=HPHt+R, K=PHt@inv(S), x+=Ky, P=(I-KH)@P"""
    n, m = N_STATE, 3
    flops = (2*m*n*n         # H@P (3x15 @ 15x15)
             + 2*m*n*m       # HP@H.T (3x15 @ 15x3)
             + m*m           # HPHt + R
             + 27            # inv(3x3) ~ 27 FLOPs
             + 2*n*m*m       # PHt @ inv(S) (15x3 @ 3x3)
             + 2*n*m         # K@y (15x3 @ 3)
             + n             # x += Ky
             + 2*n*m*n       # KH = K@H (15x3 @ 3x15)
             + n*n           # I - KH
             + 2*n*n*n)      # (I-KH) @ P
    bytes_io = (n*n + n*m + n*m + m*m  # read P, H, K, R
                + n + n*n              # write x, P
               ) * 4
    return flops, bytes_io


# ---------------------------------------------------------------------------
# Bandwidth measurement — achieved GB/s per stage
# ---------------------------------------------------------------------------

def measure_stage1_bw(torch_mod, frames=100):
    left  = torch_mod.zeros(H, W, device="cuda")
    prev  = torch_mod.zeros(H, W, device="cuda")
    k     = torch_mod.ones(1, 1, KERNEL_SZ, KERNEL_SZ, device="cuda") / (KERNEL_SZ**2)

    def run():
        Ix = left[:, 1:] - left[:, :-1]
        Iy = left[1:, :] - left[:-1, :]
        Ix = Ix[:H1, :W1]; Iy = Iy[:H1, :W1]
        Ixx = Ix**2; Iyy = Iy**2; Ixy = Ix*Iy
        Sxx = F.conv2d(Ixx.unsqueeze(0).unsqueeze(0), k, padding=2).squeeze()
        Syy = F.conv2d(Iyy.unsqueeze(0).unsqueeze(0), k, padding=2).squeeze()
        Sxy = F.conv2d(Ixy.unsqueeze(0).unsqueeze(0), k, padding=2).squeeze()
        det = Sxx*Syy - Sxy**2
        R   = det - 0.05*(Sxx+Syy)**2
        torch_mod.topk(R.reshape(-1), N_FEAT)
        It  = left[:H1, :W1] - prev[:H1, :W1]
        u   = -(Ixx*It)/(Ixx+Iyy+1e-6)
        v   = -(Iyy*It)/(Ixx+Iyy+1e-6)

    mean_ms = time_kernel(run, frames=frames)

    # Total bytes moved across all Stage 1 kernels
    total_bytes = sum([
        oi_gradients()[1],
        oi_structure_tensor()[1],
        oi_conv2d_harris()[1],
        oi_harris_response()[1],
        oi_topk()[1],
        oi_lk_flow()[1],
    ])
    achieved_bw = (total_bytes / 1e9) / (mean_ms / 1e3)
    return mean_ms, achieved_bw, total_bytes


def measure_stage2_bw(torch_mod, frames=100):
    x  = torch_mod.zeros(N_STATE, device="cuda")
    P  = torch_mod.eye(N_STATE, device="cuda") * 0.01
    F_ = torch_mod.eye(N_STATE, device="cuda")
    Q  = torch_mod.eye(N_STATE, device="cuda") * 1e-4
    H_ = torch_mod.zeros(3, N_STATE, device="cuda")
    R_ = torch_mod.eye(3, device="cuda") * 0.01
    z  = torch_mod.zeros(3, device="cuda")

    def run():
        F_[:3, 3:6] = torch_mod.eye(3, device="cuda") * 0.033
        xp   = F_ @ x
        FP   = torch_mod.matmul(F_, P)
        FPFt = torch_mod.matmul(FP, F_.t())
        Pp   = FPFt + Q
        y    = z - (H_ @ xp)
        HP   = torch_mod.matmul(H_, Pp)
        HPHt = torch_mod.matmul(HP, H_.t())
        S    = HPHt + R_
        PHt  = torch_mod.matmul(Pp, H_.t())
        K    = torch_mod.matmul(PHt, torch_mod.linalg.inv(S))
        xp.add_(K @ y)
        KH   = torch_mod.matmul(K, H_)
        IKH  = torch_mod.eye(N_STATE, device="cuda") - KH
        torch_mod.matmul(IKH, Pp)

    mean_ms = time_kernel(run, frames=frames)

    total_bytes = oi_ekf_predict()[1] + oi_ekf_update()[1]
    achieved_bw = (total_bytes / 1e9) / (mean_ms / 1e3)
    return mean_ms, achieved_bw, total_bytes


def roofline_bound(oi, hw="rtx"):
    """Returns 'MEMORY' or 'COMPUTE' and projected GFLOPS."""
    ridge = RTX_RIDGE if hw == "rtx" else A100_RIDGE
    bw    = RTX_BW_GBS if hw == "rtx" else A100_BW_GBS
    peak  = RTX_FP32_GFLOPS if hw == "rtx" else A100_FP32_GFLOPS
    if oi < ridge:
        return "MEMORY", oi * bw
    return "COMPUTE", peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  DOMAIN A - ROOFLINE ANALYSIS  (Phase 3)")
    print("  Operational Intensity + Achieved BW per SLAM kernel")
    print("=" * 70)
    print()

    torch_mod, mode = setup_device()
    print()

    if mode != "cuda":
        print("ERROR: CUDA required.")
        return

    print(f"  RTX 4060 Laptop  : FP32={RTX_FP32_GFLOPS:,} GFLOPS  BW={RTX_BW_GBS} GB/s  Ridge={RTX_RIDGE:.1f} FLOPs/byte")
    print(f"  A100 SXM (target): FP32={A100_FP32_GFLOPS:,} GFLOPS  BW={A100_BW_GBS:,} GB/s  Ridge={A100_RIDGE:.1f} FLOPs/byte")
    print()

    # --- Analytical OI table ---
    kernels = [
        ("S1: Gradients (Ix, Iy)",       *oi_gradients()),
        ("S1: Structure Tensor (Ixx...)", *oi_structure_tensor()),
        ("S1: Conv2d 5x5 (Harris smooth)",*oi_conv2d_harris()),
        ("S1: Harris Response R",         *oi_harris_response()),
        ("S1: TopK-500 (radixSort)",      *oi_topk()),
        ("S1: LK Optical Flow (u, v)",    *oi_lk_flow()),
        ("S2: EKF Predict (F@x, F@P@Ft)",*oi_ekf_predict()),
        ("S2: EKF Update (K, x, P)",      *oi_ekf_update()),
    ]

    print("=" * 70)
    print("  KERNEL ROOFLINE TABLE")
    print("=" * 70)
    print(f"  {'Kernel':<36} {'OI':>8} {'Bound':>8} {'RTX proj':>10} {'A100 proj':>10}")
    print(f"  {'-'*36} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    results = []
    for name, flops, bytes_io in kernels:
        oi   = flops / bytes_io
        bnd, proj_rtx  = roofline_bound(oi, "rtx")
        _,   proj_a100 = roofline_bound(oi, "a100")
        print(f"  {name:<36} {oi:>7.2f}f/B {bnd:>8} {proj_rtx:>8.0f}G  {proj_a100:>8.0f}G")
        results.append({
            "kernel": name, "flops": flops, "bytes": bytes_io,
            "oi": oi, "bound": bnd,
            "proj_rtx_gflops": proj_rtx, "proj_a100_gflops": proj_a100,
        })

    print()

    # --- Achieved bandwidth measurement ---
    print("=" * 70)
    print("  ACHIEVED BANDWIDTH (measured)")
    print("=" * 70)
    print(f"  Measuring Stage 1 ({args.frames} frames)...")
    s1_ms, s1_bw, s1_bytes = measure_stage1_bw(torch_mod, args.frames)
    s1_util = s1_bw / RTX_BW_GBS * 100

    print(f"  Measuring Stage 2 ({args.frames} frames)...")
    s2_ms, s2_bw, s2_bytes = measure_stage2_bw(torch_mod, args.frames)
    s2_util = s2_bw / RTX_BW_GBS * 100

    print()
    print(f"  {'Stage':<36} {'Mean ms':>8} {'BW (GB/s)':>10} {'% of peak':>10} {'Bound':>8}")
    print(f"  {'-'*36} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'Stage 1  Harris + LK Flow':<36} {s1_ms:>7.3f}ms {s1_bw:>9.1f}  {s1_util:>9.1f}%  MEMORY")
    print(f"  {'Stage 2  VIO / EKF':<36} {s2_ms:>7.3f}ms {s2_bw:>9.1f}  {s2_util:>9.1f}%  MEMORY")
    print()

    # --- A100 projection ---
    print("=" * 70)
    print("  A100 SPEEDUP PROJECTION (bandwidth scaling)")
    print("=" * 70)
    bw_ratio = A100_BW_GBS / RTX_BW_GBS
    s1_a100  = s1_ms / bw_ratio
    s2_a100  = s2_ms / bw_ratio
    print(f"  BW ratio  A100/RTX : {bw_ratio:.1f}x  ({A100_BW_GBS} / {RTX_BW_GBS} GB/s)")
    print()
    print(f"  {'Stage':<36} {'RTX 4060':>10} {'A100':>10} {'Speedup':>8}")
    print(f"  {'-'*36} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'Stage 1  Harris + LK Flow':<36} {s1_ms:>8.3f}ms {s1_a100:>9.3f}ms {bw_ratio:>7.1f}x")
    print(f"  {'Stage 2  VIO / EKF':<36} {s2_ms:>8.3f}ms {s2_a100:>9.3f}ms {bw_ratio:>7.1f}x")
    print(f"  {'TOTAL':<36} {s1_ms+s2_ms:>8.3f}ms {s1_a100+s2_a100:>9.3f}ms {(s1_ms+s2_ms)/(s1_a100+s2_a100):>7.1f}x")
    print()
    print("  Note: actual A100 speedup may differ — cuVSLAM replaces Stage 1+2")
    print("  with native C++ CUDA kernels (better memory access patterns, no JIT).")
    print("  This projection is a lower bound based on bandwidth ratio alone.")
    print("=" * 70)

    if args.save:
        out = {
            "timestamp":  datetime.now().isoformat(),
            "phase":      3,
            "device":     torch_mod.cuda.get_device_name(0),
            "frames":     args.frames,
            "ceilings": {
                "rtx_fp32_gflops": RTX_FP32_GFLOPS,
                "rtx_bw_gbs":      RTX_BW_GBS,
                "rtx_ridge":       RTX_RIDGE,
                "a100_fp32_gflops":A100_FP32_GFLOPS,
                "a100_bw_gbs":     A100_BW_GBS,
                "a100_ridge":      A100_RIDGE,
            },
            "kernel_oi":  results,
            "achieved": {
                "stage1_ms": s1_ms, "stage1_bw_gbs": s1_bw, "stage1_util_pct": s1_util,
                "stage2_ms": s2_ms, "stage2_bw_gbs": s2_bw, "stage2_util_pct": s2_util,
            },
            "a100_projection": {
                "bw_ratio":    bw_ratio,
                "stage1_a100_ms": s1_a100,
                "stage2_a100_ms": s2_a100,
                "total_rtx_ms":   s1_ms + s2_ms,
                "total_a100_ms":  s1_a100 + s2_a100,
                "speedup":        (s1_ms + s2_ms) / (s1_a100 + s2_a100),
            },
        }
        p = Path("results/timing_data/benchmark_phase3_roofline.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {p}")


if __name__ == "__main__":
    main()
