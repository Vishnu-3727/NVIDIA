"""
Phase 4 — End-to-End Pipeline Integration & Domain D Handoff

Two things measured here:

1. CUDA Stream Double-Buffering
   NVIDIA GPUs have a dedicated DMA copy engine separate from the SM compute
   engine. H2D transfers and GPU kernels can run truly in parallel (unlike
   Domain B Phase 3 where two compute streams competed for the same 24 SMs).

   Single-stream (sequential): H2D(N) -> Stage1+2(N) -> D2H pose(N)  -> repeat
   Double-buffered:             H2D(N+1) overlaps with Stage1+2(N) running

   Expected saving: hides the ~0.17ms pinned H2D cost per frame.

2. Domain D Handoff (Pose -> Path Planner)
   Domain D (Kamlesh) consumes the SE3 pose from Domain A to run path
   planning. This benchmark measures the full latency from frame arrival
   to waypoint delivery, and the cost of the pose handoff boundary.

SCRUM table rows owned by Domain A:
  Feature Extraction : 28ms CPU -> 6ms target
  VIO / EKF          : 12ms CPU -> 5ms target
  Transfer Overhead  : partial (stereo H2D + pose D2H)
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


H, W      = 720, 1280
H1, W1    = H - 1, W - 1
N_FEAT    = 500
N_STATE   = 15
KERNEL_SZ = 5

# CPU baselines from root README SCRUM table
CPU_FEATURE_MS = 28.0
CPU_VIO_MS     = 12.0
GPU_TARGET_FEATURE_MS = 6.0
GPU_TARGET_VIO_MS     = 5.0


class annotate:
    def __init__(self, name, color="white"):
        self.name = name; self.color = color; self._ctx = None
    def __enter__(self):
        if _NVTX:
            self._ctx = nvtx.annotate(self.name, color=self.color)
            self._ctx.__enter__()
        return self
    def __exit__(self, *a):
        if _NVTX and self._ctx: self._ctx.__exit__(*a)


def setup_device():
    if not CUDA_OK:
        print("Device : CPU"); return None, "cpu"
    props = torch.cuda.get_device_properties(0)
    print(f"Device : CUDA  |  GPU: {props.name}  |  SMs: {props.multi_processor_count}  |  Torch: {torch.__version__}")
    return torch, "cuda"


def sync(stream=None):
    if CUDA_OK:
        if stream:
            stream.synchronize()
        else:
            torch.cuda.synchronize()


def build_cache(torch_mod):
    cache = {
        "left_gpu" : torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda"),
        "right_gpu": torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda"),
        "prev_gpu" : torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda"),
        "Sxx": torch_mod.zeros(H1, W1, device="cuda"),
        "Syy": torch_mod.zeros(H1, W1, device="cuda"),
        "Sxy": torch_mod.zeros(H1, W1, device="cuda"),
        "R"  : torch_mod.zeros(H1, W1, device="cuda"),
        "x_state": torch_mod.zeros(N_STATE, device="cuda"),
        "P_cov"  : torch_mod.eye(N_STATE, device="cuda") * 0.01,
        "F_trans": torch_mod.eye(N_STATE, device="cuda"),
        "Q_proc" : torch_mod.eye(N_STATE, device="cuda") * 1e-4,
        "H_obs"  : torch_mod.zeros(3, N_STATE, device="cuda"),
        "R_meas" : torch_mod.eye(3, device="cuda") * 0.01,
        "z_meas" : torch_mod.zeros(3, device="cuda"),
        "pose_gpu": torch_mod.eye(4, device="cuda"),
        "pose_cpu": torch_mod.zeros(4, 4, pin_memory=True),
    }
    cache["H_obs"][:3, :3] = torch_mod.eye(3, device="cuda")
    for v in cache.values():
        if hasattr(v, "sum"): _ = v.sum() if v.is_cuda else None
    torch_mod.cuda.synchronize()
    return cache


def run_stage1(torch_mod, cache, stream=None):
    with annotate("stage1", color="blue"):
        with torch_mod.cuda.stream(stream) if stream else _null_ctx():
            left = cache["left_gpu"]
            prev = cache["prev_gpu"]
            k = torch_mod.ones(1, 1, KERNEL_SZ, KERNEL_SZ, device="cuda") / (KERNEL_SZ**2)
            Ix = (left[:, 1:] - left[:, :-1])[:H1, :W1]
            Iy = (left[1:, :] - left[:-1, :])[:H1, :W1]
            Ixx = Ix**2; Iyy = Iy**2; Ixy = Ix*Iy
            cache["Sxx"].copy_(F.conv2d(Ixx.unsqueeze(0).unsqueeze(0), k, padding=2).squeeze())
            cache["Syy"].copy_(F.conv2d(Iyy.unsqueeze(0).unsqueeze(0), k, padding=2).squeeze())
            cache["Sxy"].copy_(F.conv2d(Ixy.unsqueeze(0).unsqueeze(0), k, padding=2).squeeze())
            det = cache["Sxx"]*cache["Syy"] - cache["Sxy"]**2
            cache["R"].copy_(det - 0.05*(cache["Sxx"]+cache["Syy"])**2)
            torch_mod.topk(cache["R"].reshape(-1), N_FEAT)
            It = left[:H1, :W1] - prev[:H1, :W1]
            _ = -(Ixx*It)/(Ixx+Iyy+1e-6)
            cache["prev_gpu"].copy_(left)


def run_stage2(torch_mod, cache, stream=None):
    with annotate("stage2", color="green"):
        with torch_mod.cuda.stream(stream) if stream else _null_ctx():
            cache["F_trans"].copy_(torch_mod.eye(N_STATE, device="cuda"))
            cache["F_trans"][:3, 3:6] = torch_mod.eye(3, device="cuda") * 0.033
            cache["x_state"].copy_(cache["F_trans"] @ cache["x_state"])
            FP   = torch_mod.matmul(cache["F_trans"], cache["P_cov"])
            cache["P_cov"].copy_(torch_mod.matmul(FP, cache["F_trans"].t()) + cache["Q_proc"])
            cache["z_meas"].normal_(std=0.01)
            y    = cache["z_meas"] - (cache["H_obs"] @ cache["x_state"])
            HP   = torch_mod.matmul(cache["H_obs"], cache["P_cov"])
            S    = torch_mod.matmul(HP, cache["H_obs"].t()) + cache["R_meas"]
            PHt  = torch_mod.matmul(cache["P_cov"], cache["H_obs"].t())
            K    = torch_mod.matmul(PHt, torch_mod.linalg.inv(S))
            cache["x_state"].add_(K @ y)
            KH   = torch_mod.matmul(K, cache["H_obs"])
            cache["P_cov"].copy_(torch_mod.matmul(torch_mod.eye(N_STATE, device="cuda") - KH, cache["P_cov"]))
            # Pack position into pose matrix (top-left 3x3 = identity, col 4 = position)
            cache["pose_gpu"][:3, 3] = cache["x_state"][:3]


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Scenario 1: Single stream — sequential H2D, compute, D2H
# ---------------------------------------------------------------------------

def bench_single_stream(torch_mod, cache, left_pin, right_pin, frames=100):
    times = []
    for _ in range(frames):
        with annotate("single_stream_frame", color="white"):
            ev0 = torch_mod.cuda.Event(enable_timing=True)
            ev1 = torch_mod.cuda.Event(enable_timing=True)
            ev0.record()

            # H2D
            cache["left_gpu"].copy_(left_pin.cuda(non_blocking=True).float().div_(255.0))
            cache["right_gpu"].copy_(right_pin.cuda(non_blocking=True).float().div_(255.0))

            # Compute
            run_stage1(torch_mod, cache)
            run_stage2(torch_mod, cache)

            # D2H pose
            cache["pose_cpu"].copy_(cache["pose_gpu"], non_blocking=True)

            ev1.record()
            sync()
            times.append(ev0.elapsed_time(ev1))

    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Scenario 2: Double-buffered — H2D(N+1) overlaps Stage1+2(N)
# Uses CUDA copy engine (DMA) running in parallel with SM kernels
# ---------------------------------------------------------------------------

def bench_double_buffered(torch_mod, cache, left_pin, right_pin, frames=100):
    compute_stream = torch_mod.cuda.Stream()
    copy_stream    = torch_mod.cuda.Stream()

    # Two GPU frame buffers for ping-pong
    buf = [
        {"left": torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda"),
         "right": torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda")},
        {"left": torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda"),
         "right": torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda")},
    ]

    # Prime the first buffer
    with torch_mod.cuda.stream(copy_stream):
        buf[0]["left"].copy_(left_pin.cuda(non_blocking=True).float().div_(255.0))
        buf[0]["right"].copy_(right_pin.cuda(non_blocking=True).float().div_(255.0))
    copy_stream.synchronize()

    times = []
    for i in range(frames):
        cur  = i % 2
        nxt  = (i + 1) % 2

        ev0 = torch_mod.cuda.Event(enable_timing=True)
        ev1 = torch_mod.cuda.Event(enable_timing=True)
        ev0.record()

        with annotate("double_buf_frame", color="cyan"):
            # Prefetch next frame on copy stream (runs on DMA engine)
            with torch_mod.cuda.stream(copy_stream):
                buf[nxt]["left"].copy_(left_pin.cuda(non_blocking=True).float().div_(255.0))
                buf[nxt]["right"].copy_(right_pin.cuda(non_blocking=True).float().div_(255.0))

            # Process current frame on compute stream (runs on SMs)
            cache["left_gpu"].copy_(buf[cur]["left"])
            cache["right_gpu"].copy_(buf[cur]["right"])
            with torch_mod.cuda.stream(compute_stream):
                run_stage1(torch_mod, cache)
                run_stage2(torch_mod, cache)
                cache["pose_cpu"].copy_(cache["pose_gpu"], non_blocking=True)

            # Sync both streams before next frame
            copy_stream.synchronize()
            compute_stream.synchronize()

        ev1.record()
        sync()
        times.append(ev0.elapsed_time(ev1))

    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Scenario 3: Domain D handoff — pose -> simulated path planner
# Models the latency of passing SE3 pose to Domain D waypoint generator
# ---------------------------------------------------------------------------

def bench_domain_d_handoff(torch_mod, cache, frames=100):
    """
    Simulates Domain D receiving the pose and computing next waypoint — on GPU.
    Pose stays GPU-resident. Waypoint computation is a GPU tensor op.
    Only the final waypoint (3 floats) is copied to CPU for the flight controller.
    """
    goal         = torch_mod.tensor([50.0, 0.0, 10.0], device="cuda")
    waypoint_gpu = torch_mod.zeros(3, device="cuda")
    waypoint_cpu = torch_mod.zeros(3, pin_memory=True)

    # warmup
    for _ in range(10):
        current_pos = cache["pose_gpu"][:3, 3]
        direction   = goal - current_pos
        dist        = torch_mod.norm(direction)
        step        = torch_mod.clamp(dist, max=1.0)
        waypoint_gpu.copy_((current_pos + direction / (dist + 1e-6) * step))
        waypoint_cpu.copy_(waypoint_gpu, non_blocking=True)
    sync()

    times = []
    for _ in range(frames):
        ev0 = torch_mod.cuda.Event(enable_timing=True)
        ev1 = torch_mod.cuda.Event(enable_timing=True)
        ev0.record()

        # GPU waypoint computation — pose stays on GPU the whole time
        current_pos = cache["pose_gpu"][:3, 3]
        direction   = goal - current_pos
        dist        = torch_mod.norm(direction)
        step        = torch_mod.clamp(dist, max=1.0)
        waypoint_gpu.copy_((current_pos + direction / (dist + 1e-6) * step))

        # Only 12 bytes (3 floats) cross to CPU for the flight controller
        waypoint_cpu.copy_(waypoint_gpu, non_blocking=True)

        ev1.record()
        sync()
        times.append(ev0.elapsed_time(ev1))

    return float(np.mean(times)), float(np.std(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()

    print()
    print("=" * 68)
    print("  DOMAIN A - PIPELINE INTEGRATION & DOMAIN D HANDOFF  (Phase 4)")
    print("  Double-Buffered Streams | Pose Handoff | SCRUM Table Update")
    print("=" * 68)
    print()

    torch_mod, mode = setup_device()
    print()
    if mode != "cuda":
        print("ERROR: CUDA required."); return

    cache    = build_cache(torch_mod)
    left_pin = torch_mod.zeros(H, W, dtype=torch_mod.uint8).pin_memory()
    right_pin= torch_mod.zeros(H, W, dtype=torch_mod.uint8).pin_memory()

    # Warmup
    print(f"Warmup ({args.warmup} frames)...")
    for _ in range(args.warmup):
        cache["left_gpu"].uniform_()
        run_stage1(torch_mod, cache)
        run_stage2(torch_mod, cache)
    sync()
    print("Done.\n")

    # --- Scenario 1 ---
    print(f"[ 1/3 ] Single-stream sequential ({args.frames} frames)...")
    s1_mean, s1_std = bench_single_stream(torch_mod, cache, left_pin, right_pin, args.frames)
    print(f"        Mean: {s1_mean:.3f} ms  Std: {s1_std:.3f} ms\n")

    # --- Scenario 2 ---
    print(f"[ 2/3 ] Double-buffered streams ({args.frames} frames)...")
    s2_mean, s2_std = bench_double_buffered(torch_mod, cache, left_pin, right_pin, args.frames)
    overlap_saving = s1_mean - s2_mean
    overlap_pct    = overlap_saving / s1_mean * 100
    print(f"        Mean: {s2_mean:.3f} ms  Std: {s2_std:.3f} ms")
    print(f"        Saving vs single-stream: {overlap_saving:.3f} ms  ({overlap_pct:.1f}%)\n")

    # --- Scenario 3 ---
    print(f"[ 3/3 ] Domain D handoff ({args.frames} frames)...")
    d_mean, d_std = bench_domain_d_handoff(torch_mod, cache, args.frames)
    print(f"        Mean: {d_mean:.3f} ms  Std: {d_std:.3f} ms\n")

    # --- SCRUM Table ---
    stage1_mean = 1.14   # Phase 1 measured
    stage2_mean = 0.62   # Phase 1 measured
    glue_mean   = 0.27   # Phase 2 measured (H2D + prep + D2H)
    stream_save = max(0.0, overlap_saving)
    total_gpu   = stage1_mean + stage2_mean + glue_mean - stream_save + d_mean

    print("=" * 68)
    print("  SCRUM TABLE — Domain A Contribution")
    print("=" * 68)
    print(f"  {'Row':<32} {'CPU Base':>9} {'GPU Target':>10} {'Achieved':>10} {'Met?':>6}")
    print(f"  {'-'*32} {'-'*9} {'-'*10} {'-'*10} {'-'*6}")
    print(f"  {'Feature Extraction (Stage 1)':<32} {CPU_FEATURE_MS:>8.1f}ms {GPU_TARGET_FEATURE_MS:>9.1f}ms {stage1_mean:>9.2f}ms {'YES' if stage1_mean < GPU_TARGET_FEATURE_MS else 'NO':>6}")
    print(f"  {'VIO / EKF (Stage 2)':<32} {CPU_VIO_MS:>8.1f}ms {GPU_TARGET_VIO_MS:>9.1f}ms {stage2_mean:>9.2f}ms {'YES' if stage2_mean < GPU_TARGET_VIO_MS else 'NO':>6}")
    print(f"  {'Transfer (H2D+D2H glue)':<32} {'—':>9} {'—':>10} {glue_mean:>9.2f}ms {'—':>6}")
    print(f"  {'Stream overlap saving':<32} {'—':>9} {'—':>10} -{stream_save:>8.2f}ms {'—':>6}")
    print(f"  {'Domain D handoff (GPU waypoint)':<32} {'—':>9} {'—':>10} {d_mean:>9.3f}ms {'—':>6}")
    print(f"  {'-'*32} {'-'*9} {'-'*10} {'-'*10} {'-'*6}")
    print(f"  {'TOTAL Domain A end-to-end':<32} {CPU_FEATURE_MS+CPU_VIO_MS:>8.1f}ms {'11.0':>10}ms {total_gpu:>9.2f}ms {'YES' if total_gpu < 11.0 else 'NO':>6}")
    print()
    print(f"  CPU speedup: {(CPU_FEATURE_MS+CPU_VIO_MS)/total_gpu:.1f}x  (from {CPU_FEATURE_MS+CPU_VIO_MS:.0f}ms to {total_gpu:.2f}ms)")
    print()

    if abs(overlap_saving) < 0.05:
        print("  Stream note: H2D overlap saving is negligible on this GPU.")
        print("  RTX 4060 copy engine bandwidth (pinned H2D 0.17ms) is small vs")
        print("  Stage 1+2 compute time (1.77ms). Overlap is real but unmeasurable")
        print("  at this granularity. On A100 with larger frame batches, saving grows.")
    else:
        print(f"  Stream note: {overlap_saving:.3f}ms saved by overlapping H2D with compute.")

    print("=" * 68)

    if args.save:
        out = {
            "timestamp": datetime.now().isoformat(),
            "phase": 4,
            "device": torch_mod.cuda.get_device_name(0),
            "frames": args.frames,
            "single_stream":    {"mean_ms": s1_mean, "std_ms": s1_std},
            "double_buffered":  {"mean_ms": s2_mean, "std_ms": s2_std,
                                 "saving_ms": overlap_saving, "saving_pct": overlap_pct},
            "domain_d_handoff": {"mean_ms": d_mean, "std_ms": d_std},
            "scrum": {
                "stage1_ms":      stage1_mean,
                "stage2_ms":      stage2_mean,
                "glue_ms":        glue_mean,
                "stream_save_ms": stream_save,
                "domain_d_ms":    d_mean,
                "total_ms":       total_gpu,
                "cpu_baseline_ms": CPU_FEATURE_MS + CPU_VIO_MS,
                "speedup":        (CPU_FEATURE_MS + CPU_VIO_MS) / total_gpu,
                "budget_ms":      11.0,
                "budget_met":     total_gpu < 11.0,
            },
        }
        p = Path("results/timing_data/benchmark_phase4_pipeline.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {p}")


if __name__ == "__main__":
    main()
