"""
Phase 2 — cuVSLAM Integration Interface Benchmark

cuVSLAM (Isaac ROS) is a C++ CUDA library that replaces our PyTorch Stage 1+2
simulation with production-grade Visual SLAM. It cannot run on Windows — this
benchmark measures everything AROUND the cuVSLAM call:

  - Stereo frame H2D transfer (pageable vs pinned memory)
  - GPU-resident data preparation (format conversion, normalisation)
  - Pose output extraction (D2H for downstream pipeline)
  - End-to-end interface overhead (what cuVSLAM sees on entry and exit)

On A100 (Perlmutter), the cuVSLAM call itself replaces Stage 1+2.
This benchmark isolates the surrounding glue cost so we know the true overhead.
"""

import time, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except ImportError:
    CUDA_OK = False

try:
    import nvtx
    _NVTX = True
except ImportError:
    _NVTX = False


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
        print("Device : CPU (CUDA not available — projections only)")
        return None, "cpu"
    props = torch.cuda.get_device_properties(0)
    print(f"Device : CUDA")
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1024**3:.1f} GB")
    print(f"SMs    : {props.multi_processor_count}")
    print(f"CC     : {props.major}.{props.minor}")
    print(f"Torch  : {torch.__version__}")
    print(f"CUDA rt: {torch.version.cuda}")
    return torch, "cuda"


def sync():
    if CUDA_OK:
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# SECTION 1 — H2D Transfer: pageable vs pinned memory
# cuVSLAM receives stereo frames from the camera driver.
# Pinned (page-locked) memory avoids an extra CPU copy during DMA.
# ---------------------------------------------------------------------------

def bench_h2d_transfer(torch_mod, n_frames=100):
    """Measure H2D transfer cost for one stereo pair (left + right 720x1280 uint8)."""
    H, W = 720, 1280
    frame_bytes = H * W  # uint8 grayscale, one frame

    results = {}

    # --- Pageable (standard numpy array → GPU) ---
    left_page  = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    right_page = np.random.randint(0, 255, (H, W), dtype=np.uint8)

    # warmup
    for _ in range(10):
        tl = torch_mod.from_numpy(left_page).cuda()
        tr = torch_mod.from_numpy(right_page).cuda()
    sync()

    times = []
    for _ in range(n_frames):
        ev0 = torch_mod.cuda.Event(enable_timing=True)
        ev1 = torch_mod.cuda.Event(enable_timing=True)
        ev0.record()
        tl = torch_mod.from_numpy(left_page).cuda()
        tr = torch_mod.from_numpy(right_page).cuda()
        ev1.record()
        sync()
        times.append(ev0.elapsed_time(ev1))
    results["pageable_ms"] = float(np.mean(times))
    results["pageable_bw_gbs"] = (2 * frame_bytes / 1e9) / (results["pageable_ms"] / 1e3)

    # --- Pinned (page-locked host buffer → GPU) ---
    left_pin  = torch_mod.zeros(H, W, dtype=torch_mod.uint8).pin_memory()
    right_pin = torch_mod.zeros(H, W, dtype=torch_mod.uint8).pin_memory()
    left_pin.numpy()[:] = left_page
    right_pin.numpy()[:] = right_page

    for _ in range(10):
        tl = left_pin.cuda(non_blocking=True)
        tr = right_pin.cuda(non_blocking=True)
    sync()

    times = []
    for _ in range(n_frames):
        ev0 = torch_mod.cuda.Event(enable_timing=True)
        ev1 = torch_mod.cuda.Event(enable_timing=True)
        ev0.record()
        tl = left_pin.cuda(non_blocking=True)
        tr = right_pin.cuda(non_blocking=True)
        ev1.record()
        sync()
        times.append(ev0.elapsed_time(ev1))
    results["pinned_ms"] = float(np.mean(times))
    results["pinned_bw_gbs"] = (2 * frame_bytes / 1e9) / (results["pinned_ms"] / 1e3)
    results["speedup"] = results["pageable_ms"] / results["pinned_ms"]

    return results


# ---------------------------------------------------------------------------
# SECTION 2 — GPU-resident data preparation
# cuVSLAM expects float32 normalised frames [0,1].
# We measure: uint8→float32 conversion + normalisation on GPU.
# ---------------------------------------------------------------------------

def bench_data_prep(torch_mod, n_frames=100):
    """Measure GPU-side uint8→float32 normalisation for both stereo frames."""
    H, W = 720, 1280

    left_gpu  = torch_mod.randint(0, 255, (H, W), dtype=torch_mod.uint8, device="cuda")
    right_gpu = torch_mod.randint(0, 255, (H, W), dtype=torch_mod.uint8, device="cuda")
    left_f32  = torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda")
    right_f32 = torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda")

    for _ in range(10):
        left_f32.copy_(left_gpu.float().div_(255.0))
        right_f32.copy_(right_gpu.float().div_(255.0))
    sync()

    times = []
    for _ in range(n_frames):
        with annotate("data_prep", color="yellow"):
            ev0 = torch_mod.cuda.Event(enable_timing=True)
            ev1 = torch_mod.cuda.Event(enable_timing=True)
            ev0.record()
            left_f32.copy_(left_gpu.float().div_(255.0))
            right_f32.copy_(right_gpu.float().div_(255.0))
            ev1.record()
            sync()
            times.append(ev0.elapsed_time(ev1))

    frame_bytes = H * W * 4 * 2  # float32, two frames
    mean_ms = float(np.mean(times))
    return {
        "mean_ms":    mean_ms,
        "min_ms":     float(np.min(times)),
        "max_ms":     float(np.max(times)),
        "bw_gbs":     (frame_bytes / 1e9) / (mean_ms / 1e3),
    }


# ---------------------------------------------------------------------------
# SECTION 3 — Pose output extraction (D2H)
# cuVSLAM writes the 6-DOF pose as a 4×4 SE3 matrix on GPU.
# Downstream path planner (Domain D) reads it on CPU.
# ---------------------------------------------------------------------------

def bench_pose_extraction(torch_mod, n_frames=100):
    """Measure D2H cost of extracting one 4x4 pose matrix per frame."""
    pose_gpu = torch_mod.eye(4, device="cuda")
    pose_cpu = torch_mod.zeros(4, 4, pin_memory=True)

    for _ in range(10):
        pose_cpu.copy_(pose_gpu, non_blocking=True)
    sync()

    times = []
    for _ in range(n_frames):
        with annotate("pose_extraction", color="orange"):
            ev0 = torch_mod.cuda.Event(enable_timing=True)
            ev1 = torch_mod.cuda.Event(enable_timing=True)
            ev0.record()
            pose_cpu.copy_(pose_gpu, non_blocking=True)
            ev1.record()
            sync()
            times.append(ev0.elapsed_time(ev1))

    pose_bytes = 4 * 4 * 4  # float32
    mean_ms = float(np.mean(times))
    return {
        "mean_ms": mean_ms,
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
        "bw_gbs":  (pose_bytes / 1e9) / (mean_ms / 1e3),
    }


# ---------------------------------------------------------------------------
# SECTION 4 — Full interface round-trip
# Simulates one complete frame: H2D → data_prep → [cuVSLAM stub] → D2H
# ---------------------------------------------------------------------------

def bench_full_interface(torch_mod, n_frames=100):
    """End-to-end interface overhead around the cuVSLAM call."""
    H, W = 720, 1280

    left_pin  = torch_mod.zeros(H, W, dtype=torch_mod.uint8).pin_memory()
    right_pin = torch_mod.zeros(H, W, dtype=torch_mod.uint8).pin_memory()
    left_gpu  = torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda")
    right_gpu = torch_mod.zeros(H, W, dtype=torch_mod.float32, device="cuda")
    pose_gpu  = torch_mod.eye(4, device="cuda")
    pose_cpu  = torch_mod.zeros(4, 4, pin_memory=True)

    for _ in range(10):
        left_gpu.copy_(left_pin.cuda(non_blocking=True).float().div_(255.0))
        right_gpu.copy_(right_pin.cuda(non_blocking=True).float().div_(255.0))
        pose_cpu.copy_(pose_gpu, non_blocking=True)
    sync()

    times = []
    for _ in range(n_frames):
        with annotate("full_interface", color="cyan"):
            ev0 = torch_mod.cuda.Event(enable_timing=True)
            ev1 = torch_mod.cuda.Event(enable_timing=True)
            ev0.record()

            # Step 1: H2D stereo frames (pinned)
            with annotate("h2d_stereo", color="blue"):
                lg = left_pin.cuda(non_blocking=True)
                rg = right_pin.cuda(non_blocking=True)

            # Step 2: uint8 → float32 normalise
            with annotate("normalise", color="yellow"):
                left_gpu.copy_(lg.float().div_(255.0))
                right_gpu.copy_(rg.float().div_(255.0))

            # Step 3: [cuVSLAM call would go here — ~0.35ms on A100]
            # On this laptop we simulate with a no-op to measure glue cost only

            # Step 4: D2H pose extraction
            with annotate("d2h_pose", color="orange"):
                pose_cpu.copy_(pose_gpu, non_blocking=True)

            ev1.record()
            sync()
            times.append(ev0.elapsed_time(ev1))

    mean_ms = float(np.mean(times))
    return {
        "mean_ms": mean_ms,
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
        "std_ms":  float(np.std(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  DOMAIN A — cuVSLAM INTEGRATION INTERFACE BENCHMARK (Phase 2)")
    print("  Stereo H2D | Data Prep | Pose D2H | Full Interface Round-trip")
    print("=" * 65)
    print()

    torch_mod, mode = setup_device()
    print()

    if mode != "cuda":
        print("ERROR: CUDA required for this benchmark.")
        return

    # --- Section 1: H2D Transfer ---
    print("[ 1/4 ] Stereo H2D Transfer (pageable vs pinned) ...")
    h2d = bench_h2d_transfer(torch_mod, args.frames)
    print(f"        Pageable : {h2d['pageable_ms']:.4f} ms  ({h2d['pageable_bw_gbs']:.2f} GB/s)")
    print(f"        Pinned   : {h2d['pinned_ms']:.4f} ms  ({h2d['pinned_bw_gbs']:.2f} GB/s)")
    print(f"        Speedup  : {h2d['speedup']:.2f}x\n")

    # --- Section 2: Data Prep ---
    print("[ 2/4 ] GPU Data Preparation (uint8 -> float32 normalise) ...")
    prep = bench_data_prep(torch_mod, args.frames)
    print(f"        Mean : {prep['mean_ms']:.4f} ms  ({prep['bw_gbs']:.2f} GB/s)\n")

    # --- Section 3: Pose Extraction ---
    print("[ 3/4 ] Pose Output Extraction (D2H 4x4 SE3 matrix) ...")
    pose = bench_pose_extraction(torch_mod, args.frames)
    print(f"        Mean : {pose['mean_ms']:.4f} ms  ({pose['bw_gbs']:.2f} GB/s)\n")

    # --- Section 4: Full Interface ---
    print("[ 4/4 ] Full Interface Round-trip (H2D + prep + stub + D2H) ...")
    full = bench_full_interface(torch_mod, args.frames)
    print(f"        Mean : {full['mean_ms']:.4f} ms\n")

    # --- Summary ---
    cuvslam_rtx   = 1.77   # Phase 1 measured (PyTorch sim)
    cuvslam_a100  = 0.40   # projected: 7.5x BW gain + algorithmic speedup
    glue_cost     = full["mean_ms"]
    total_rtx     = cuvslam_rtx  + glue_cost
    total_a100    = cuvslam_a100 + glue_cost
    budget        = 11.0

    print("=" * 65)
    print("  PHASE 2 SUMMARY — cuVSLAM Integration Cost Model")
    print("=" * 65)
    print(f"  GPU    : {torch_mod.cuda.get_device_name(0)}")
    print(f"  Frames : {args.frames}\n")

    print(f"  {'Component':<40} {'RTX 4060':>10}  {'A100':>10}")
    print(f"  {'-'*40} {'-'*10}  {'-'*10}")
    print(f"  {'Stage 1+2 (Phase 1 measured / projected)':<40} {cuvslam_rtx:>9.2f}ms  {cuvslam_a100:>9.2f}ms")
    print(f"  {'Interface glue (H2D + prep + D2H)':<40} {glue_cost:>9.4f}ms  {glue_cost:>9.4f}ms")
    print(f"  {'-'*40} {'-'*10}  {'-'*10}")
    print(f"  {'TOTAL':<40} {total_rtx:>9.2f}ms  {total_a100:>9.2f}ms")
    print(f"  {'Budget':<40} {budget:>9.1f}ms  {budget:>9.1f}ms")
    print(f"  {'Budget Met?':<40} {'YES' if total_rtx < budget else 'NO':>10}  {'YES' if total_a100 < budget else 'NO':>10}\n")

    print("  Pinned memory recommendation:")
    if h2d["speedup"] > 1.1:
        print(f"  USE pinned buffers for stereo input — {h2d['speedup']:.2f}x faster H2D")
    else:
        print(f"  Pinned vs pageable difference negligible at this frame size ({h2d['speedup']:.2f}x)")

    print()
    print("  cuVSLAM API boundary (on A100 / Isaac ROS):")
    print("  Input  : GPU float32 720x1280 left + right frames (already on GPU)")
    print("  Output : GPU float32 4x4 SE3 pose matrix")
    print("  Call   : cuvslam::Tracker::Track(left_gpu, right_gpu) -> pose")
    print("=" * 65)

    if args.save:
        out = {
            "timestamp":      datetime.now().isoformat(),
            "phase":          2,
            "device":         torch_mod.cuda.get_device_name(0),
            "frames":         args.frames,
            "h2d_transfer":   h2d,
            "data_prep":      prep,
            "pose_extraction": pose,
            "full_interface": full,
            "cost_model": {
                "stage12_rtx_ms":   cuvslam_rtx,
                "stage12_a100_ms":  cuvslam_a100,
                "glue_ms":          glue_cost,
                "total_rtx_ms":     total_rtx,
                "total_a100_ms":    total_a100,
                "budget_ms":        budget,
            },
        }
        p = Path("results/timing_data/benchmark_phase2_integration.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {p}")


if __name__ == "__main__":
    main()
