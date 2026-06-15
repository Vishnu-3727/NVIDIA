# Phase 1 Findings — Domain A Visual SLAM Baseline

**Date:** 2026-06-14
**Hardware:** RTX 4060 Laptop, 8GB, 24 SMs, CC 8.9
**PyTorch:** 2.12.0+cu126, CUDA 12.6

## Per-Stage Timing (Mean, 100 frames, 20 warmup)

| Stage | Mean (ms) | Min (ms) | Max (ms) | Budget (ms) | Budget Met? | Characteristic |
|---|---|---|---|---|---|---|
| Stage 1 Harris + Optical Flow | 1.14 | 1.05 | 1.60 | 6.0 | **YES** (5.3× headroom) | Memory-bound (pixel gradients) |
| Stage 2 VIO / EKF Predict+Update | 0.62 | 0.55 | 1.05 | 5.0 | **YES** (8.1× headroom) | Memory-bound (15×15 matrix ops) |
| **TOTAL** | **1.77** | **1.61** | **2.39** | **11.0** | **YES** | — |

- Spike check: **PASS** — all stages max < 3× mean (no WDDM batching issues)
- Zero GPU allocations in timing loop — pre-allocated cache working correctly
- Biggest bottleneck: Stage 1 Harris + Optical Flow (64.7% of total)

> Run: `uv run python benchmarks/slam/slam_benchmark.py --frames 100 --save`

## Trajectory Accuracy (ATE / RPE)

| Metric | IMU-only (drift) | cuVSLAM (simulated) | Target | Pass? |
|---|---|---|---|---|
| ATE (500 frames) | 0.0845 m | 0.0051 m | < 0.05 m | IMU: FAIL / cuVSLAM: **PASS** |
| RPE @ 5 frames | 0.0187 m | 0.0011 m | — | — |
| RPE @ 10 frames | 0.0252 m | 0.0015 m | < 0.01 m | **PASS** |
| RPE @ 30 frames | 0.0417 m | 0.0025 m | — | — |
| RPE @ 50 frames | 0.0534 m | 0.0032 m | — | — |
| RPE @ 100 frames | 0.0643 m | 0.0039 m | < 0.05 m | **PASS** |

- cuVSLAM gives **16.7× improvement** over pure IMU drift across all metrics
- Trajectory: helical UAV path, radius=20m, vz=0.3m/s, 16.5s of flight

> Run: `uv run python benchmarks/trajectory/ate_rpe_benchmark.py`

## Key Observations

- Both pipeline stages have massive headroom vs. budget on RTX 4060 (total 1.77ms vs 11ms budget)
- Stage 1 dominates at 64.7% of total — Harris conv2d is the memory-bound kernel to watch
- No WDDM spike behaviour observed (unlike Domain B Stage 5) — Stage 1 conv is smaller than Stage 5's large convolutions
- ATE computation: CPU numpy 0.082ms for 500 frames; GPU torch 88ms first-call (CUDA init overhead), will be <0.1ms warm on A100

## A100 Projection

- Domain B Phase 4: ALL pipeline stages memory-bound on RTX 4060
- A100 HBM2e (2,039 GB/s) vs RTX 4060 GDDR6 (272 GB/s) = 7.5× bandwidth advantage
- Expected Stage 1+2 speedup on A100: ~4–5× from bandwidth alone (before cuVSLAM)
- cuVSLAM replaces the PyTorch simulation → additional algorithmic speedup on top
- Projected total on A100: ~0.35–0.45ms (well within 11ms budget)

## Windows WDDM Note

No spikes observed in this run. Stage 1 conv kernels are smaller than Domain B Stage 5 — WDDM batching threshold not triggered. Mean timing is the valid measurement regardless. Will not occur on Perlmutter (Linux, TCC mode).

## Nsight Systems Profile — session1_slam_baseline

**Report:** `profiling/nsight_reports/session1_slam_baseline.nsys-rep`
**Tool:** NVIDIA Nsight Systems 2026.3.1
**Traces:** CUDA, NVTX (WDDM disabled — requires admin privileges)

### NVTX Timeline Observations

| NVTX Region | First-call Duration | Warmed Duration | Cause |
|---|---|---|---|
| `stage1_feature_extraction` | 149.233 ms | ~1.14 ms | JIT spike on first dispatch |
| `stage2_vio_ekf` | 66.830 ms | ~0.62 ms | JIT spike on first dispatch |
| `full_slam_cycle` | 3.509 s total | — | Covers all 30 profiled frames |

- **First-call JIT spikes:** PyTorch compiles CUDA kernels on their very first GPU dispatch. The 20-frame warmup in the benchmark runs before Nsight attaches, but Nsight captured the first actual GPU calls which still included JIT compilation. These 149ms / 66ms durations are **not real runtime** — they are one-time compilation cost.
- **Warmed cycles (right side of timeline):** The repeating tiny kernel blocks after the first two large bars are the actual steady-state cycles, consistent with benchmark numbers (1.14ms + 0.62ms).

### CUDA Kernel Breakdown (Stage 1)

Kernels visible in the CUDA API row under `stage1_feature_extraction`:
- `elementwise_*` — pixel gradient ops (Ix, Iy, Ixx, Iyy, Ixy), det/trace/Harris R computation
- `vectorized_elementwise_*` — fused element-wise ops (memory-bound, wide flat bars)
- `fill` — tensor cache reset between frames
- `radixSort` — `torch.topk` for top-500 keypoint selection (compute-bound, narrow)

### Key Nsight Findings

- **Stage 1 is memory-bound confirmed** — elementwise and vectorized kernels appear as wide, flat bars with low SM occupancy relative to duration, consistent with GDDR6 bandwidth ceiling
- **Stage 2 matmul kernels** (`linalg.inv`, `mm`) appear narrow and fast — 15×15 matrices are too small to fully utilise tensor cores; compute overhead is minimal
- **No idle gaps** between Stage 1 and Stage 2 — kernel launches are back-to-back, no CPU stall visible in the warmed region
- **`radixSort` in Stage 1** is the only non-memory-bound kernel — `topk(500)` on a 919×1279 flattened tensor; worth replacing with a threshold-based NMS on A100 to eliminate sort entirely

### A100 Implication from Profile

On A100 (HBM2e, 2,039 GB/s vs RTX 4060 GDDR6 272 GB/s):
- Memory-bound elementwise kernels in Stage 1 will be ~7.5× faster
- `radixSort` (compute-bound) will benefit less — NMS replacement recommended
- cuVSLAM replaces the entire PyTorch Stage 1+2 simulation with optimised C++ CUDA kernels — JIT spikes disappear entirely

## Next Phase

- Phase 2: cuVSLAM SDK integration plan (Isaac ROS binary on A100)
