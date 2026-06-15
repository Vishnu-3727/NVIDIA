# Phase 2 Findings — cuVSLAM Integration Interface

**Date:** 2026-06-14
**Hardware:** RTX 4060 Laptop, 8GB, 24 SMs, CC 8.9
**PyTorch:** 2.12.0+cu126, CUDA 12.6

## What Phase 2 Is

cuVSLAM is a production C++ CUDA Visual SLAM library from NVIDIA (Isaac ROS).
It replaces the entire PyTorch Stage 1+2 simulation with optimised native kernels.
It runs on Linux + A100 only — not on Windows.

Phase 2 measures everything **around** the cuVSLAM call:
- How fast stereo frames move from CPU to GPU (H2D)
- How fast the GPU formats the frames for cuVSLAM (uint8 to float32)
- How fast the pose result moves back to CPU for Domain D (D2H)
- The total glue cost so we know exactly what overhead cuVSLAM has to fit inside

> Run: `uv run python benchmarks/cuvslam/integration_benchmark.py --frames 100 --save`

## Section 1 — Stereo H2D Transfer

| Method | Mean (ms) | Bandwidth (GB/s) | Speedup |
|---|---|---|---|
| Pageable (standard numpy) | 0.2456 ms | 7.51 GB/s | 1.0x |
| Pinned (page-locked) | 0.1724 ms | 10.69 GB/s | **1.42x** |

- **Use pinned memory** for stereo frame input — 1.42x faster H2D, no code complexity cost
- RTX 4060 PCIe theoretical: 32 GB/s. Achieved 10.69 GB/s (33%) — consistent with Domain B Phase 2 findings (Windows WDDM overhead)
- A100 on Perlmutter (Linux, no WDDM): expect ~28-30 GB/s achieved (same pinned approach)

## Section 2 — GPU Data Preparation

| Operation | Mean (ms) | Bandwidth (GB/s) |
|---|---|---|
| uint8 → float32 + normalise (both frames) | 0.0872 ms | 84.57 GB/s |

- Conversion is memory-bound — 84.57 GB/s approaches GDDR6 ceiling (272 GB/s theoretical, typical effective ~100-130 GB/s for elementwise)
- On A100 (HBM2e, 2,039 GB/s): expect ~0.012 ms (~7.5x faster)
- cuVSLAM may accept uint8 natively — if so, this step disappears entirely

## Section 3 — Pose Output Extraction (D2H)

| Operation | Mean (ms) |
|---|---|
| D2H 4×4 SE3 matrix (64 bytes, pinned) | 0.0307 ms |

- 64 bytes is tiny — latency is dominated by PCIe round-trip overhead, not data size
- On A100: same cost (~0.03 ms) — PCIe latency is hardware-fixed regardless of GPU
- Pose goes to Domain D (path planner) — pinned destination buffer already in place

## Section 4 — Full Interface Round-trip

| Scenario | RTX 4060 | A100 (projected) |
|---|---|---|
| Stage 1+2 (Phase 1 / projected) | 1.77 ms | 0.40 ms |
| Interface glue (H2D + prep + D2H) | 0.27 ms | 0.27 ms |
| **TOTAL** | **2.04 ms** | **0.67 ms** |
| Budget | 11.0 ms | 11.0 ms |
| Budget Met? | **YES** | **YES** |

- Interface glue is **0.27 ms** — only 13% of total on RTX 4060, shrinks to 40% on A100
- Even with glue overhead included, A100 total (0.67 ms) is 16x under budget

## cuVSLAM API Boundary

On A100 with Isaac ROS, the integration point is:

```
Input  : float32 720x1280 left frame  (already on GPU, pinned H2D)
         float32 720x1280 right frame (already on GPU, pinned H2D)
Output : float32 4x4 SE3 pose matrix  (GPU -> pinned CPU for Domain D)
Call   : cuvslam::Tracker::Track(left_gpu, right_gpu) -> pose
```

The frames stay GPU-resident between the camera driver and cuVSLAM — no CPU round-trip.
This mirrors Domain B Phase 2's finding: D2D is 20x cheaper than H2D/D2H round-trips.

## Key Decisions for A100 Integration

1. **Use pinned memory** for stereo frame buffers — 1.42x H2D speedup, confirmed
2. **Keep frames GPU-resident** — never copy back to CPU between Stage 1 and cuVSLAM
3. **Pinned output buffer** for pose matrix — D2H latency minimised for Domain D handoff
4. **Skip uint8→float32 conversion** if cuVSLAM accepts uint8 natively (saves 0.087 ms)

## Next Phase

- Phase 3: Nsight Compute (ncu) kernel-level roofline analysis on Stage 1 Harris conv
  (mirrors Domain B Phase 4 — identifies whether bandwidth or compute is the true ceiling)
