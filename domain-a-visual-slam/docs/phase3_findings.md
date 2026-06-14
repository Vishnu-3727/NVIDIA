# Phase 3 Findings — Roofline Analysis

**Date:** 2026-06-14
**Hardware:** RTX 4060 Laptop, 8GB, 24 SMs, CC 8.9
**PyTorch:** 2.12.0+cu126, CUDA 12.6

> Run: `uv run python benchmarks/roofline/roofline_benchmark.py --frames 100 --save`

## Hardware Ceilings

| GPU | FP32 Peak | Memory BW | Ridge Point |
|---|---|---|---|
| RTX 4060 Laptop (dev) | 15,100 GFLOPS | 272 GB/s | 55.5 FLOPs/byte |
| A100 SXM (Perlmutter) | 19,500 GFLOPS | 2,039 GB/s | 9.6 FLOPs/byte |

The ridge point is where a kernel switches from memory-bound to compute-bound.
Any kernel with OI below the ridge is limited by memory bandwidth, not compute.

## Kernel Roofline Table (Analytical OI)

| Kernel | OI (FLOPs/byte) | Bound | RTX proj (GFLOPS) | A100 proj (GFLOPS) |
|---|---|---|---|---|
| S1: Gradients (Ix, Iy) | 0.17 | **MEMORY** | 45 | 340 |
| S1: Structure Tensor (Ixx, Iyy, Ixy) | 0.15 | **MEMORY** | 41 | 306 |
| S1: Conv2d 5×5 (Harris smooth) | 6.25 | **MEMORY** | 1,700 | 12,744 |
| S1: Harris Response R | 0.38 | **MEMORY** | 102 | 765 |
| S1: TopK-500 (radixSort) | 2.48 | **MEMORY** | 674 | 5,049 |
| S1: LK Optical Flow (u, v) | 0.29 | **MEMORY** | 79 | 595 |
| S2: EKF Predict (F@x, F@P@Ft) | 3.81 | **MEMORY** | 1,036 | 7,770 |
| S2: EKF Update (K, x, P) | 4.59 | **MEMORY** | 1,249 | 9,360 |

**ALL 8 kernels are memory-bound on both RTX 4060 and A100.**
Consistent with Domain B Phase 4 finding (all pipeline stages memory-bound).

### Why Conv2d has the highest OI (6.25 FLOPs/byte)

The 5×5 box filter does 50 MACs (25 multiply + 25 add) per output pixel, but reads only 8 bytes (one float32 per element). This gives a higher intensity than simple elementwise ops — but 6.25 is still far below the RTX ridge (55.5), so it remains memory-bound.

### Why EKF Update has high OI (4.59 FLOPs/byte)

The EKF matmuls (H@P, PHt@inv(S), (I-KH)@P) are FLOPs-heavy relative to data moved — but the matrices are tiny (15×15, 3×3). Tiny matrices mean kernel launch overhead dominates over actual BW or compute, which is why Stage 2 measured bandwidth shows near-zero (see below).

## Achieved Bandwidth (Measured)

| Stage | Mean (ms) | Achieved BW | % of RTX Peak | Bound |
|---|---|---|---|---|
| Stage 1 Harris + LK Flow | 0.928 ms | 103.1 GB/s | **37.9%** | MEMORY |
| Stage 2 VIO / EKF | 0.434 ms | ~0 GB/s | — | Overhead |

### Stage 1 — 37.9% bandwidth utilisation

Achieved 103.1 GB/s out of 272 GB/s peak. Gap from theoretical peak explained by:
- Multiple separate kernel launches (gradient, conv2d x3, topk, flow) — each has fixed launch overhead ~5-10 µs
- Conv2d intermediate results written and re-read between passes (no kernel fusion)
- On A100, PyTorch's `torch.compile` or cuVSLAM's fused kernels would close this gap

### Stage 2 — kernel launch overhead regime

Stage 2 shows near-zero measured bandwidth because the 15×15 EKF matrices are tiny:
- Largest data movement: P matrix = 15×15×4 = 900 bytes
- Kernel launch overhead (~5 µs) >> actual data transfer time
- This is the same small-chunk overhead Domain B observed in occupancy sweep (chunks < 64K elements)
- cuVSLAM's C++ EKF implementation fuses these ops into a single kernel — eliminates all launch overhead

## A100 Speedup Projection

| Stage | RTX 4060 | A100 (projected) | Speedup |
|---|---|---|---|
| Stage 1 Harris + LK Flow | 0.928 ms | 0.124 ms | **7.5x** |
| Stage 2 VIO / EKF | 0.434 ms | 0.058 ms | **7.5x** |
| **TOTAL** | **1.361 ms** | **0.182 ms** | **7.5x** |

BW ratio: 2,039 / 272 = **7.5x** (A100 HBM2e vs RTX 4060 GDDR6)

> This is a conservative lower bound — cuVSLAM native kernels will additionally benefit from:
> - Fused kernel launches (eliminates per-kernel ~5 µs overhead)
> - Better memory access coalescing (C++ vs PyTorch eager mode)
> - No JIT compilation cost

## Key Findings vs Domain B

| | Domain B (all stages) | Domain A (Stage 1+2) |
|---|---|---|
| All stages memory-bound? | YES | YES |
| Achieved BW % of peak | ~41% (Stage 1) | ~38% (Stage 1) |
| A100 BW ratio | 5.7x (GDDR6 272 → HBM2 1555) | 7.5x (GDDR6 272 → HBM2e 2039) |
| Small matrix overhead? | Stage 2 EKF (15x15) | Same |

Note: Domain B used A100 HBM2 (1,555 GB/s). Perlmutter uses A100 SXM with HBM2e (2,039 GB/s) — giving a higher BW ratio than Domain B projected.

## Optimization Recommendations for A100

1. **Fuse Stage 1 kernels** (gradient → structure tensor → conv → Harris R → flow) into one kernel — eliminates 5 separate launch overheads, improves cache reuse. cuVSLAM does this natively.
2. **Replace TopK with threshold NMS** — radixSort is the highest-OI kernel (2.48), still memory-bound but avoids full sort. cuVSLAM uses GPU-native NMS.
3. **Fuse EKF predict+update** into single kernel — eliminates ~8 tiny launch overheads. cuVSLAM's EKF is one fused C++ call.

## Next Phase

- Phase 4: End-to-end integration handoff — connect Domain A output (SE3 pose) to Domain D path planner, measure latency across the boundary.
