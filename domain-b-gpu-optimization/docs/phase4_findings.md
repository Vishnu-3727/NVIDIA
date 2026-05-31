\# Phase 4 Findings — Occupancy and Roofline Analysis



\*\*Date:\*\* 2026-05-26

\*\*Hardware:\*\* RTX 4060 Laptop

\*\*Ceilings:\*\* FP32=15,100 GFLOPS, BW=272 GB/s, Ridge=55.5 FLOPs/byte



\## Roofline Sweep (Matrix Multiply N×N)



| N | OI (FLOPs/byte) | Bound | Achieved (GFLOPS) | % Peak |

|---|---|---|---|---|

| 32-256 | 5.3-42.7 | MEMORY | 1-594 | <4% |

| 512 | 85.3 | COMPUTE | 3,086 | 20% |

| 1024 | 170.7 | COMPUTE | 5,495 | 36% |

| 2048+ | 341-682 | COMPUTE | 8,299-8,399 | 55-56% |



Hardware ceiling confirmed: \~8,400 GFLOPS achieved at large N (56% of peak).



\## Occupancy Sweep (Vector Addition, 64MB)



Peak bandwidth achieved: \*\*112.7 GB/s (41.4% of 272 GB/s theoretical)\*\*



Small chunks (<64K elements) dominated by kernel launch overhead (\~0.05ms fixed cost).

SM fully occupied only at chunk sizes >256K elements.

Implication: Stage 5 scatter/gather on 10K samples hits this overhead regime.



\## Pipeline Stage Classification



\*\*ALL STAGES ARE MEMORY-BOUND on RTX 4060\*\*



| Stage | OI (FLOPs/byte) | Bound | Optimization Direction |

|---|---|---|---|

| Stage 1 Feature Extraction | 0.25 | MEMORY | Coalesce pixel access, reduce redundant reads |

| Stage 2 VIO EKF (15×15) | 2.50 | MEMORY | Batch EKF ops (cuBLAS batched GEMM) |

| Stage 3 Depth Estimation | 4.17 | MEMORY | Reduce disparity loop passes, shared memory tiling |

| Stage 4 Obstacle Detection | 0.17 | MEMORY (batch=1) | TensorRT INT8 + batch=8+ → compute-bound |

| Stage 5 Path Planning | 0.08 | MEMORY | Coalesce occupancy grid access pattern |



\## Why A100 Gives Large Speedups



All stages memory-bound → speedup scales with bandwidth ratio:

A100 HBM2 = 1,555 GB/s vs RTX 4060 GDDR6 = 272 GB/s → \*\*5.7× bandwidth advantage\*\*



Expected A100 speedup from bandwidth alone (before algorithmic optimization):

\- Stage 1: \~4-5× (pixel access pattern scales with BW)

\- Stage 2: \~3× (small matrix, partially overhead-limited)

\- Stage 3: \~4-5× (conv BW-bound)

\- Stage 4: \~15-20× (TensorRT INT8 tensor cores, moves to compute-bound)

\- Stage 5: \~4× (scatter pattern scales with BW)



\## Stage 4 Exception: TensorRT Batch Effect



batch=1:  OI \~0.17 FLOPs/byte  → MEMORY-BOUND (weights read once, not reused)

batch=8:  OI \~1.4  FLOPs/byte  → still memory-bound but improving

batch=32: OI \~5.4  FLOPs/byte  → approaching ridge

INT8 + batch=32 on A100: likely COMPUTE-BOUND → tensor cores fully engaged



This is why Domain C's TensorRT INT8 work matters most for Stage 4.



\## Next Step: Nsight Compute Kernel Deep-Dive

Run ncu to get exact per-kernel occupancy, stall reasons, and actual memory throughput.

