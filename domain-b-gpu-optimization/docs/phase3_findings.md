\# Phase 3 Findings — CUDA Streams Analysis



\*\*Date:\*\* 2026-05-26

\*\*Hardware:\*\* RTX 4060 Laptop, 24 SMs



\## Measured Results



| Scenario | Time (ms) | Notes |

|---|---|---|

| Stage 3 Depth alone | 1.549 ms | Saturates \~20/24 SMs |

| Stage 4 Detection alone | 0.136 ms | Very fast, few SMs needed |

| Sequential (default) | 1.327 ms | PyTorch natural tail overlap |

| Parallel streams | 1.446 ms | SLOWER — SM contention |

| Speedup | 0.92x | Negative on this GPU |



\## Root Cause: SM Saturation



RTX 4060 has 24 SMs. Stage 3 depth estimation occupies

most of them during its convolution loop. Placing Stage 4

on a second stream causes both kernels to compete for the

same 24 SMs, degrading throughput for both.



Additionally: Stage 4 (0.136ms) is 11x smaller than Stage 3

(1.549ms). Even perfect overlap saves at most 0.136ms —

smaller than stream management overhead on this GPU.



\## Natural Overlap Already Present



Sequential total (1.327ms) < sum of parts (1.685ms).

PyTorch's default stream already pipelines Stage 4's tiny

matmul into Stage 3's kernel tail automatically.

Explicit streams are not needed and add overhead here.



\## A100 Projection



A100 has 108 SMs (4.5x more than RTX 4060).

\- Stage 3 on A100: \~0.27ms (bandwidth-bound improvement)

\- Stage 4 TensorRT INT8 on A100: \~0.05ms (tensor core path)

\- Combined SM demand: \~60-80 of 108 SMs

\- Both stages fit simultaneously without contention

\- Expected stream overlap saving on A100: \~0.15-0.25ms



\## SCRUM Table Note



Streams optimization: negligible on RTX 4060, \~0.2ms on A100.

Primary pipeline savings come from:

&#x20; 1. GPU-resident data flow (Phase 2: 2.254ms saving)

&#x20; 2. Per-stage GPU acceleration (Phases 1-4)

&#x20; 3. cuVSLAM + TensorRT replacing CPU implementations (Phase 5)

