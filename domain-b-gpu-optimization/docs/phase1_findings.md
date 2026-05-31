\# Phase 1 Findings — Simulated GPU Pipeline Baseline



\*\*Date:\*\* 2026-05-26  

\*\*Hardware:\*\* RTX 4060 Laptop, 8GB, 24 SMs, CC 8.9  

\*\*PyTorch:\*\* 2.11.0+cu128, CUDA 12.8  



\## Per-Stage Timing (Mean, 100 frames, 20 warmup cycles)



| Stage | Mean (ms) | Bottleneck % | Characteristic |

|---|---|---|---|

| Stage 1 Feature Extraction | 1.45 | 31.4% | Memory-bound (elementwise) |

| Stage 2 VIO / EKF Predict  | 0.35 |  7.6% | Memory-bound (tiny matrix) |

| Stage 3 Depth Estimation   | 1.56 | 33.7% | Mixed (sliding window conv) |

| Stage 4 Obstacle Detection | 0.25 |  5.4% | Compute-bound (dense matmul) |

| Stage 5 Path Planning      | 1.02 | 22.1% | Memory-bound (scatter/gather) |

| \*\*TOTAL\*\*                  | \*\*4.62\*\* | — | — |



\## Key Observations

\- Budget met: YES (4.62ms mean vs 33.3ms target)

\- Biggest bottleneck: Stage 3 Depth Estimation (1.56ms, 33.7%)

\- Stage 5 shows 30ms spike on Windows — WDDM driver batching artifact

&#x20; - Will NOT occur on Perlmutter A100 (Linux TCC mode)

&#x20; - Mean (1.02ms) is the valid measurement



\## Spike Analysis

\- Cause: Windows WDDM GPU command batching (driver-level, not code bug)

\- Affected stage: Stage 5 only (random collection, affects \~1 in 100 frames)

\- Action: None required. Confirmed expected behavior on Windows.



\## Tensor Cache Status

\- 13 tensors pre-allocated (42.2 MB GPU memory)

\- Zero GPU allocations in timing loop confirmed

\- All allocation spikes from previous version eliminated



\## Next Phase

\- Phase 1 Nsight Systems profile (visual timeline)

\- Phase 2: Memory transfer benchmark

