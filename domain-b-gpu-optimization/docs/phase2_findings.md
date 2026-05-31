\# Phase 2 Findings — Memory Transfer Analysis



\*\*Date:\*\* 2026-05-26

\*\*Hardware:\*\* RTX 4060 Laptop, PCIe Gen4



\## Measured PCIe Bandwidth (Windows WDDM mode)

\- Effective H2D: \~11-13 GB/s (37% of 32 GB/s theoretical)

\- Effective D2H: \~11-12 GB/s

\- D2D (GPU internal): 60-243 GB/s (HBM bandwidth, 5-21x faster than PCIe)

\- Note: Perlmutter A100 will achieve \~28-30 GB/s PCIe (Linux, no WDDM overhead)



\## Pinned Memory Improvement

\- Small buffers (<10KB): No improvement (overhead dominated by kernel launch \~40us)

\- Large buffers (>3MB): 1.08-1.26x faster than pageable

\- Pinned memory worth implementing for stereo frame input (saves \~140-180us)



\## Pipeline Transfer Overhead

| Scenario | Total Transfer Cost | Notes |

|---|---|---|

| Naive (CPU round-trip every stage) | 3.482 ms | Two full stereo H2D dominate |

| GPU-resident (input + output only) | 1.229 ms | One stereo H2D only |

| \*\*Saving\*\* | \*\*2.254 ms/frame\*\* | Goes into SCRUM Transfer row |



\## Key Insight: D2D is 9-21x Faster Than D2H

Keeping data on GPU between pipeline stages avoids PCIe completely.

Stage 1→Stage 2 handoff via GPU memory: 0.046ms

Stage 1→CPU→Stage 2 round-trip: 0.935ms

Ratio: 20x. This is the core justification for GPU-resident pipeline design.



\## SCRUM Table Update

Transfer Overhead row: \~3.5ms (naive) → \~1.2ms (GPU-resident) = 2.9x improvement

