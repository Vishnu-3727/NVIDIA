# ASCEND GPU Pipeline
## GPU-Accelerated Visual-Inertial Perception Pipeline for GPS-Denied UAV Navigation

**Event:** NERSC Open Hackathon, July 15, 2026  
**Goal:** Reduce 65ms CPU pipeline to <11ms on Perlmutter A100

---

## Team Structure

| Domain | Folder | Owner | Responsibility |
|---|---|---|---|
| A | domain-a-visual-slam/ | Paresh | cuVSLAM integration, VIO, trajectory accuracy |
| B | domain-b-gpu-optimization/ | Vishnu | Profiling, roofline, memory optimization, CUDA streams |
| C | domain-c-ai-inference/ | Yugawathi | TensorRT INT8, obstacle detection, model optimization |
| D | domain-d-robotics-planning/ | Kamalesh | Pipeline integration, path planning, SLURM, Perlmutter |

## Final Target (SCRUM Table)

| Stage | CPU Baseline | GPU Target | Speedup |
|---|---|---|---|
| Feature Extraction | 28 ms | 6 ms | 4.7× |
| VIO / EKF | 12 ms | 5 ms | 2.4× |
| Depth Estimation | 18 ms | 3 ms | 6× |
| Obstacle Detection | 42 ms | 2 ms | 21× |
| Path Planning | 8 ms | 1.5 ms | 5.3× |
| Transfer Overhead | ~5 ms | 0.5 ms | 10× |
| **End-to-End** | **~65 ms** | **~11 ms** | **~5.9×** |

## Current Status

- [x] Domain B: Phase 1-4 complete (profiling, transfers, streams, roofline, ncu analysis)
- [x] Domain A: Phase 1-4 complete (baseline 1.78ms, 22.5x CPU speedup, cuVSLAM integration plan, roofline, pipeline handoff)
- [ ] Domain C: TensorRT INT8 conversion
- [ ] Domain D: Pipeline integration + Perlmutter setup

## Environment

- Python: 3.11.9 (pyenv)
- Package manager: uv (no conda, no system pip)
- CUDA: 12.8 (PyTorch) / 13.2 (nvcc)
- Target GPU: NVIDIA A100 (Perlmutter), Dev: RTX 4060 Laptop

## Quick Start

```bash
git clone <repo-url>
cd ascend-gpu-pipeline/domain-b-gpu-optimization
uv sync
uv run python benchmarks/pipeline/pipeline_benchmark.py --frames 100 --save
```
