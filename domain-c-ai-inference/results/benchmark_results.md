# Benchmark Results

## Objective

Evaluate inference performance of the UAV obstacle detection model across different deployment backends and optimization levels.

---

## Test Configuration

### Hardware

* GPU: NVIDIA RTX 3050 Laptop GPU
* VRAM: 6 GB

### Software

* CUDA 12.x
* TensorRT 10.14.1
* ONNX Runtime GPU
* Python 3.11

### Input Configuration

* Input Resolution: 640 × 640
* Batch Size: 1
* Benchmark Iterations: 100

---

## ONNX Runtime FP32

### Execution Provider

```text
CUDAExecutionProvider
CPUExecutionProvider
```

### Results

| Metric  | Value    |
| ------- | -------- |
| Latency | 14.00 ms |
| FPS     | 71.43    |

---

## TensorRT FP16

### Results

| Metric  | Value    |
| ------- | -------- |
| Latency | 10.22 ms |
| FPS     | 97.86    |

---

## TensorRT INT8

### Results

| Metric  | Value   |
| ------- | ------- |
| Latency | 8.31 ms |
| FPS     | 120.37  |

---

## Performance Comparison

| Backend      | Precision | Latency (ms) | FPS    |
| ------------ | --------- | ------------ | ------ |
| ONNX Runtime | FP32      | 14.00        | 71.43  |
| TensorRT     | FP16      | 10.22        | 97.86  |
| TensorRT     | INT8      | 8.31         | 120.37 |

---

## Speedup Analysis

### FP32 → FP16

97.86 / 71.43 = **1.37× speedup**

### FP32 → INT8

120.37 / 71.43 = **1.68× speedup**

### FP16 → INT8

120.37 / 97.86 = **1.23× speedup**

---

## Key Observations

* TensorRT FP16 significantly reduced inference latency compared to ONNX Runtime FP32.
* TensorRT INT8 achieved the highest throughput.
* INT8 quantization delivered approximately 68% higher throughput than the FP32 baseline.
* GPU utilization remained stable throughout profiling sessions.
* TensorRT optimization provided substantial deployment benefits for UAV perception workloads.

---

## Conclusion

TensorRT INT8 provided the best overall performance, reducing latency from 14.00 ms to 8.31 ms and increasing throughput from 71.43 FPS to 120.37 FPS.

This represents a 1.68× inference acceleration compared to the ONNX Runtime FP32 baseline.
