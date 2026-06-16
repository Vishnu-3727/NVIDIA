# Domain C – AI Inference & Model Optimization Report

## Objective

Optimize a UAV obstacle detection model for accelerated GPU inference using NVIDIA deployment technologies while maintaining acceptable detection performance.

---

## Environment

### Hardware

* NVIDIA RTX 3050 Laptop GPU
* 6 GB VRAM
* Windows 11

### Software

* Python 3.11
* CUDA 12.x
* cuDNN 9.23
* TensorRT 10.14.1
* ONNX Runtime GPU
* PyTorch
* Ultralytics YOLO

---

## Model Artifacts

Original Model:

* best.pt

Exported Model:

* best.onnx

TensorRT Engines:

* best.engine
* best_int8.engine

---

## Dataset

### VisDrone2019-DET Validation

* 548 validation images
* 548 annotation files
* UAV aerial imagery

Used for:

* Validation
* Accuracy measurement
* INT8 calibration
* Performance evaluation

---

## Optimization Workflow

1. Export PyTorch model to ONNX
2. Benchmark ONNX Runtime FP32
3. Generate TensorRT FP16 engine
4. Generate TensorRT INT8 engine
5. Validate model performance
6. Add NVTX instrumentation
7. Profile using NVIDIA Nsight Systems

---

## Benchmark Results

| Backend      | Precision | Latency (ms) | FPS    |
| ------------ | --------- | ------------ | ------ |
| ONNX Runtime | FP32      | 14.00        | 71.43  |
| TensorRT     | FP16      | 10.22        | 97.86  |
| TensorRT     | INT8      | 8.31         | 120.37 |

### Throughput Improvement

120.37 / 71.43 ≈ 1.68×

---

## Validation Results

| Metric    | Value |
| --------- | ----- |
| Precision | 0.521 |
| Recall    | 0.389 |
| mAP50     | 0.370 |
| mAP50-95  | 0.221 |

Validation completed successfully on all 548 images.

---

## NVTX Instrumentation

Added timeline markers:

* ONNX_FP32
* TensorRT_FP16
* TensorRT_INT8

Purpose:

* Timeline segmentation
* Profiling visibility
* GPU workload analysis

---

## Nsight Systems Profiling

Generated Reports:

* fp32_profile.nsys-rep
* fp16_profile.nsys-rep
* int8_profile.nsys-rep

Verified:

* CUDA execution
* TensorRT inference
* NVTX visibility
* GPU stream activity

---

## Deliverables

### Models

* best.pt
* best.onnx

### Benchmark Scripts

* fp32_onnx_benchmark.py
* fp16_trt_benchmark.py
* int8_trt_benchmark.py

### Utility Scripts

* export_fp32.py
* export_fp16.py
* verify_model.py
* test_inference.py

### Profiling Reports

* fp32_profile.nsys-rep
* fp16_profile.nsys-rep
* int8_profile.nsys-rep

---

## Conclusion

The optimization workflow successfully improved UAV perception inference throughput from 71.43 FPS to 120.37 FPS while maintaining successful validation performance.

TensorRT INT8 provided the best performance, achieving approximately 1.68× higher throughput than ONNX Runtime FP32.
