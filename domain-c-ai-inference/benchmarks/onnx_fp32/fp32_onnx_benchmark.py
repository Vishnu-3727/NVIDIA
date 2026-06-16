import onnxruntime as ort
import numpy as np
import time
import nvtx

session = ort.InferenceSession(
    "models/weights/best.onnx",
    providers=[
        ("CUDAExecutionProvider", {}),
        "CPUExecutionProvider"
    ]
)

print("Provider:", session.get_providers())

input_name = session.get_inputs()[0].name
dummy = np.random.rand(
    1, 3, 640, 640
).astype(np.float32)

# Warmup
for _ in range(20):
   nvtx.push_range("ONNX_FP32")
session.run(None, {input_name: dummy})
nvtx.pop_range()

times = []

for _ in range(100):
    t0 = time.perf_counter()
    session.run(None, {input_name: dummy})
    times.append((time.perf_counter() - t0) * 1000)

latency = np.mean(times)
fps = 1000 / latency

print(f"Latency: {latency:.2f} ms")
print(f"FPS: {fps:.2f}")