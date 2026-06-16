from ultralytics import YOLO
import time
import numpy as np
import nvtx

model = YOLO("models/weights/best.engine")

dummy = np.random.randint(
    0, 255,
    (640, 640, 3),
    dtype=np.uint8
)

# Warmup
for _ in range(20):
    nvtx.push_range("TensorRT_FP16")
model(dummy, verbose=False)
nvtx.pop_range()
times = []

for _ in range(100):
    t0 = time.perf_counter()
    model(dummy, verbose=False)
    times.append((time.perf_counter() - t0) * 1000)

latency = np.mean(times)
fps = 1000 / latency

print(f"Latency: {latency:.2f} ms")
print(f"FPS: {fps:.2f}")