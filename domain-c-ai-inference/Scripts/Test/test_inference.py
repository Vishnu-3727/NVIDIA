from ultralytics import YOLO
import torch

print("Torch CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("models/weights/best.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    save=True
)

print("Inference completed")