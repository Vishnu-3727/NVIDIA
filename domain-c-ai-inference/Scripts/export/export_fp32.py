from ultralytics import YOLO

model = YOLO("models/weights/best.pt")

model.export(
    format="onnx",
    opset=17,
    simplify=True
)