from ultralytics import YOLO

model = YOLO("models/weights/best.pt")

print("\nModel loaded successfully\n")

print("Classes:")
print(model.names)

print("\nNumber of classes:", len(model.names))