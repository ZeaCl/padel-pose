from ultralytics import YOLO

# Carga el modelo de pose YOLOv8n (nano) - es el más rápido; también hay 'yolov8s-pose.pt', etc.
model = YOLO("yolov8n-pose.pt")

# Usa la cámara (source=0), mostrando la ventana (show=True)
# device="mps" para Mac M1/M2 o "cpu"/"cuda" según corresponda.
results = model.predict(source=0, show=True, device="mps")