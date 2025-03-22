from ultralytics import YOLO

# Carga un modelo preentrenado, por ejemplo 'yolov8n.pt' (modelo "nano")
model = YOLO("yolov8n.pt")

# Ejecuta la detección en tiempo real:
# - source=0 para la cámara
# - show=True para mostrar la ventana
# - device="mps" para usar aceleración MPS
results = model.predict(source=0, show=True, device="mps")