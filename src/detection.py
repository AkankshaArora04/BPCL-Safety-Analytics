from ultralytics import YOLO

# load model once
model = YOLO("yolov8n.pt")

def detect_objects(image):
    results = model(image)
    return results