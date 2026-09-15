from ultralytics import YOLO

from config import settings

class VehicleDetector:
    def __init__(self, model_path: str = settings.detection.vehicle_model,
                 vehicle_classes: list[int] = settings.detection.vehicle_class_ids,
                 conf_threshold: float = settings.detection.confidence_threshold,
                 verbose: bool = settings.detection.verbose):
        self.model = YOLO(model_path)
        self.vehicle_classes = vehicle_classes
        self.conf_threshold = conf_threshold
        self.verbose = verbose

    def detect(self, frame):
        results = self.model(frame, classes=self.vehicle_classes,
                             conf=self.conf_threshold, verbose=self.verbose)
        vehicles = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                vehicles.append({
                    "bbox": (x1, y1, x2, y2), 
                    "conf": float(box.conf[0]), 
                    "class_id": int(box.cls[0])
                })
        return vehicles