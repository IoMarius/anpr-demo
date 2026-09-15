from ultralytics import YOLO
from config import settings

class VehicleDetector:
    def __init__(
        self,
        model_path: str = settings.detection.vehicle_model,
        vehicle_classes: list[int] = settings.detection.vehicle_class_ids,
        conf_threshold: float = settings.detection.confidence_threshold,
        verbose: bool = settings.detection.verbose,
        device: str = settings.detection.device
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.vehicle_classes = vehicle_classes
        self.conf_threshold = conf_threshold
        self.verbose = verbose

    def detect(self, frame):
        results = self.model.track(
            frame,
            classes=self.vehicle_classes,
            conf=self.conf_threshold,
            persist=True,
            verbose=self.verbose,
            device=self.device
        )
        vehicles = []
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    vehicles.append({
                        "bbox": (x1, y1, x2, y2),
                        "track_id": track_id,
                        "class_id": class_id,
                    })
        return vehicles