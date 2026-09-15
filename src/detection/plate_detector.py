from ultralytics import YOLO

from config import settings

class PlateDetector:
    def __init__(
        self, model_path: str = settings.detection.plate_model,
        conf_threshold: float = settings.detection.confidence_threshold,                 
        verbose: bool = settings.detection.verbose,
        device: str = settings.detection.device
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.verbose = verbose
        self.device = device

    def detect(self, frame, vehicle_bbox):
        x1, y1, x2, y2 = vehicle_bbox
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
            
        results = self.model(
            crop, 
            conf=self.conf_threshold, 
            verbose=self.verbose, 
            device=self.device);
        
        best_plate = None
        max_conf = 0.0
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                    
                    # Translate coordinates back to the original frame
                    best_plate = {
                        "bbox": (px1 + x1, py1 + y1, px2 + x1, py2 + y1), 
                        "conf": conf
                    }
                    max_conf = conf
                    
        return best_plate