import cv2

from config import settings

class QualityGate:
    def __init__(self, min_width: int = settings.quality.min_width,
                 min_height: int = settings.quality.min_height,
                 min_conf: float = settings.quality.min_confidence,
                 blur_threshold: float = settings.quality.blur_threshold):
        self.min_width = min_width
        self.min_height = min_height
        self.min_conf = min_conf
        self.blur_threshold = blur_threshold

    def is_valid(self, plate_bbox, confidence, frame):
        # 1. Confidence check
        if confidence < self.min_conf:
            return False
            
        # 2. Size check
        x1, y1, x2, y2 = plate_bbox
        w, h = x2 - x1, y2 - y1
        if w < self.min_width or h < self.min_height:
            return False
            
        # 3. Blur check (Laplacian variance)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
            
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var >= self.blur_threshold