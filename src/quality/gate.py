import cv2

from config import settings

class QualityGate:
    def __init__(self, min_width: int = settings.quality.min_width,
                 min_height: int = settings.quality.min_height,
                 min_conf: float = settings.quality.min_confidence,
                 blur_threshold: float = settings.quality.blur_threshold,
                 min_aspect: float = settings.quality.min_aspect):
        self.min_width = min_width
        self.min_height = min_height
        self.min_conf = min_conf
        self.blur_threshold = blur_threshold
        self.min_aspect = min_aspect
        self.last_reason: str = ""

    def is_valid(self, plate_bbox, confidence, frame):
        # 1. Confidence check (cheapest, first)
        if confidence < self.min_conf:
            self.last_reason = "conf"
            return False

        # 2. Size check (numeric only, stays on CPU)
        x1, y1, x2, y2 = self._clip(plate_bbox, frame)
        w, h = x2 - x1, y2 - y1
        if w < self.min_width or h < self.min_height:
            self.last_reason = "size"
            return False

        # 3. Aspect check: plates are wide, reject square/noise boxes
        if h <= 0 or (w / h) < self.min_aspect:
            self.last_reason = "aspect"
            return False

        # 4. Blur check (Laplacian variance).
        # Phase 14 decision: KEEPS CPU. One small-crop cvtColor+Laplacian is
        # ~0.1ms vs ms-scale inference; a GPU port buys nothing. Revisit only
        # if the quality_gate stage ever dominates the baseline report.
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self.last_reason = "size"
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < self.blur_threshold:
            self.last_reason = "blur"
            return False
        self.last_reason = ""
        return True

    @staticmethod
    def _clip(bbox, frame):
        h, w = frame.shape[0], frame.shape[1]
        x1, y1, x2, y2 = bbox
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))