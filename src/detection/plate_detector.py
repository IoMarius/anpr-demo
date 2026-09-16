from ultralytics import YOLO
import torch

from config import settings
from utils.device import resolve_device, log_gpu_status, touch_cuda
from utils.gpu_preprocess import letterbox_gpu, unscale_boxes

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
        self.device = resolve_device(device, settings.gpu.enabled)
        if self.device.startswith("cuda") and not touch_cuda(self.device):
            self.device = "cpu"
        self.use_fp16 = (settings.gpu.fp16 and torch.cuda.is_available()
                     and self.device.startswith("cuda"))
        self._precision = {"quantize": "fp16"} if self.use_fp16 else {}
        log_gpu_status(f"plate_detector resolved={self.device} quantize={'fp16' if self.use_fp16 else 'fp32'}")

    def detect_batch(self, crops):
        """Batched inference over vehicle crops.

        crops: list of (track_id, v_bbox, crop) where crop is a CPU numpy
        array or a GPU HWC uint8 tensor. Returns dict
        track_id -> {"bbox": frame-space bbox, "conf": float}.
        Runs one batched model call instead of one per vehicle.
        """
        valid = [(tid, vb, c) for tid, vb, c in crops if not _crop_empty(c)]
        if not valid:
            return {}
        if isinstance(valid[0][2], torch.Tensor):
            return self._detect_batch_gpu(valid)
        imgs = [c for _, _, c in valid]
        results = self.model(
            imgs,
            conf=self.conf_threshold,
            verbose=self.verbose,
            device=self.device,
            **self._precision)
        out = {}
        for (track_id, (x1, y1, _, _), _), result in zip(valid, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            best, max_conf = None, 0.0
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                    best = {"bbox": (px1 + x1, py1 + y1, px2 + x1, py2 + y1),
                            "conf": conf}
                    max_conf = conf
            if best:
                out[track_id] = best
        return out

    def _detect_batch_gpu(self, valid):
        prepped, meta = [], []
        for track_id, (x1, y1, x2, y2), crop in valid:
            batch, scale, pad_w, pad_h = letterbox_gpu(crop)
            if self.use_fp16:
                batch = batch.half()
            prepped.append(batch)
            meta.append((track_id, x1, y1, x2 - x1, y2 - y1,
                         scale, pad_w, pad_h))
        results = self.model(
            torch.cat(prepped),
            conf=self.conf_threshold,
            verbose=self.verbose,
            device=self.device,
            **self._precision)
        out = {}
        for (track_id, ox, oy, cw, ch, scale, pad_w, pad_h), result in zip(meta, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = unscale_boxes(result.boxes.xyxy, scale, pad_w, pad_h, cw, ch)
            confs = result.boxes.conf.flatten().tolist()
            best_idx = max(range(len(confs)), key=lambda i: confs[i])
            px1, py1, px2, py2 = map(int, boxes[best_idx].tolist())
            out[track_id] = {"bbox": (px1 + ox, py1 + oy, px2 + ox, py2 + oy),
                             "conf": confs[best_idx]}
        return out

    def detect(self, frame, vehicle_bbox):
        x1, y1, x2, y2 = vehicle_bbox
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
            
        results = self.model(
            crop, 
            conf=self.conf_threshold, 
            verbose=self.verbose, 
            device=self.device,
            **self._precision);
        
        best_plate = None
        max_conf = 0.0
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
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


def _crop_empty(c) -> bool:
    if isinstance(c, torch.Tensor):
        return c.numel() == 0
    return c.size == 0