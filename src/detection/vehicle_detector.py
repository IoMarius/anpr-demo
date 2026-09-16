from ultralytics import YOLO
import torch
from config import settings
from utils.device import resolve_device, log_gpu_status, touch_cuda
from utils.gpu_preprocess import letterbox_gpu, unscale_boxes
from video.frames import GPUFrame, CPUFrame

class VehicleDetector:
    """Motion owner: Ultralytics YOLO tracking is the single source of track IDs.

    TrackManager owns application state (observations, timestamps, fusion
    input) and must NOT run its own motion association.
    """
    def __init__(
        self,
        model_path: str = settings.detection.vehicle_model,
        vehicle_classes: list[int] = settings.detection.vehicle_class_ids,
        conf_threshold: float = settings.detection.confidence_threshold,
        verbose: bool = settings.detection.verbose,
        device: str = settings.detection.device
    ):
        self.model = YOLO(model_path)
        self.device = resolve_device(device, settings.gpu.enabled)
        if self.device.startswith("cuda") and not touch_cuda(self.device):
            self.device = "cpu"
        # Phase 18: FP16 only where CUDA actually runs; CPU stays FP32 with
        # no extra kwargs (avoids deprecated-flag warnings on CPU boxes).
        self.use_fp16 = (settings.gpu.fp16 and torch.cuda.is_available()
                     and self.device.startswith("cuda"))
        self._precision = {"quantize": "fp16"} if self.use_fp16 else {}
        self.vehicle_classes = vehicle_classes
        self.conf_threshold = conf_threshold
        self.verbose = verbose
        log_gpu_status(f"vehicle_detector resolved={self.device} quantize={'fp16' if self.use_fp16 else 'fp32'}")

    def detect(self, frame):
        if isinstance(frame, GPUFrame):
            return self._detect_gpu(frame)
        arr = frame.data if isinstance(frame, CPUFrame) else frame
        return self._detect_numpy(arr)

    def _detect_numpy(self, arr):
        results = self.model.track(
            arr,
            classes=self.vehicle_classes,
            conf=self.conf_threshold,
            persist=True,
            verbose=self.verbose,
            device=self.device,
            **self._precision
        )
        return self._parse(results)

    def _detect_gpu(self, frame: GPUFrame):
        h, w = frame.height, frame.width
        batch, scale, pad_w, pad_h = letterbox_gpu(frame.tensor)
        if self.use_fp16:
            batch = batch.half()
        results = self.model.track(
            batch,
            classes=self.vehicle_classes,
            conf=self.conf_threshold,
            persist=True,
            verbose=self.verbose,
            device=self.device,
            **self._precision
        )
        vehicles = []
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = unscale_boxes(result.boxes.xyxy, scale, pad_w, pad_h, w, h)
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                for box, track_id, class_id in zip(boxes.cpu().numpy(), track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    vehicles.append({
                        "bbox": (x1, y1, x2, y2),
                        "track_id": track_id,
                        "class_id": class_id,
                    })
        return vehicles

    def _parse(self, results):
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