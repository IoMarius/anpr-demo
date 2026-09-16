import time
from datetime import datetime, timezone

import cv2
import numpy as np

from config import settings
from video.frames import GPUFrame
from visualization.renderer import HUD
from events.models import PlateEvent


class ANPRPipeline:
    """Logical pipeline stages (migration plan Phase 17).

    Source -> Vehicle Detection -> Tracking -> Plate Scheduling ->
    Plate Detection -> Quality Gate -> OCR Scheduling -> OCR ->
    Fusion -> Events -> Visualization.

    Stages share only small CPU metadata; image tensors stay GPU-resident
    where the frame source provides them. No threading inside: stages run
    inline today so each step stays measurable; workers can be added later
    behind these same boundaries.
    """

    def __init__(self, vehicle_det, plate_det, tracker, gate, ocr,
                 fusion, metrics, camera_id: str = settings.demo.camera_id):
        self.vehicle_det = vehicle_det
        self.plate_det = plate_det
        self.tracker = tracker
        self.gate = gate
        self.ocr = ocr
        self.fusion = fusion
        self.metrics = metrics
        self.camera_id = camera_id
        self._frame_idx = 0
        self._cached_vehicles: list = []

    def process_frame(self, frame):
        self.metrics.start("total")
        if frame is None:
            self.metrics.stop("total")
            return None

        # Single controlled GPU->CPU download for legacy CPU consumers
        # (gate/OCR/viz). Detection uses the GPU tensor directly.
        np_frame: np.ndarray = frame.as_numpy()
        if isinstance(frame, GPUFrame):
            self.metrics.mark_transfer()

        current_time = time.time()
        vehicles = self.detect_vehicles(frame)
        active_tracks = self.update_tracks(vehicles, current_time)
        plate_results = self.detect_plates(frame, np_frame, active_tracks)
        jobs, job_crops = self.schedule_ocr(np_frame, plate_results,
                                            current_time)
        self.run_ocr(np_frame, jobs, job_crops)
        self.emit_events(current_time)
        viz = self.visualize(np_frame, active_tracks)

        self.metrics.stop("total")
        self.metrics.end_frame(len(active_tracks))
        return viz

    def detect_vehicles(self, frame):
        self.metrics.start("vehicle_detection")
        try:
            interval = settings.detection.vehicle_interval
            self._frame_idx += 1
            if (interval > 1 and self._cached_vehicles
                    and (self._frame_idx - 1) % interval != 0):
                self.metrics.mark_vehicle_skipped()
                return self._cached_vehicles
            self._cached_vehicles = self.vehicle_det.detect(frame)
            self.metrics.mark_vehicle_call()
            return self._cached_vehicles
        finally:
            self.metrics.stop("vehicle_detection")

    def update_tracks(self, vehicles, current_time):
        self.metrics.start("tracking")
        try:
            tracks = self.tracker.update(vehicles, current_time)
            self.metrics.update(len(tracks))
            return tracks
        finally:
            self.metrics.stop("tracking")

    def detect_plates(self, frame, np_frame, active_tracks):
        self.metrics.start("plate_detection")
        try:
            crops = []
            for track_id, v_bbox in active_tracks:
                if not self.tracker.eligible_for_plate(track_id):
                    self.metrics.mark_plate_skipped()
                    continue
                x1, y1, x2, y2 = v_bbox
                if isinstance(frame, GPUFrame):
                    crops.append((track_id, v_bbox, frame.gpu_crop(v_bbox)))
                else:
                    crops.append((track_id, v_bbox, np_frame[y1:y2, x1:x2]))
            results = self.plate_det.detect_batch(crops) if crops else {}
            self.metrics.mark_plate_call(1 if crops else 0)
            return results
        finally:
            self.metrics.stop("plate_detection")

    def schedule_ocr(self, np_frame, plate_results, current_time):
        self.metrics.start("quality_gate")
        try:
            ocr_jobs = []
            for track_id, pres in plate_results.items():
                p_bbox, p_conf = pres["bbox"], pres["conf"]
                if self.gate.is_valid(p_bbox, p_conf, np_frame):
                    ocr_jobs.append((track_id, p_bbox, p_conf))
                else:
                    self.metrics.mark_gate_reject(self.gate.last_reason)
        finally:
            self.metrics.stop("quality_gate")

        ocr_jobs.sort(key=lambda j: j[2], reverse=True)
        ocr_jobs = ocr_jobs[:settings.recognition.batch_size]
        stop_conf = settings.recognition.stop_confidence
        jobs, job_crops = [], []
        for track_id, p_bbox, _ in ocr_jobs:
            if stop_conf > 0 and self.tracker.best_observation_conf(track_id) >= stop_conf:
                self.metrics.mark_ocr_skipped()
                continue
            if not self.tracker.eligible_for_ocr(track_id, current_time):
                self.metrics.mark_ocr_skipped()
                continue
            px1, py1, px2, py2 = p_bbox
            jobs.append((track_id, p_bbox))
            job_crops.append(np_frame[py1:py2, px1:px2])
        self.metrics.mark_ocr_call(len(jobs))
        return jobs, job_crops

    def run_ocr(self, np_frame, jobs, job_crops):
        self.metrics.start("ocr")
        try:
            for (track_id, p_bbox), (text, ocr_conf) in zip(
                    jobs, self.ocr.recognize_batch(job_crops)):
                if text:
                    self.tracker.add_plate_observation(track_id, text,
                                                       float(ocr_conf), p_bbox)
                    self.metrics.mark_recognition()
                    px1, py1, px2, py2 = p_bbox
                    cv2.rectangle(np_frame, (px1, py1), (px2, py2),
                                  (0, 0, 255), 3)
                else:
                    self.metrics.mark_ocr_rejected()
        finally:
            self.metrics.stop("ocr")

    def emit_events(self, current_time):
        stale_ids = self.tracker.pop_stale_ids(current_time)
        emitted = []
        for tid in stale_ids:
            state = self.tracker.active_tracks[tid]
            fused_plate, fused_conf = self.fusion.fuse(state.plate_observations)
            if fused_plate:
                event = PlateEvent(
                    track_id=tid,
                    plate=fused_plate,
                    confidence=float(fused_conf),
                    first_seen=datetime.fromtimestamp(
                        state.first_seen, tz=timezone.utc).isoformat(),
                    last_seen=datetime.fromtimestamp(
                        state.last_seen, tz=timezone.utc).isoformat(),
                    camera_id=self.camera_id
                )
                print(f"\n[EVENT] {event}\n")
                emitted.append(event)
            del self.tracker.active_tracks[tid]
        return emitted

    def visualize(self, np_frame, active_tracks):
        cfg = settings.visualization
        if not cfg.enabled:
            return None
        self.metrics.start("rendering")
        try:
            fh, fw = np_frame.shape[0], np_frame.shape[1]
            if (cfg.output_width, cfg.output_height) != (fw, fh):
                viz = cv2.resize(np_frame,
                                 (cfg.output_width, cfg.output_height))
                fx, fy = cfg.output_width / fw, cfg.output_height / fh
                tracks = [(tid, (int(x1 * fx), int(y1 * fy),
                                 int(x2 * fx), int(y2 * fy)))
                          for tid, (x1, y1, x2, y2) in active_tracks]
            else:
                viz, tracks = np_frame.copy(), active_tracks
            return HUD.draw(viz, tracks, self.tracker.active_tracks,
                            self.metrics)
        finally:
            self.metrics.stop("rendering")
