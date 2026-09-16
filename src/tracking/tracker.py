import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Any, Tuple

import cv2
import numpy as np

from config import settings
from fusion.plate_fusion import PlateFusion

@dataclass
class TrackState:
    id: int
    first_seen: float
    last_seen: float
    vehicle_class: int
    plate_observations: List[Any] = field(default_factory=list)
    last_plate_crop: Any = None
    last_plate_text: str = ""
    last_plate_conf: float = 0.0
    last_plate_time: float = 0.0

class TrackManager:
    """Application state store, NOT a motion tracker.

    Ownership (migration plan Phase 2, Option B):
      motion/track IDs -> Ultralytics YOLO tracking (VehicleDetector)
      application state -> this manager (observations, timestamps, fusion input)
    """
    def __init__(self, stale_timeout : float = settings.tracking.stale_timeout):
        self.active_tracks = {}
        self.stale_timeout = stale_timeout
        self.frame_count = 0
        self._last_plate_frame: dict = {}
        self._last_ocr_time: dict = {}
        # Sidebar history, independent of track liveness: stale tracks are
        # deleted from active_tracks (taking their reads with them), which
        # made sidebar rows vanish seconds after appearing.
        self.recent_reads: dict = {}
        self.recent_ttl: float = 60.0
        self.recent_max: int = 32
        # Temporal crop stacking: per-track history of normalized plate
        # crops; averaging N frames buys sqrt(N) SNR against blur for
        # slow-moving far plates.
        self.stack_size: int = settings.recognition.stack_size
        self.stack_width: int = settings.recognition.stack_width
        self.stack_height: int = settings.recognition.stack_height
        self.stack_reset_ratio: float = settings.recognition.stack_reset_ratio
        self._crop_hist: dict = {}

    def update(self, vehicles, current_time):
        self.frame_count += 1
        current_frame_tracks = []
        
        for v in vehicles:
            track_id = v['track_id']
            bbox = v['bbox']
            
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = TrackState(
                    id=track_id,
                    first_seen=current_time,
                    last_seen=current_time,
                    vehicle_class=v['class_id']
                )
            else:
                self.active_tracks[track_id].last_seen = current_time
                
            current_frame_tracks.append((track_id, bbox))
                
        return current_frame_tracks

    def add_plate_observation(self, track_id, text, confidence, bbox):
        if track_id in self.active_tracks:
            self.active_tracks[track_id].plate_observations.append({
                "text": text,
                "conf": confidence,
                "bbox": bbox  # Save the coordinates here
            })

    def note_plate_crop(self, track_id, crop_thumb, text, confidence,
                        current_time):
        state = self.active_tracks.get(track_id)
        if state is None:
            return
        # Keep the thumbnail of the best read; the sidebar/HUD show fused
        # text, so the crop should represent the strongest evidence, not
        # the latest (possibly junk) read.
        if (state.last_plate_crop is None
                or float(confidence) >= state.last_plate_conf):
            state.last_plate_crop = crop_thumb
            state.last_plate_text = text
            state.last_plate_conf = float(confidence)
        state.last_plate_time = current_time
        # Record event-grade reads for the sidebar. add_plate_observation
        # runs before this, so fusion sees the new read; only fused-worthy
        # text persists (same criteria as emitted events).
        fused_text, fused_conf = PlateFusion.fuse(state.plate_observations)
        if fused_text:
            self.recent_reads[track_id] = {
                "thumb": state.last_plate_crop,
                "text": fused_text,
                "conf": float(fused_conf),
                "time": current_time,
            }
            self._prune_recent(current_time)

    def _prune_recent(self, current_time):
        stale = [tid for tid, r in self.recent_reads.items()
                 if current_time - r["time"] > self.recent_ttl]
        for tid in stale:
            del self.recent_reads[tid]
        while len(self.recent_reads) > self.recent_max:
            oldest = min(self.recent_reads,
                         key=lambda tid: self.recent_reads[tid]["time"])
            del self.recent_reads[oldest]

    def display(self, track_id) -> Tuple[str | None, float]:
        """Screen-worthy text: same fused criteria as emitted events.

        HUD/sidebar previously showed the latest raw read, so junk that
        fusion would reject was still painted on screen. Now what you see
        is what would get emitted.
        """
        state = self.active_tracks.get(track_id)
        if state is None or not state.plate_observations:
            return None, 0.0
        return PlateFusion.fuse(state.plate_observations)

    def stacked_crop(self, track_id, crop, bbox):
        """Accumulate a normalized crop; return the temporal mean.

        Crops are resized to a common shape so frames average pixel-wise.
        History resets when the plate box jumps (fast motion breaks the
        alignment assumption). Single observation returns the crop itself.
        """
        hist = self._crop_hist.get(track_id)
        if hist is None:
            hist = {"crops": deque(maxlen=self.stack_size), "bbox": None}
            self._crop_hist[track_id] = hist
        if hist["bbox"] is not None:
            px, py = hist["bbox"]
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            w = max(bbox[2] - bbox[0], 1)
            if abs(cx - px) > w * self.stack_reset_ratio:
                hist["crops"].clear()
        hist["bbox"] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        normed = cv2.resize(crop, (self.stack_width, self.stack_height),
                            interpolation=cv2.INTER_LINEAR)
        hist["crops"].append(normed.astype(np.float32))
        mean = sum(hist["crops"]) / len(hist["crops"])
        return np.clip(mean, 0, 255).astype(np.uint8)

    def eligible_for_plate(self, track_id) -> bool:
        interval = settings.detection.plate_interval
        if interval <= 1:
            return True
        last = self._last_plate_frame.get(track_id, -interval)
        if self.frame_count - last >= interval:
            self._last_plate_frame[track_id] = self.frame_count
            return True
        return False

    def eligible_for_ocr(self, track_id, current_time, quality_improved=True) -> bool:
        interval_s = settings.recognition.interval_ms / 1000.0
        if interval_s <= 0:
            return True
        last = self._last_ocr_time.get(track_id, 0.0)
        if current_time - last >= interval_s and quality_improved:
            self._last_ocr_time[track_id] = current_time
            return True
        return False

    def pop_stale_ids(self, current_time) -> list:
        return [tid for tid, state in self.active_tracks.items()
                if current_time - state.last_seen > self.stale_timeout]

    def best_observation_conf(self, track_id) -> float:
        state = self.active_tracks.get(track_id)
        if not state or not state.plate_observations:
            return 0.0
        return max(o["conf"] for o in state.plate_observations)