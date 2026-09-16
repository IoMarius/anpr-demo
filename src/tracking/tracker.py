import time
from dataclasses import dataclass, field
from typing import List, Any, Tuple

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