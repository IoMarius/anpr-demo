import time
from dataclasses import dataclass, field
from typing import List, Any

from config import settings

@dataclass
class TrackState:
    id: int
    first_seen: float
    last_seen: float
    vehicle_class: int
    plate_observations: List[Any] = field(default_factory=list)

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