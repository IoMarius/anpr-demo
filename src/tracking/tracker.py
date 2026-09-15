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
    plate_observations: List[Any] = field(default_factory=list) # Holds OCR reads

class TrackManager:
    def __init__(self, stale_timeout: float = settings.tracking.stale_timeout,
                 mock_id_space: int = settings.tracking.mock_id_space):
        # In a real implementation, you initialize ByteTrack/SORT here.
        # self.vision_tracker = ByteTracker(...)
        self.active_tracks = {}
        self.stale_timeout = stale_timeout
        self.mock_id_space = mock_id_space

    def update(self, vehicles, current_time):
        """
        vehicles: List of dicts with 'bbox' and 'class_id'
        Returns: List of active track IDs and their current bboxes
        """
        # WARNING: Stub for actual ByteTrack update. 
        # tracked_results = self.vision_tracker.update(vehicles)
        tracked_results = self._mock_track(vehicles) 
        
        current_frame_tracks = []
        
        for bbox, track_id, class_id in tracked_results:
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = TrackState(
                    id=track_id,
                    first_seen=current_time,
                    last_seen=current_time,
                    vehicle_class=class_id
                )
            else:
                self.active_tracks[track_id].last_seen = current_time
                
            current_frame_tracks.append((track_id, bbox))
                
        self._cleanup(current_time)
        return current_frame_tracks

    def add_plate_observation(self, track_id, text, confidence):
        if track_id in self.active_tracks:
            self.active_tracks[track_id].plate_observations.append({
                "text": text,
                "conf": confidence
            })

    def _cleanup(self, current_time):
        stale_ids = [tid for tid, state in self.active_tracks.items() 
                     if current_time - state.last_seen > self.stale_timeout]
        for tid in stale_ids:
            # Note: This is where you will eventually trigger Fusion and emit a PlateEvent
            del self.active_tracks[tid]

    def _mock_track(self, vehicles):
        # Placeholder mapping logic to keep the pipeline compiling
        return [(v['bbox'], hash(str(v['bbox'])) % self.mock_id_space, v['class_id']) for v in vehicles]