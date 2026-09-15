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
    def __init__(self, stale_timeout : float = settings.tracking.stale_timeout):
        self.active_tracks = {}
        self.stale_timeout = stale_timeout

    def update(self, vehicles, current_time):
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