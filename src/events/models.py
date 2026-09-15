from dataclasses import dataclass

@dataclass
class PlateEvent:
    track_id: int
    plate: str
    confidence: float
    first_seen: str
    last_seen: str
    camera_id: str