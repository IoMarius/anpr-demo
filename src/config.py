from pydantic_settings import BaseSettings

class DetectionSettings(BaseSettings):
    vehicle_model: str = "models/yolov8n.pt"
    plate_model: str = "models/plate_yolov8n.pt"
    vehicle_class_ids: list[int] = [2, 3, 5, 7]
    confidence_threshold: float = 0.25
    verbose: bool = False


class RecognitionSettings(BaseSettings):
    languages: list[str] = ["en"]
    gpu: bool = True
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    no_read_confidence: float = 0.0

# for paddle
# class RecognitionSettings(BaseSettings):
#     lang: str = "en"
#     gpu: bool = True
#     allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#     no_read_confidence: float = 0.0
#     show_log: bool = False

class QualitySettings(BaseSettings):
    min_width: int = 80
    min_height: int = 25
    min_confidence: float = 0.6
    blur_threshold: float = 100.0


class TrackingSettings(BaseSettings):
    stale_timeout: float = 2.0

class VideoSettings(BaseSettings):
    queue_size: int = 30
    thread_join_timeout: float = 1.0


class VisualizationSettings(BaseSettings):
    track_box_color: tuple[int, int, int] = (0, 255, 0)
    track_box_thickness: int = 2
    label_offset: int = 10
    label_font_scale: float = 0.5
    label_color: tuple[int, int, int] = (0, 255, 0)
    label_thickness: int = 2
    hud_left: int = 10
    hud_start_y: int = 30
    hud_font_scale: float = 0.7
    hud_color: tuple[int, int, int] = (0, 0, 255)
    hud_thickness: int = 2
    hud_line_spacing: int = 30




class Settings(BaseSettings):    
    detection: DetectionSettings = DetectionSettings()
    recognition: RecognitionSettings = RecognitionSettings()
    quality: QualitySettings = QualitySettings()
    tracking: TrackingSettings = TrackingSettings()
    video: VideoSettings = VideoSettings()
    visualization: VisualizationSettings = VisualizationSettings()


settings = Settings()