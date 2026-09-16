from pydantic_settings import BaseSettings

class GpuSettings(BaseSettings):
    enabled: bool = True
    device: str = "cuda:0"
    fp16: bool = True


class DetectionSettings(BaseSettings):    
    vehicle_model: str = "models/yolov8n.pt"
    plate_model: str = "models/best.pt"
    vehicle_class_ids: list[int] = [2, 3, 5, 7]
    confidence_threshold: float = 0.3
    device: str = "cuda:0"
    verbose: bool = False
    vehicle_interval: int = 3
    plate_interval: int = 3
    imgsz: int = 640
    plate_imgsz: int = 320
    use_tensorrt: bool = False


class RecognitionSettings(BaseSettings):
    languages: list[str] = ["en"]
    gpu: bool = True
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    no_read_confidence: float = 0.0
    device: str = "cuda:0"
    interval_ms: int = 200
    batch_size: int = 8
    min_quality_gain: float = 0.0
    stop_confidence: float = 0.7
    easyocr_detector: bool = False
    quantize: bool = False
    min_ocr_conf: float = 0.5
    min_plate_length: int = 4
    upscale_target: int = 300
    upscale_max_factor: float = 4.0
    sharpen: bool = False
    plate_format: str = "^[A-Z]{3}[0-9]{3,4}$"


# for paddle
# class RecognitionSettings(BaseSettings):
#     lang: str = "en"
#     gpu: bool = True
#     allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#     no_read_confidence: float = 0.0
#     show_log: bool = False

class QualitySettings(BaseSettings):
    min_width: int = 40
    min_height: int = 12
    min_confidence: float = 0.45
    blur_threshold: float = 90.0
    min_aspect: float = 1.5


class TrackingSettings(BaseSettings):
    stale_timeout: float = 2.0

class VideoSettings(BaseSettings):
    queue_size: int = 30
    thread_join_timeout: float = 1.0
    hardware_decode: bool = False
    gpu_frames: bool = False
    mode: str = "offline"
    max_decode_failures: int = 60
    target_fps: int = 30
    realtime_pacing: bool = True


class VisualizationSettings(BaseSettings):
    enabled: bool = True
    track_box_color: tuple[int, int, int] = (0, 255, 0)
    track_box_thickness: int = 3
    
    label_color: tuple[int, int, int] = (0, 255, 0)
    label_font_scale: float = 1.2
    label_thickness: int = 3
    
    plate_text_color: tuple[int, int, int] = (0, 255, 255)
    plate_font_scale: float = 1.6
    plate_thickness: int = 4
    
    hud_left: int = 15
    hud_start_y: int = 45
    hud_font_scale: float = 1.0
    hud_color: tuple[int, int, int] = (0, 0, 255)
    hud_thickness: int = 3
    hud_line_spacing: int = 40
    output_width: int = 1280
    output_height: int = 720
    jpeg_quality: int = 80
    sidebar_enabled: bool = True
    sidebar_width: int = 320
    sidebar_max_rows: int = 8

class DemoSettings(BaseSettings):
    video_url: str = "/home/smartgov/Pictures/sample/output.mp4"
    camera_id: str = "DEMO-CAM-1"
    host: str = "0.0.0.0"
    port: int = 5000

class Settings(BaseSettings):    
    gpu: GpuSettings = GpuSettings()
    detection: DetectionSettings = DetectionSettings()
    recognition: RecognitionSettings = RecognitionSettings()
    quality: QualitySettings = QualitySettings()
    tracking: TrackingSettings = TrackingSettings()
    video: VideoSettings = VideoSettings()
    visualization: VisualizationSettings = VisualizationSettings()
    demo: DemoSettings = DemoSettings()


settings = Settings()