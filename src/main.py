import cv2
import time
from video.source import VideoSource
from detection.vehicle_detector import VehicleDetector
from detection.plate_detector import PlateDetector
from tracking.tracker import TrackManager
from quality.gate import QualityGate
from recognition.recognizer import PlateRecognizer
from fusion.plate_fusion import PlateFusion
from pipeline import ANPRPipeline
from metrics import PipelineMetrics
from config import settings
from utils.device import log_gpu_status
from utils.preflight import preflight_report
import threading
from flask import Flask, Response

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1) # Prevent thrashing before video starts
                continue
            
            # Compress the frame to JPEG for the browser
            ret, buffer = cv2.imencode(
                '.jpg', latest_frame,
                [cv2.IMWRITE_JPEG_QUALITY, settings.visualization.jpeg_quality])
            
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        time.sleep(0.03) # Limit stream to ~30 FPS

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
def start_server():
    HOST = settings.demo.host
    PORT = settings.demo.port
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)

def main():
    # Configuration
    VIDEO_URL = settings.demo.video_url
    CAMERA_ID = settings.demo.camera_id
    
    # Initialize components
    print("Initializing pipeline components...")
    print(preflight_report(VIDEO_URL))
    log_gpu_status("startup")
    print(f"[CONFIG] gpu.enabled={settings.gpu.enabled} device={settings.detection.device} "
          f"plate_interval={settings.detection.plate_interval} "
          f"ocr_interval_ms={settings.recognition.interval_ms} "
          f"video.mode={settings.video.mode} hw_decode={settings.video.hardware_decode} "
          f"gpu_frames={settings.video.gpu_frames} "
          f"target_fps={settings.video.target_fps} pacing={settings.video.realtime_pacing} "
          f"tensorrt={settings.detection.use_tensorrt}")
    source = VideoSource(VIDEO_URL)
    print(f"[VIDEO] backend={source.backend} mode={source.mode}")
    pipeline = ANPRPipeline(
        vehicle_det=VehicleDetector(),
        plate_det=PlateDetector(),
        tracker=TrackManager(),
        gate=QualityGate(),
        ocr=PlateRecognizer(),
        fusion=PlateFusion(),
        metrics=PipelineMetrics(),
        camera_id=CAMERA_ID,
    )

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print(f"Video stream available at http:{settings.demo.host}//:{settings.demo.port}/")
    print("Pipeline started. Press 'q' to quit.")

    pace_fps = 0.0
    if settings.video.realtime_pacing and settings.video.target_fps > 0:
        pace_fps = min(float(settings.video.target_fps), source.fps)
        print(f"[VIDEO] pacing to {pace_fps:.1f} FPS "
              f"(source={source.fps:.1f} target={settings.video.target_fps})")
    next_deadline = time.time()

    try:
        while source.more():
            pipeline.metrics.start("decode")
            frame = source.read()
            pipeline.metrics.stop("decode")
            render_frame = pipeline.process_frame(frame)

            if pace_fps > 0:
                next_deadline += 1.0 / pace_fps
                delay = next_deadline - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # Pipeline slower than realtime; resync instead of
                    # accumulating debt.
                    next_deadline = time.time()

            if render_frame is None:
                continue
            global latest_frame
            with frame_lock:
                latest_frame = render_frame.copy()

        if source.error:
            raise RuntimeError(source.error)
    finally:
        source.stop()        

if __name__ == "__main__":
    main()