import cv2
import time
from video.source import VideoSource
from detection.vehicle_detector import VehicleDetector
from detection.plate_detector import PlateDetector
from tracking.tracker import TrackManager
from quality.gate import QualityGate
from recognition.recognizer import PlateRecognizer
from fusion.plate_fusion import PlateFusion
from events.models import PlateEvent
from visualization.renderer import HUD
from metrics import PipelineMetrics
from config import settings
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
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            
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
    source = VideoSource(VIDEO_URL)
    vehicle_det = VehicleDetector()
    plate_det = PlateDetector()
    tracker = TrackManager()
    gate = QualityGate()
    ocr = PlateRecognizer()
    fusion = PlateFusion()
    metrics = PipelineMetrics()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("Video stream available at http://172.18.64.20:5000/")
    print("Pipeline started. Press 'q' to quit.")
    
    try:
        while source.more():
            frame = source.read()
            if frame is None:
                continue

            current_time = time.time()
            
            # 1. Vehicle Detection
            vehicles = vehicle_det.detect(frame)
            
            # 2. Tracking
            active_tracks = tracker.update(vehicles, current_time)
            metrics.update(len(active_tracks))
            
            # 3. Two-Stage Plate Detection & OCR
            for track_id, v_bbox in active_tracks:
                plate_result = plate_det.detect(frame, v_bbox)
                
                if plate_result:
                    p_bbox, p_conf = plate_result["bbox"], plate_result["conf"]
                    
                    # Quality Gate
                    if gate.is_valid(p_bbox, p_conf, frame):
                        # Crop and OCR
                        px1, py1, px2, py2 = p_bbox
                        plate_crop = frame[py1:py2, px1:px2]
                        
                        text, ocr_conf = ocr.recognize(plate_crop)
                        if text:
                            tracker.add_plate_observation(track_id, text, ocr_conf, p_bbox)
                            metrics.mark_recognition()
                            
                            # Draw plate box for visualization
                            # Draw RED plate box instantly only on this specific frame
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                            # cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

            # 4. Process Stale Tracks & Emit Events
            # Extract finished tracks from the tracker's internal state
            stale_ids = [tid for tid, state in tracker.active_tracks.items() 
                         if current_time - state.last_seen > tracker.stale_timeout]
            
            for tid in stale_ids:
                state = tracker.active_tracks[tid]
                fused_plate, fused_conf = fusion.fuse(state.plate_observations)
                
                if fused_plate:
                    event = PlateEvent(
                        track_id=tid,
                        plate=fused_plate,
                        confidence=fused_conf,
                        first_seen=str(state.first_seen),
                        last_seen=str(state.last_seen),
                        camera_id=CAMERA_ID
                    )
                    print(f"\n[EVENT] {event}\n") # The only output contract[cite: 1]
                
                del tracker.active_tracks[tid]

            # 5. Visualization            
            render_frame = HUD.draw(frame.copy(), active_tracks, tracker.active_tracks, metrics)
            # cv2.imshow("ANPR Pipeline", render_frame)
            
            global latest_frame
            with frame_lock:
                latest_frame = render_frame.copy()
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
    finally:
        source.stop()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()