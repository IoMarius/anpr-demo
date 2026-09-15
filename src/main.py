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

def main():
    # Configuration
    VIDEO_URL = "../sample/highway-4k.mp4"
    CAMERA_ID = "demo-camera-1"
    
    # Initialize components
    print("Initializing pipeline components...")
    source = VideoSource(VIDEO_URL)
    vehicle_det = VehicleDetector("yolov8n.pt")
    plate_det = PlateDetector("plate_yolov8n.pt")
    tracker = TrackManager(stale_timeout=2.0)
    gate = QualityGate()
    ocr = PlateRecognizer()
    fusion = PlateFusion()
    metrics = PipelineMetrics()

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
                            tracker.add_plate_observation(track_id, text, ocr_conf)
                            metrics.mark_recognition()
                            
                            # Draw plate box for visualization
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

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
            render_frame = HUD.draw(frame.copy(), active_tracks, metrics)
            cv2.imshow("ANPR Pipeline", render_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        source.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()