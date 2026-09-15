import cv2

from config import settings

class HUD:
    @staticmethod
    def draw(frame, tracks, metrics):
        v = settings.visualization

        # Draw tracks and text
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), v.track_box_color, v.track_box_thickness)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - v.label_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, v.label_font_scale, v.label_color, v.label_thickness)

        # Draw Metrics HUD
        fps = metrics.get_fps()
        hud_text = [
            f"FPS: {fps:.1f}",
            f"Active Tracks: {metrics.active_tracks}",
            f"Total Recognitions: {metrics.recognized_plates}"
        ]
        
        y_offset = v.hud_start_y
        for text in hud_text:
            cv2.putText(frame, text, (v.hud_left, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, v.hud_font_scale, v.hud_color, v.hud_thickness)
            y_offset += v.hud_line_spacing
            
        return frame