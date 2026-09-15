import cv2
from config import settings

class HUD:
    @staticmethod
    def draw(frame, current_tracks, all_tracks_state, metrics):
        v_cfg = settings.visualization
        
        for track_id, bbox in current_tracks:
            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                v_cfg.track_box_color,
                v_cfg.track_box_thickness
            )

            state = all_tracks_state.get(track_id)

            # Large ID
            id_text = f"ID: {track_id}"
            cv2.putText(
                frame,
                id_text,
                (x1, max(y1 - 45, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                v_cfg.label_font_scale,
                v_cfg.label_color,
                v_cfg.label_thickness,
                cv2.LINE_AA
            )

            # Large plate number (Persistent Text)
            if state and state.plate_observations:
                plate_text = state.plate_observations[-1]["text"]

                cv2.putText(
                    frame,
                    plate_text,
                    (x1, max(y1 - 5, 65)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    v_cfg.plate_font_scale,
                    v_cfg.plate_text_color,
                    v_cfg.plate_thickness,
                    cv2.LINE_AA
                )

        # Metrics HUD
        fps = metrics.get_fps()
        hud_text = [
            f"FPS: {fps:.1f}",
            f"Active Tracks: {metrics.active_tracks}",
            f"Total Recognitions: {metrics.recognized_plates}",
        ]

        y_offset = v_cfg.hud_start_y
        for text in hud_text:
            cv2.putText(
                frame,
                text,
                (v_cfg.hud_left, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                v_cfg.hud_font_scale,
                v_cfg.hud_color,
                v_cfg.hud_thickness,
                cv2.LINE_AA,
            )
            y_offset += v_cfg.hud_line_spacing

        return frame