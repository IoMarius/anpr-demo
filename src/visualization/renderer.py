import cv2
import numpy as np
from config import settings
from fusion.plate_fusion import PlateFusion

class HUD:
    @staticmethod
    def draw(frame, current_tracks, all_tracks_state, metrics,
             fx: float = 1.0, fy: float = 1.0):
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

            # Fused plate text (same criteria as emitted events): junk
            # reads that fusion rejects are never painted on screen.
            # Last observed plate box in red (drawn here on the viz copy so
            # shared/preloaded source frames are never mutated).
            if state and state.plate_observations:
                last = state.plate_observations[-1]
                px1, py1, px2, py2 = last["bbox"]
                cv2.rectangle(frame,
                              (int(px1 * fx), int(py1 * fy)),
                              (int(px2 * fx), int(py2 * fy)),
                              (0, 0, 255), 2)
                plate_text, _ = PlateFusion.fuse(state.plate_observations)
                if plate_text:
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

    @staticmethod
    def draw_sidebar(frame, tracker):
        """Left detection bar: persistent recent reads.

        Rows come from tracker.recent_reads (60s history), not live tracks,
        so a read stays visible after its car leaves or its track ID rolls.
        Only fused, event-grade text is recorded there.
        """
        v_cfg = settings.visualization
        width = v_cfg.sidebar_width
        height = frame.shape[0]
        bar = np.full((height, width, 3), 30, dtype=np.uint8)

        cv2.putText(bar, "DETECTIONS", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    cv2.LINE_AA)

        readers = sorted(tracker.recent_reads.items(),
                         key=lambda kv: kv[1]["time"], reverse=True)
        readers = readers[:v_cfg.sidebar_max_rows]

        if not readers:
            cv2.putText(bar, "NO READS YET", (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2,
                        cv2.LINE_AA)
            return np.hstack([bar, frame])

        top, row_h = 48, max(1, (height - 48) // v_cfg.sidebar_max_rows)
        for track_id, entry in readers:
            thumb = entry["thumb"]
            text, conf = entry["text"], entry["conf"]
            th, tw = thumb.shape[0], thumb.shape[1]
            scale = min(160 / max(tw, 1), (row_h - 8) / max(th, 1))
            thumb = cv2.resize(thumb,
                               (max(1, int(tw * scale)),
                                max(1, int(th * scale))),
                               interpolation=cv2.INTER_LINEAR)
            th, tw = thumb.shape[0], thumb.shape[1]
            y = top + 4
            bar[y:y + th, 8:8 + tw] = thumb

            tx = 180
            cv2.putText(bar, f"ID {track_id}", (tx, top + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(bar, f"{text}", (tx, top + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(bar, f"{conf:.2f}", (tx, top + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        cv2.LINE_AA)
            top += row_h

        return np.hstack([bar, frame])