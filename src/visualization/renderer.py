import cv2


class HUD:

    @staticmethod
    def draw(frame, current_tracks, all_tracks_state, metrics):

        for track_id, bbox in current_tracks:
            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            state = all_tracks_state.get(track_id)

            # Large ID
            id_text = f"ID: {track_id}"

            cv2.putText(
                frame,
                id_text,
                (x1, max(y1 - 45, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            # Large plate number
            if state and state.plate_observations:
                plate_text = state.plate_observations[-1]["text"]

                cv2.putText(
                    frame,
                    plate_text,
                    (x1, max(y1 - 5, 65)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA
                )

        # Metrics HUD
        fps = metrics.get_fps()

        hud_text = [
            f"FPS: {fps:.1f}",
            f"Active Tracks: {metrics.active_tracks}",
            f"Total Recognitions: {metrics.recognized_plates}",
        ]

        y_offset = 45

        for text in hud_text:
            cv2.putText(
                frame,
                text,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            y_offset += 40

        return frame