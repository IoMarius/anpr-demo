import time

class PipelineMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.active_tracks = 0
        self.recognized_plates = 0

    def update(self, active_tracks):
        self.frame_count += 1
        self.active_tracks = active_tracks

    def mark_recognition(self):
        self.recognized_plates += 1

    def get_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0
        return self.frame_count / elapsed