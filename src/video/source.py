import cv2
import threading
import queue
import time

from config import settings

class VideoSource:
    def __init__(self, source_path: str, queue_size: int = settings.video.queue_size,
                 join_timeout: float = settings.video.thread_join_timeout):
        self.stream = cv2.VideoCapture(source_path)
        if not self.stream.isOpened():
            raise ValueError(f"Failed to open video source: {source_path}")
            
        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)
        self.join_timeout = join_timeout
        
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    # Rewind to frame 0 for continuous looping
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.Q.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=self.join_timeout)
        self.stream.release()