import cv2
import threading
import queue

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
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                return
            
            # If full, dump the oldest frame to catch up to real-time
            if self.Q.full():
                try:
                    self.Q.get_nowait()
                except queue.Empty:
                    pass
                    
            self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=self.join_timeout)
        self.stream.release()