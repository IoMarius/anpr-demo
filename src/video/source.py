import cv2
import threading
import queue
import time

import torch

from config import settings
from video.frames import CPUFrame, GPUFrame
from video.gstreamer import build_nvdec_pipeline, probe_report
from utils.device import resolve_device


class VideoSource:
    def __init__(self, source_path: str, queue_size: int = settings.video.queue_size,
                 join_timeout: float = settings.video.thread_join_timeout,
                 hardware_decode: bool = settings.video.hardware_decode,
                 gpu_frames: bool = settings.video.gpu_frames,
                 mode: str = settings.video.mode,
                 device: str = settings.detection.device):
        self.source_path = source_path
        self.mode = mode
        self.join_timeout = join_timeout
        self.device = resolve_device(device, settings.gpu.enabled)
        self.gpu_frames = gpu_frames and torch.cuda.is_available()
        if gpu_frames and not torch.cuda.is_available():
            print("[NVDEC] gpu_frames requested but CUDA unavailable, using CPU frames")
        self.backend = "cpu-ffmpeg"
        self.frames_decoded = 0
        self.frames_dropped = 0

        self.stream = self._open(source_path, hardware_decode)
        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)

        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _open(self, source_path: str, hardware_decode: bool):
        if hardware_decode:
            print(probe_report(hardware_decode))
            pipeline, decoder = build_nvdec_pipeline(source_path)
            print(f"[NVDEC] trying GStreamer pipeline with {decoder}")
            stream = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if stream.isOpened():
                self.backend = f"gstreamer-{decoder}"
                print(f"[NVDEC] using hardware decoder: {decoder}")
                return stream
            print("[NVDEC] GStreamer NVDEC open failed, falling back to CPU decode")
        stream = cv2.VideoCapture(source_path)
        if not stream.isOpened():
            raise ValueError(f"Failed to open video source: {source_path}")
        return stream

    def _wrap(self, frame, frame_id: int):
        if self.gpu_frames:
            tensor = torch.from_numpy(frame).to(self.device)
            return GPUFrame(tensor=tensor, frame_id=frame_id)
        return CPUFrame(data=frame, frame_id=frame_id)

    def _update(self):
        while not self.stopped:
            if self.mode == "live" and self.Q.full():
                try:
                    self.Q.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            if not self.Q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    if self.mode == "live":
                        time.sleep(0.01)
                        continue
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.frames_decoded += 1
                item = self._wrap(frame, self.frames_decoded)
                try:
                    if self.mode == "live":
                        self.Q.put_nowait(item)
                    else:
                        while not self.stopped:
                            try:
                                self.Q.put(item, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                except queue.Full:
                    self.frames_dropped += 1
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
