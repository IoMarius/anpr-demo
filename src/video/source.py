import cv2
import threading
import queue
import time

import torch

from config import settings
from video.frames import CPUFrame, GPUFrame
from video.gstreamer import (build_nvdec_pipeline, probe_report, probe_codec,
                             apply_ffmpeg_hwaccel)
from utils.device import resolve_device


class VideoSource:
    def __init__(self, source_path: str, queue_size: int = settings.video.queue_size,
                 join_timeout: float = settings.video.thread_join_timeout,
                 hardware_decode: bool = settings.video.hardware_decode,
                 gpu_frames: bool = settings.video.gpu_frames,
                 mode: str = settings.video.mode,
                 device: str = settings.detection.device,
                 max_failures: int = settings.video.max_decode_failures):
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
        self.max_failures = max_failures
        self._consec_failures = 0
        self.error: str | None = None

        self.stream = self._open(source_path, hardware_decode)
        self.fps = self._probe_fps()
        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)

        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _open(self, source_path: str, hardware_decode: bool):
        self.codec = probe_codec(source_path)
        if hardware_decode:
            print(probe_report(hardware_decode) + f" codec={self.codec}")
            pipeline, decoder = build_nvdec_pipeline(
                source_path, codec=self.codec)
            if pipeline is not None:
                print(f"[NVDEC] trying GStreamer pipeline with {decoder} "
                      f"for {self.codec}")
                stream = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if stream.isOpened():
                    self.backend = f"gstreamer-{decoder}({self.codec})"
                    print(f"[NVDEC] using hardware decoder: {decoder}")
                    return stream
                print("[NVDEC] GStreamer NVDEC open failed, "
                      "trying FFmpeg HW acceleration")
            else:
                print(f"[NVDEC] no GStreamer decoder plugin for {self.codec}, "
                      "trying FFmpeg HW acceleration")
            stream = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
            if stream.isOpened() and apply_ffmpeg_hwaccel(stream):
                self.backend = f"ffmpeg-hwaccel({self.codec})"
                print(f"[NVDEC] FFmpeg HW acceleration requested "
                      f"for {self.codec} (verify with nvidia-smi dmon)")
                return stream
            print("[NVDEC] FFmpeg HW acceleration unavailable, "
                  "falling back to CPU decode")
        stream = cv2.VideoCapture(source_path)
        if not stream.isOpened():
            raise ValueError(f"Failed to open video source: {source_path}")
        self.backend = f"cpu-ffmpeg({self.codec})"
        return stream

    def _probe_fps(self) -> float:
        try:
            fps = float(self.stream.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps = 0.0
        if not fps or fps != fps or fps <= 0 or fps > 240:
            return 30.0
        return fps

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
                    self._consec_failures += 1
                    if self._consec_failures >= self.max_failures:
                        self.error = (
                            f"decoder produced no frames "
                            f"{self.max_failures}x in a row "
                            f"(backend={self.backend} codec={self.codec} "
                            f"source={self.source_path}). "
                            f"For AV1 without NVDEC support, transcode to "
                            f"H264 or enable hardware_decode on NVDEC-AV1 "
                            f"hardware.")
                        print(f"[VIDEO] FATAL {self.error}")
                        self.stopped = True
                        return
                    if self.mode == "live":
                        time.sleep(0.01)
                        continue
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._consec_failures = 0
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
