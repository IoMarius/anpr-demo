import time
from collections import defaultdict, deque

STAGES = (
    "decode",
    "vehicle_detection",
    "tracking",
    "plate_detection",
    "quality_gate",
    "ocr",
    "rendering",
    "total",
)


class PipelineMetrics:
    def __init__(self, report_interval: int = 100):
        self.start_time = time.time()
        self.frame_count = 0
        self.active_tracks = 0
        self.recognized_plates = 0
        self.report_interval = report_interval
        self._timings = defaultdict(list)
        self._pending = {}
        self.vehicles_per_frame: list[int] = []
        self.plate_calls_per_frame: list[int] = []
        self.ocr_calls_per_frame: list[int] = []
        self.cpu_gpu_transfers = 0
        self.plate_skipped = 0
        self.ocr_skipped = 0
        self.vehicle_calls = 0
        self.vehicle_skipped = 0
        self.plate_gate_filtered = 0
        self.ocr_rejected_low_conf = 0
        self.gate_reject_conf = 0
        self.gate_reject_size = 0
        self.gate_reject_aspect = 0
        self.gate_reject_blur = 0
        self._frame_plate_calls = 0
        self._frame_ocr_calls = 0
        # Rolling FPS window: cumulative frame_count/elapsed includes model
        # load + warmup, so the HUD number ramps for minutes. A 60-frame
        # window shows the true current rate within ~2s at 30fps.
        self._frame_times: deque = deque(maxlen=60)

    def start(self, stage: str):
        self._pending[stage] = time.perf_counter()

    def stop(self, stage: str):
        t0 = self._pending.pop(stage, None)
        if t0 is not None:
            self._timings[stage].append(time.perf_counter() - t0)

    def update(self, active_tracks):
        self.frame_count += 1
        self.active_tracks = active_tracks
        self._frame_times.append(time.time())

    def mark_recognition(self):
        self.recognized_plates += 1

    def mark_plate_call(self, n: int = 1):
        self._frame_plate_calls += n

    def mark_ocr_call(self, n: int = 1):
        self._frame_ocr_calls += n

    def mark_transfer(self, n: int = 1):
        self.cpu_gpu_transfers += n

    def mark_plate_skipped(self, n: int = 1):
        self.plate_skipped += n

    def mark_ocr_skipped(self, n: int = 1):
        self.ocr_skipped += n

    def mark_vehicle_call(self, n: int = 1):
        self.vehicle_calls += n

    def mark_vehicle_skipped(self, n: int = 1):
        self.vehicle_skipped += n

    def mark_gate_filtered(self, n: int = 1):
        self.plate_gate_filtered += n

    def mark_gate_reject(self, reason: str, n: int = 1):
        self.plate_gate_filtered += n
        if reason == "conf":
            self.gate_reject_conf += n
        elif reason == "size":
            self.gate_reject_size += n
        elif reason == "aspect":
            self.gate_reject_aspect += n
        elif reason == "blur":
            self.gate_reject_blur += n

    def mark_ocr_rejected(self, n: int = 1):
        self.ocr_rejected_low_conf += n

    def end_frame(self, vehicles: int):
        self.vehicles_per_frame.append(vehicles)
        self.plate_calls_per_frame.append(self._frame_plate_calls)
        self.ocr_calls_per_frame.append(self._frame_ocr_calls)
        self._frame_plate_calls = 0
        self._frame_ocr_calls = 0
        if self.report_interval and self.frame_count % self.report_interval == 0:
            print(self.report())

    def _avg_ms(self, stage: str) -> float:
        vals = self._timings.get(stage, [])
        return sum(vals) / len(vals) * 1000 if vals else 0.0

    def _avg_ms_per_call(self, stage: str, calls: int) -> float:
        vals = self._timings.get(stage, [])
        if not vals or not calls:
            return 0.0
        return sum(vals) / calls * 1000

    def _avg(self, vals: list) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def get_fps(self):
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                return (len(self._frame_times) - 1) / span
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0
        return self.frame_count / elapsed

    def report(self) -> str:
        return (
            f"\n[BASELINE] frames={self.frame_count} "
            f"FPS={self.get_fps():.1f} total={self._avg_ms('total'):.0f}ms\n"
            f"  decode={self._avg_ms('decode'):.0f}ms "
            f"vehicle={self._avg_ms('vehicle_detection'):.0f}ms "
            f"track={self._avg_ms('tracking'):.0f}ms "
            f"plate={self._avg_ms('plate_detection'):.0f}ms\n"
            f"  gate={self._avg_ms('quality_gate'):.0f}ms "
            f"ocr={self._avg_ms('ocr'):.0f}ms "
            f"render={self._avg_ms('rendering'):.0f}ms\n"
            f"  vehicles/frame={self._avg(self.vehicles_per_frame):.1f} "
            f"plate_calls/frame={self._avg(self.plate_calls_per_frame):.1f} "
            f"ocr_calls/frame={self._avg(self.ocr_calls_per_frame):.1f}\n"
            f"  cpu_gpu_transfers={self.cpu_gpu_transfers}\n"
            f"  skipped plate={self.plate_skipped} ocr={self.ocr_skipped}\n"
            f"  vehicle calls={self.vehicle_calls} skipped={self.vehicle_skipped} "
            f"per_call={self._avg_ms_per_call('vehicle_detection', self.vehicle_calls):.0f}ms\n"
            f"  gate_filtered={self.plate_gate_filtered} "
            f"(conf={self.gate_reject_conf} size={self.gate_reject_size} "
            f"aspect={self.gate_reject_aspect} blur={self.gate_reject_blur}) "
            f"ocr_rejected_low_conf={self.ocr_rejected_low_conf}"
        )