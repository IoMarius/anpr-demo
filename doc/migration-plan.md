# ANPR GPU Pipeline Migration Plan

## 1. Objective

Refactor the current ANPR pipeline to maximize useful GPU utilization while minimizing unnecessary CPU/GPU memory transfers.

The target architecture is:

```text
Video source
    │
    ▼
NVIDIA hardware decode
    │
    ▼
GPU-resident frame
    │
    ├── Vehicle detection
    │
    ├── GPU vehicle crops
    │
    ├── Batched plate detection
    │
    ├── GPU plate crops
    │
    └── OCR
            │
            ▼
       CPU metadata/state
            │
            ├── tracking
            ├── fusion
            ├── events
            └── visualization
```

The goal is **not** to force every operation onto the GPU. Small stateful operations should remain on CPU. The goal is to keep image/tensor processing on the GPU and avoid unnecessary transfers.

---

# 2. Current Architecture

The current pipeline is approximately:

```text
VideoCapture
    │
    ▼
CPU NumPy 4K frame
    │
    ▼
Vehicle YOLO
    │
    ├── GPU inference
    └── GPU → CPU results
    │
    ▼
CPU vehicle boxes
    │
    ├── Vehicle 1 crop → Plate YOLO → CPU
    ├── Vehicle 2 crop → Plate YOLO → CPU
    ├── Vehicle 3 crop → Plate YOLO → CPU
    └── ...
    │
    ▼
CPU plate crop
    │
    ▼
EasyOCR
    │
    ├── GPU inference
    └── CPU result
    │
    ▼
CPU rendering
    │
    ▼
CPU JPEG encoding
    │
    ▼
Flask
```

The main problems are:

1. Video decoding is not explicitly GPU accelerated.
2. The source frame is CPU-resident.
3. Images repeatedly cross CPU/GPU boundaries.
4. Plate detection runs once per vehicle per frame.
5. OCR runs synchronously for every accepted plate.
6. Plate detection is not batched.
7. OCR is not batched.
8. Vehicle tracking may be duplicated between Ultralytics and `TrackManager`.
9. The main processing loop is completely sequential.
10. The visualization path copies and JPEG-encodes 4K frames on CPU.

---

# 3. Phase 0: Establish a Baseline

Before changing the architecture, measure the existing implementation.

Add timings for:

```text
decode
vehicle detection
vehicle tracking
plate detection
quality gate
OCR
rendering
JPEG encoding
total frame time
```

Also record:

```text
FPS
number of active vehicles
number of plate detection calls
number of OCR calls
GPU utilization
GPU memory usage
CPU utilization
```

For every frame, record enough information to answer:

```text
How many vehicles?
How many plate inferences?
How many OCR calls?
How long did each stage take?
```

Do not optimize based solely on the current 5 FPS figure.

## Acceptance criterion

A baseline report should look approximately like:

```text
Average FPS:             5.1
Average frame time:    196 ms

Decode:                 22 ms
Vehicle detection:      61 ms
Tracking:                4 ms
Plate detection:        67 ms
Quality gate:            3 ms
OCR:                    25 ms
Rendering:               8 ms
JPEG:                    6 ms

Vehicles/frame:         3.7
Plate detections/frame: 3.7
OCR calls/frame:        1.8
```

The actual numbers will determine where optimization effort should go.

---

# 4. Phase 1: Verify GPU Configuration

Verify that both YOLO models are actually running on CUDA.

Vehicle detector:

```python
device = "cuda:0"
```

Plate detector:

```python
device = "cuda:0"
```

EasyOCR:

```python
gpu=True
```

Verify the actual PyTorch environment:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Verify the YOLO model device as well.

Do not assume that passing `device="cuda"` guarantees every part of the operation is GPU-resident.

---

# 5. Phase 2: Remove Unnecessary Duplicate Tracking

Current vehicle detection uses:

```python
self.model.track(
    frame,
    persist=True,
    ...
)
```

while the application also has:

```python
tracker = TrackManager()
```

and:

```python
active_tracks = tracker.update(...)
```

Investigate `TrackManager` before making a decision.

There should ultimately be one clear ownership model:

```text
Option A

YOLO detection
    ↓
TrackManager
```

or:

```text
Option B

YOLO tracking
    ↓
Application state/fusion manager
```

Avoid:

```text
YOLO detection
    ↓
Ultralytics tracker
    ↓
TrackManager
```

unless both layers have a clearly necessary purpose.

## Acceptance criterion

There should be one authoritative vehicle track ID and one authoritative vehicle tracking mechanism.

---

# 6. Phase 3: Introduce an Explicit GPU Frame Representation

The next architectural change is to stop treating the CPU NumPy array as the universal representation of a frame.

Introduce a distinction between:

```text
CPU frame
GPU frame
```

For example:

```python
CPUFrame
GPUFrame
```

or equivalent internal conventions.

The important rule is:

> Image data should remain on GPU once it enters the GPU processing pipeline whenever practical.

Avoid patterns such as:

```python
gpu_tensor -> numpy -> gpu_tensor
```

for image data.

Small metadata such as:

```text
bounding boxes
track IDs
confidence values
```

can remain CPU-side.

---

# 7. Phase 4: Replace CPU Video Decode with NVIDIA Hardware Decode

Replace:

```python
cv2.VideoCapture(source_path)
```

with an explicit GStreamer pipeline using NVIDIA hardware decoding.

The desired conceptual pipeline is:

```text
H264/H265
    ↓
demux
    ↓
parser
    ↓
NVIDIA hardware decoder
    ↓
GPU/device memory
```

The exact GStreamer pipeline depends on:

* NVIDIA GPU vs Jetson
* operating system
* GStreamer installation
* codec
* RTSP/file source
* CUDA/GStreamer integration

Do not assume `CAP_GSTREAMER` alone provides hardware decoding.

The pipeline must explicitly use the appropriate NVIDIA decoder.

## Acceptance criterion

Verify with profiling/tools that:

```text
video decode = NVIDIA hardware decoder
```

rather than simply assuming it from the API.

---

# 8. Phase 5: Avoid Immediate GPU → CPU Conversion

A critical requirement is:

```text
NVDEC
  ↓
GPU frame
  ↓
GPU processing
```

not:

```text
NVDEC
  ↓
GPU frame
  ↓
CPU NumPy
  ↓
GPU upload
```

The second architecture largely defeats the purpose of GPU decoding.

The first version should therefore establish a controlled interface for obtaining GPU-resident frames.

---

# 9. Phase 6: Refactor Vehicle Detection

Vehicle detection should consume the GPU-resident frame.

Target:

```text
GPU frame
    ↓
Vehicle YOLO
    ↓
GPU inference
    ↓
vehicle metadata
```

Keep only the metadata required by application logic on CPU:

```python
{
    "bbox": (...),
    "track_id": ...,
    "class_id": ...
}
```

Do not transfer image tensors back to CPU.

Small bounding-box transfers are acceptable.

The optimization target is:

```text
Do not transfer full 4K frames unnecessarily.
```

---

# 10. Phase 7: Move Vehicle Cropping Toward GPU

Current:

```python
crop = frame[y1:y2, x1:x2]
```

operates on a CPU NumPy frame.

Target:

```text
GPU frame
    │
    ├── bbox 1 → GPU crop
    ├── bbox 2 → GPU crop
    ├── bbox 3 → GPU crop
    └── ...
```

Use GPU tensor slicing/cropping or an equivalent GPU-native operation.

Avoid:

```text
GPU frame
    ↓
CPU NumPy frame
    ↓
CPU crop
    ↓
GPU upload
```

---

# 11. Phase 8: Batch Plate Detection

This is one of the highest-priority changes.

Current:

```python
for track_id, v_bbox in active_tracks:
    plate_det.detect(frame, v_bbox)
```

Target:

```text
active vehicles
      │
      ▼
GPU vehicle crops
      │
      ▼
batched Plate YOLO inference
      │
      ▼
plate results
```

Instead of:

```text
plate inference
plate inference
plate inference
plate inference
```

perform:

```text
one batched inference
```

where practical.

Example conceptual API:

```python
vehicle_crops = ...
results = plate_model(
    vehicle_crops,
    device="cuda:0"
)
```

The implementation should preserve the original vehicle ID so each plate result can be mapped back to the correct track.

---

# 12. Phase 9: Stop Running Plate Detection on Every Vehicle Every Frame

This is an algorithmic optimization, not merely a GPU optimization.

Current:

```text
every vehicle
every frame
    ↓
plate detector
```

Target:

```text
vehicle track
    │
    ├── recently checked → skip
    │
    └── eligible → plate detection
```

Possible scheduling strategy:

```text
Frame N:
    plate detection

Frame N+1:
    track only

Frame N+2:
    track only

Frame N+3:
    plate detection
```

The exact interval should be configurable.

For example:

```python
plate_detection_interval = 3
```

or a time-based interval.

The system should continue collecting multiple observations for fusion.

---

# 13. Phase 10: Introduce OCR Scheduling

Current:

```python
if gate.is_valid(...):
    ocr.recognize(plate_crop)
```

Target:

```text
plate candidate
      │
      ▼
quality gate
      │
      ├── poor → discard
      │
      └── good
           │
           ▼
      OCR eligibility
           │
           ├── recently OCR'd → skip
           │
           └── eligible → OCR
```

A track does not necessarily need OCR on every good plate observation.

Useful rules can include:

```text
OCR only once every N milliseconds
OCR only if plate quality improved
OCR again if previous confidence was low
OCR again if plate geometry changed significantly
Stop OCR after sufficiently confident recognition
```

The existing fusion system should remain responsible for combining observations.

---

# 14. Phase 11: Batch OCR

Where supported by the OCR implementation, change:

```text
plate 1 → OCR
plate 2 → OCR
plate 3 → OCR
```

to:

```text
plate crops
    ↓
GPU OCR batch
    ↓
results
```

If EasyOCR cannot provide the batching/control required by the desired architecture, evaluate an alternative OCR implementation.

Do not replace EasyOCR merely for the sake of replacing it. Benchmark it first.

---

# 15. Phase 12: Keep Application State on CPU

Do not attempt to GPU-accelerate everything.

Keep these CPU-side:

```text
TrackManager state
track IDs
timestamps
PlateFusion
PlateEvent
metrics
Python dictionaries/lists
logging
business logic
```

These are not significant GPU workloads.

The target boundary should be:

```text
GPU:
    images
    tensors
    preprocessing
    detection
    OCR

CPU:
    metadata
    state
    events
    business logic
```

---

# 16. Phase 13: GPU Image Preprocessing

Once the frame is GPU-resident, investigate moving expensive image operations from CPU to GPU.

Potential candidates:

```text
resize
color conversion
normalization
cropping
letterboxing
quality measurements
```

Do not blindly rewrite OpenCV operations.

First measure them.

If preprocessing takes only 1-2 ms while inference takes 80 ms, moving it provides little value.

---

# 17. Phase 14: Review QualityGate

Inspect:

```text
quality/gate.py
```

Determine whether it performs:

* simple numeric checks
* blur detection
* contrast analysis
* image resizing
* grayscale conversion
* edge detection
* other CPU image processing

Simple checks stay on CPU.

Image-heavy operations should be candidates for GPU implementation if profiling shows they matter.

---

# 18. Phase 15: Optimize Visualization Separately

Current:

```python
render_frame = HUD.draw(frame.copy(), ...)
```

followed by:

```python
latest_frame = render_frame.copy()
```

and:

```python
cv2.imencode('.jpg', latest_frame)
```

Keep this separate from the inference pipeline.

The inference pipeline should not wait unnecessarily for browser rendering.

Target:

```text
                    ┌── GPU inference
                    │
GPU frame ──────────┤
                    │
                    └── CPU visualization branch
                              │
                              ▼
                           JPEG
                              │
                              ▼
                           Flask
```

For live video, visualization should consume the latest available processed frame rather than blocking inference.

Later, consider hardware encoding or a GStreamer-based output pipeline if JPEG encoding becomes measurable.

---

# 19. Phase 16: Redesign the Frame Queue for Live Video

The current queue:

```python
queue.Queue(maxsize=queue_size)
```

preserves frames until they are consumed.

For live ANPR, consider a latest-frame strategy:

```text
decoder
   │
   ▼
latest frame
   │
   ▼
processor
```

If processing falls behind:

```text
old frames are discarded
```

rather than accumulating latency.

For recorded-video benchmarking, retain the current sequential semantics if processing every frame is required.

This behavior should be configurable:

```python
mode = "live"
mode = "offline"
```

---

# 20. Phase 17: Introduce Pipeline Stages

Once the basic GPU path works, separate the pipeline into logical stages:

```text
Source
  ↓
Vehicle Detection
  ↓
Tracking
  ↓
Plate Scheduling
  ↓
Plate Detection
  ↓
Quality Gate
  ↓
OCR Scheduling
  ↓
OCR
  ↓
Fusion
  ↓
Events
```

This makes it possible to later run independent workers.

For example:

```text
Decode ────────► Detection
                    │
                    ▼
                 Tracking
                    │
                    ▼
              Plate Detection
                    │
                    ▼
                  OCR
                    │
                    ▼
                  Fusion
```

Do not introduce multiprocessing/threading everywhere immediately. First make the GPU data path correct.

---

# 21. Phase 18: Consider TensorRT

Once the model/input pipeline is stable, benchmark TensorRT versions of:

```text
vehicle detector
plate detector
OCR model, if supported
```

Potential progression:

```text
PyTorch YOLO
    ↓
FP16 CUDA
    ↓
TensorRT FP16
    ↓
TensorRT optimized engine
```

Start with FP16 if the models and hardware support it.

Only investigate INT8 after establishing whether the accuracy tradeoff is acceptable for ANPR.

---

# 22. Phase 19: Consider DeepStream

DeepStream should be evaluated after the pipeline behavior is understood.

It becomes particularly attractive if the final requirements include:

```text
multiple cameras
hardware decode
GPU-resident buffers
batched inference
tracking
metadata propagation
multiple streams
hardware encoding
high throughput
```

A possible final architecture:

```text
RTSP
 │
 ▼
DeepStream / GStreamer
 │
 ▼
NVDEC
 │
 ▼
GPU buffers
 │
 ▼
Vehicle detector
 │
 ▼
Tracker
 │
 ▼
Plate processing
 │
 ▼
OCR
 │
 ▼
Metadata
 │
 ├── events
 ├── database
 └── visualization
```

Do not migrate to DeepStream before the simpler GPU pipeline has been profiled. Otherwise it becomes difficult to determine whether improvements came from:

* hardware decoding
* batching
* TensorRT
* scheduling
* buffer residency
* tracking changes
* or simply doing less work.

---

# 23. Proposed Configuration

Introduce explicit configuration for the major decisions:

```python
gpu:
    enabled: true
    device: "cuda:0"
    fp16: true

video:
    hardware_decode: true
    gpu_frames: true

detection:
    vehicle_interval: 1
    plate_interval: 3

recognition:
    gpu: true
    interval_ms: 300
    batch_size: 8

visualization:
    enabled: true
    output_width: 1280
    output_height: 720
    jpeg_quality: 80
```

The exact configuration format should follow the project's existing settings architecture.

---

# 24. Implementation Order

Implement in this order:

```text
1. Instrument current pipeline
        ↓
2. Verify CUDA
        ↓
3. Verify YOLO GPU execution
        ↓
4. Inspect/remove duplicate tracking
        ↓
5. Introduce GPU frame representation
        ↓
6. Implement NVIDIA hardware decode
        ↓
7. Keep decoded frames on GPU
        ↓
8. Feed GPU frames into vehicle detection
        ↓
9. GPU vehicle cropping
        ↓
10. Batch plate detection
        ↓
11. Reduce plate detection frequency
        ↓
12. Add OCR scheduling
        ↓
13. Batch OCR
        ↓
14. Optimize QualityGate where justified
        ↓
15. Separate visualization from inference
        ↓
16. Optimize live frame buffering
        ↓
17. Benchmark TensorRT
        ↓
18. Evaluate DeepStream
```

---

# 25. What Should Not Be Changed Yet

Do not initially:

* rewrite every OpenCV operation in CUDA
* move Python dictionaries to GPU
* move tracking state to GPU
* introduce multiprocessing everywhere
* replace EasyOCR immediately
* migrate the entire application to DeepStream
* optimize JPEG encoding before measuring it
* optimize tiny `.cpu()` metadata transfers
* assume GPU utilization percentage alone represents performance

The primary target is **large image/tensor transfers and unnecessary inference**.

---

# 26. Success Criteria

The migration should be evaluated using more than FPS.

Measure:

```text
Throughput
Latency
GPU utilization
GPU memory
CPU utilization
Decode time
Vehicle inference time
Plate inference time
OCR time
Number of inferences/frame
Number of OCR calls/frame
CPU ↔ GPU transfers
Recognition accuracy
Plate-event accuracy
Live-video latency
```

The final comparison should be:

```text
                       CURRENT       TARGET
FPS
Frame latency
CPU utilization
GPU utilization
Decode time
Vehicle inference
Plate inference
OCR
Plate calls/frame
OCR calls/frame
Recognition accuracy
Live latency
```

The most important metric is not simply:

```text
GPU utilization = 95%
```

A better system is one that produces the required ANPR accuracy and throughput with sensible latency and resource usage.

---

# 27. First Implementation Milestone

The first milestone should be deliberately small:

```text
Current video source
        ↓
NVIDIA hardware decode
        ↓
GPU-resident frame
        ↓
Vehicle YOLO on GPU
        ↓
CPU vehicle metadata
```

Nothing else needs to change yet.

Once this works and is benchmarked, proceed to:

```text
GPU frame
    ↓
GPU vehicle crops
    ↓
batched Plate YOLO
```

Then:

```text
plate crops
    ↓
scheduled/batched OCR
```

This incremental approach makes each performance improvement measurable and prevents the entire ANPR system from becoming difficult to debug.

---

# 28. Final Target

The intended final design is:

```text
                         NVIDIA GPU
                    ┌─────────────────────┐
                    │                     │
RTSP/File ──► NVDEC │ GPU Frame           │
                    │    │                │
                    │    ▼                │
                    │ Vehicle YOLO         │
                    │    │                │
                    │    ▼                │
                    │ GPU vehicle crops   │
                    │    │                │
                    │    ▼                │
                    │ Batched Plate YOLO  │
                    │    │                │
                    │    ▼                │
                    │ GPU plate crops     │
                    │    │                │
                    │    ▼                │
                    │ OCR                 │
                    │                     │
                    └─────────┬───────────┘
                              │
                       small metadata
                              │
                              ▼
                           CPU
                    ┌─────────────────────┐
                    │ Tracking state      │
                    │ Quality decisions   │
                    │ Plate fusion        │
                    │ Events              │
                    │ Metrics              │
                    │ Visualization       │
                    │ HTTP output         │
                    └─────────────────────┘
```

The central design rule is:

> **Keep pixels on the GPU. Move metadata to the CPU. Do expensive inference in batches. Do not perform inference more often than the application actually needs.**
