# ANPR Demo — Refined Plan (v2)

Goal is still the same: prove the **architecture**, not build a production ANPR system. One Python project, structured as if each stage could later be split out.

---

## 1. Architecture (unchanged)

```text
VIDEO
  |
  v
Frame Reader
  |
  v
Vehicle Detector --> GPU
  |
  v
Plate Detector ----> GPU   (NEW: separate stage, see Section 2)
  |
  v
Tracker ------------> CPU
  |
  v plate crops
Quality Gate --------> CPU
  |
  v
Recognition / OCR --> GPU
  |
  v
Fusion --------------> CPU
  |
  v
PlateEvent
  |
  +--> Display
  +--> Events
```

Stage separation is the core architectural decision. Everything below refines *how* to build each stage for a demo that actually works.

---

## 2. Key correction: plate detection is two-stage, not one model

Standard pretrained YOLO (COCO) has no license-plate class — it only gives vehicle boxes. A single "vehicle/plate" detector doesn't exist off the shelf.

**Fixed approach:**

1. COCO YOLO detects vehicles (works out of the box, no training needed).
2. Crop to each vehicle's bounding box.
3. Run a **separate pretrained plate detector** (open license-plate YOLO weights, e.g. from Roboflow Universe / HuggingFace) only inside that crop.

This is better than a single global detector anyway — smaller search area, fewer false positives from background clutter, less wasted GPU work. Build it this way from day one rather than retrofitting later.

---

## 3. OCR accuracy — set expectations, lean on fusion

Generic OCR (Tesseract, EasyOCR) on plate crops will be mediocre on any single frame — that's expected, not a bug to chase. The fusion stage exists specifically to compensate for this: we don't need any one frame to be right, we need the pipeline to be right often enough across N observations of the same track.

**Fusion upgrade over plain majority vote:** do per-character positional voting across same-length OCR observations (fall back to string-level vote when lengths disagree). Cheap (~20 lines), meaningfully more robust than naive majority vote, and handles the common failure mode of one dropped/misread character per frame.

---

## 4. Build order — correctness before optimization

Don't start with frame-skipping. Sequence it so the optimization is *demonstrated*, not just asserted:

1. **Correctness first:** run detection every frame. Get the pipeline producing correct `PlateEvent`s end to end.
2. **Measure:** use the metrics module (built alongside everything else) to get real detector/recognizer latency numbers.
3. **Then optimize:** introduce frame-skipping (detect every Nth frame, track bridges the gaps) as a deliberate change, and show the before/after FPS numbers.

Same logic for the quality gate:
- v1: size threshold + detector confidence threshold (cheap, effective).
- Add if time allows: blur rejection via Laplacian variance (~3 lines).
- Skip for v1: angle/oblique rejection — needs plate corner/keypoint estimation, real extra work for a filter that mostly matters at highway viewing angles.

---

## 5. Track-centric processing (unchanged)

The track, not the frame, is the central object:

```python
TrackState(
    id=42,
    first_seen=...,
    last_seen=...,
    plate_observations=[...],
    vehicle_class="car",
)
```

Each track accumulates plate observations over time; fusion collapses them into one confident result.

---

## 6. PlateEvent contract (unchanged)

```json
{
    "track_id": 42,
    "plate": "ABC123",
    "confidence": 0.95,
    "first_seen": "...",
    "last_seen": "...",
    "camera_id": "demo-camera"
}
```

This is the only artifact the outside world sees. Everything upstream (detections, raw OCR reads) stays internal. This is what lets the pipeline later publish to Kafka/RabbitMQ/a DB without touching the vision code.

---

## 7. Video source — a decision, not an implementation detail

This matters more than most of the code. Typical highway/traffic-cam footage has plates too small and too oblique for reliable OCR at any quality level, regardless of how good the pipeline is.

**Use parking-lot, toll-booth, or gate/entrance footage** — closer, more frontal, larger plates. Pick this before writing a line of recognition code; it's the single biggest factor in whether the live demo looks convincing.

---

## 8. Metrics (unchanged, still build from day one)

```text
Input FPS
Processed FPS
Detection FPS
Recognition count
GPU inference time
CPU processing time
End-to-end latency
Active tracks
Confirmed plates
Rejected observations
```

This is also what makes Section 4's "measure before optimizing" step possible — the frame-skipping win should show up as a before/after in these numbers.

---

## 9. Project structure

```text
anpr-demo/
|
+-- main.py
|
+-- video/
|   +-- source.py
|
+-- detection/
|   +-- vehicle_detector.py
|   +-- plate_detector.py        # separate stage, see Section 2
|
+-- tracking/
|   +-- tracker.py
|
+-- quality/
|   +-- gate.py                  # size + confidence (+ blur if time allows)
|
+-- recognition/
|   +-- recognizer.py
|
+-- fusion/
|   +-- plate_fusion.py          # per-character voting, see Section 3
|
+-- events/
|   +-- models.py
|
+-- visualization/
|   +-- renderer.py
|
+-- config.py
+-- metrics.py
```

Each module is still a stand-in for a future separable processing stage — that hasn't changed, only the detection module split in two.

---

## 10. GPU/CPU boundary — note, not infrastructure

Keep this as a design comment/convention (GPU: vehicle detect, plate detect, recognition; CPU: everything else), not something to build process separation around. In a single process on one machine it isn't really enforceable or measurable as a "boundary" — treat it as intent to preserve for when the pipeline is later split out.

---

## 11. Explicitly out of scope for v1

- distributed processing
- Kafka / RabbitMQ
- microservices
- Kubernetes
- databases
- cloud infrastructure
- multiple cameras
- custom-trained models
- sophisticated vehicle re-identification
- angle/oblique plate rejection

These remain scale-out concerns for after the single-process demo proves the architecture — useful as a "here's where this goes next" slide, not implementation work now.

---

## 12. v1 success criterion

> Given a traffic (parking-lot/gate-style) video, the system processes it near real time, tracks vehicles, recognizes plates from multiple frames, fuses the noisy recognitions into one result per vehicle, and emits a `PlateEvent` with confidence and timing metrics — with detection running every frame first, then optimized with measured frame-skipping.

If this works cleanly, it's the foundation for the 100/1,000-camera, edge-vs-central, event-streaming conversation — without throwing away the demo architecture.

---

## Cut-line summary

**Build for v1:**
vehicle detector -> plate detector (two-stage) -> ByteTrack -> quality gate (size + confidence) -> OCR -> per-character-vote fusion -> PlateEvent -> overlay + metrics HUD.

**Defer:**
frame-skipping (add after measuring you need it), blur/angle rejection beyond basics, GPU/CPU process separation, everything in Section 11.
