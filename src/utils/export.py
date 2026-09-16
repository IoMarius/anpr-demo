import os

import torch


def ensure_engine(pt_path: str, imgsz: int) -> str | None:
    """Return a TensorRT engine path for a .pt model, building it if needed.

    The engine is cached beside the .pt file (<stem>.engine) so the slow
    export+build happens once. Any failure (no CUDA, no TensorRT, export
    error) returns None and the caller falls back to the .pt weights.
    """
    if not torch.cuda.is_available():
        print("[TRT] CUDA unavailable, staying on .pt weights")
        return None
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[TRT] ultralytics import failed ({e}), staying on .pt")
        return None

    stem, _ = os.path.splitext(pt_path)
    engine_path = stem + ".engine"
    if os.path.isfile(engine_path):
        return engine_path
    try:
        print(f"[TRT] building FP16 engine for {pt_path} "
              f"(imgsz={imgsz}, one-time cost, may take minutes)...")
        model = YOLO(pt_path)
        model.export(format="engine", imgsz=imgsz, quantize=16,
                     device="cuda:0", verbose=False)
        if os.path.isfile(engine_path):
            print(f"[TRT] engine ready: {engine_path}")
            return engine_path
        print(f"[TRT] export finished but {engine_path} not found, "
              f"staying on .pt")
        return None
    except Exception as e:
        print(f"[TRT] export failed ({str(e)[:200]}), staying on .pt")
        return None
