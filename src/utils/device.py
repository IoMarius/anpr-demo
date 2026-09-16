import torch


def resolve_device(requested: str, gpu_enabled: bool = True) -> str:
    if gpu_enabled and requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    if requested.startswith("cuda"):
        print(f"[GPU] {requested} requested but CUDA unavailable, falling back to cpu")
        return "cpu"
    return requested


def log_gpu_status(context: str = ""):
    available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if available else "none"
    print(f"[GPU]{' ' + context if context else ''} cuda_available={available} device={name}")
