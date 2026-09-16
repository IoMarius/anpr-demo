import torch


def is_cuda_oom(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return ("out of memory" in msg or "cudamalloc" in msg
            or "cudart" in msg or "cuda_error_out_of_memory" in msg
            or "acceleratorerror" in type(exc).__name__.lower())


def cuda_mem_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        free, total = torch.cuda.mem_get_info()
        cap = torch.cuda.get_device_capability()
    except Exception as e:
        return {"available": True, "error": str(e)[:200]}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "capability": f"{cap[0]}.{cap[1]}",
        "free_mb": free // (1 << 20),
        "total_mb": total // (1 << 20),
    }


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


def touch_cuda(device: str) -> bool:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device=device)
        # Fixed-shape inference dominates; autotune cudnn kernels once.
        torch.backends.cudnn.benchmark = True
        return True
    except Exception as e:
        print(f"[GPU] CUDA context creation on {device} failed "
              f"({str(e)[:150]}), falling back to cpu")
        return False
