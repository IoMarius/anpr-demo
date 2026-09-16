import shutil
import subprocess

import torch

from config import settings
from utils.device import cuda_mem_info


def _nvidia_smi_line() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi: not found"
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        return "nvidia-smi: " + (r.stdout.strip() or "no output")
    except Exception as e:
        return f"nvidia-smi: error {str(e)[:150]}"


def _cv2_flags() -> str:
    import cv2
    build = cv2.getBuildInformation()
    return ("opencv_gstreamer="
            + str("GStreamer:                   YES" in build)
            + " ffmpeg=" + str("FFMPEG:                      YES" in build))


def preflight_report(source: str) -> str:
    mem = cuda_mem_info()
    lines = [
        "[PREFLIGHT] source=" + source,
        "[PREFLIGHT] torch=" + torch.__version__
        + " cuda_build=" + str(torch.version.cuda),
        "[PREFLIGHT] " + _nvidia_smi_line(),
        "[PREFLIGHT] cuda_mem=" + str(mem),
        "[PREFLIGHT] " + _cv2_flags(),
        "[PREFLIGHT] config gpu.enabled=" + str(settings.gpu.enabled)
        + " device=" + settings.detection.device
        + " fp16=" + str(settings.gpu.fp16)
        + " hw_decode=" + str(settings.video.hardware_decode)
        + " gpu_frames=" + str(settings.video.gpu_frames),
    ]
    if mem.get("available") and mem.get("free_mb", 0) < 512:
        lines.append("[PREFLIGHT] WARNING low CUDA free memory, "
                     "expect GPU->CPU fallbacks")
    return "\n".join(lines)
