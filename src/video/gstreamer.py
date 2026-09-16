import shutil
import subprocess

from config import settings

DGPU_DECODERS = {
    "h264": ["nvv4l2decoder", "nvh264dec"],
    "h265": ["nvv4l2decoder", "nvh265dec"],
    "hevc": ["nvv4l2decoder", "nvh265dec"],
}

JETSON_DECODERS = {
    "h264": ["nvv4l2decoder"],
    "h265": ["nvv4l2decoder"],
    "hevc": ["nvv4l2decoder"],
}


def _codec_of(source: str) -> str:
    s = source.lower()
    if "hevc" in s or "h265" in s or "265" in s:
        return "h265"
    return "h264"


def _have_plugin(name: str) -> bool:
    if shutil.which("gst-inspect-1.0") is None:
        return False
    try:
        r = subprocess.run(["gst-inspect-1.0", name],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def pick_decoder(codec: str = "h264", jetson: bool = False) -> str | None:
    table = JETSON_DECODERS if jetson else DGPU_DECODERS
    for decoder in table.get(codec, table["h264"]):
        if _have_plugin(decoder):
            return decoder
    return None


def build_nvdec_pipeline(source: str, jetson: bool = False) -> tuple[str, str]:
    codec = _codec_of(source)
    decoder = pick_decoder(codec, jetson)
    if decoder is None:
        table = JETSON_DECODERS if jetson else DGPU_DECODERS
        decoder = table[codec][0]
    is_rtsp = source.startswith("rtsp://")
    if is_rtsp:
        demux = f"rtspsrc location={source} latency=100 ! rtph264depay ! h264parse ! {decoder}"
    elif codec == "h264":
        demux = (f"filesrc location={source} ! qtdemux ! h264parse ! "
                 f"{decoder}")
    else:
        demux = (f"filesrc location={source} ! qtdemux ! h265parse ! "
                 f"{decoder}")
    pipeline = (f"{demux} ! nvvidconv ! video/x-raw,format=BGR ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink")
    return pipeline, decoder


def probe_report(hardware_decode: bool | None = None) -> str:
    import cv2
    build = cv2.getBuildInformation()
    gstreamer = "GStreamer:                   YES" in build
    decoders = {d: _have_plugin(d)
                for d in ["nvv4l2decoder", "nvh264dec", "nvh265dec"]}
    requested = settings.video.hardware_decode if hardware_decode is None else hardware_decode
    return (f"[NVDEC] opencv_gstreamer={gstreamer} decoders={decoders} "
            f"hw_decode_requested={requested}")
