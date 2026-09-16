import shutil
import subprocess

from config import settings

DGPU_DECODERS = {
    "h264": ["nvv4l2decoder", "nvh264dec"],
    "h265": ["nvv4l2decoder", "nvh265dec"],
    "hevc": ["nvv4l2decoder", "nvh265dec"],
    "av1": ["nvv4l2decoder"],
    "av01": ["nvv4l2decoder"],
}

JETSON_DECODERS = {
    "h264": ["nvv4l2decoder"],
    "h265": ["nvv4l2decoder"],
    "hevc": ["nvv4l2decoder"],
    "av1": ["nvv4l2decoder"],
    "av01": ["nvv4l2decoder"],
}

PARSER = {"h264": "h264parse", "h265": "h265parse", "hevc": "h265parse",
          "av1": "av1parse", "av01": "av1parse"}


def _codec_of(source: str) -> str:
    s = source.lower()
    if "av1" in s or "av01" in s:
        return "av1"
    if "hevc" in s or "h265" in s or "265" in s:
        return "h265"
    return "h264"


def probe_codec(source: str) -> str:
    import cv2
    import os
    if not os.path.isfile(source):
        return _codec_of(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return _codec_of(source)
    try:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        tag = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        t = tag.lower()
        if "av01" in t or "av1" in t:
            return "av1"
        if "hev1" in t or "hvc1" in t or "hevc" in t:
            return "h265"
        if "avc1" in t or "h264" in t:
            return "h264"
    except Exception:
        pass
    finally:
        cap.release()
    return _codec_of(source)


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


def build_nvdec_pipeline(source: str, jetson: bool = False,
                         codec: str | None = None) -> tuple[str | None, str | None]:
    codec = codec or _codec_of(source)
    decoder = pick_decoder(codec, jetson)
    if decoder is None:
        return None, None
    parser = PARSER.get(codec, "h264parse")
    is_rtsp = source.startswith("rtsp://")
    if is_rtsp:
        depay = {"h264": "rtph264depay", "h265": "rtph265depay",
                 "hevc": "rtph265depay"}.get(codec, "rtpav1depay")
        demux = (f"rtspsrc location={source} latency=100 ! {depay} ! "
                 f"{parser} ! {decoder}")
    else:
        demux = (f"filesrc location={source} ! qtdemux ! {parser} ! "
                 f"{decoder}")
    pipeline = (f"{demux} ! nvvidconv ! video/x-raw,format=BGR ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink")
    return pipeline, decoder


def apply_ffmpeg_hwaccel(cap) -> bool:
    import cv2
    accel = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
    if accel is None:
        return False
    try:
        cap.set(accel, getattr(cv2, "VIDEO_ACCELERATION_ANY", 1))
        dev = getattr(cv2, "CAP_PROP_HW_DEVICE", None)
        if dev is not None:
            cap.set(dev, 0)
        return True
    except Exception:
        return False


def probe_report(hardware_decode: bool | None = None) -> str:
    import cv2
    build = cv2.getBuildInformation()
    gstreamer = "GStreamer:                   YES" in build
    decoders = {d: _have_plugin(d)
                for d in ["nvv4l2decoder", "nvh264dec", "nvh265dec"]}
    requested = settings.video.hardware_decode if hardware_decode is None else hardware_decode
    return (f"[NVDEC] opencv_gstreamer={gstreamer} decoders={decoders} "
            f"hw_decode_requested={requested}")
