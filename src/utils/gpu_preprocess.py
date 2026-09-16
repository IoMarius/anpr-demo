import torch
import torch.nn.functional as F

IMGSZ = 640
STRIDE = 32


def letterbox_gpu(hwc: torch.Tensor, imgsz: int = IMGSZ) -> tuple:
    h, w = hwc.shape[0], hwc.shape[1]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    chw = hwc.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(chw, size=(nh, nw), mode="bilinear",
                            align_corners=False)
    pad_h, pad_w = imgsz - nh, imgsz - nw
    top, left = pad_h // 2, pad_w // 2
    padded = F.pad(resized, (left, pad_w - left, top, pad_h - top),
                   value=114.0)
    rgb = padded[:, [2, 1, 0], :, :] / 255.0
    return rgb.contiguous(), scale, left, top


def unscale_boxes(xyxy: torch.Tensor, scale: float, pad_w: int,
                  pad_h: int, orig_w: int, orig_h: int) -> torch.Tensor:
    out = xyxy.clone()
    out[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / scale
    out[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, orig_h)
    return out
